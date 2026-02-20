from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from skimage.metrics import structural_similarity


@dataclass
class RegionStats:
    count: int
    total_area: int


def load_image(file: Any) -> np.ndarray:
    """Decode uploaded image file to BGR ndarray."""
    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Bild konnte nicht geladen werden. Bitte PNG/JPG verwenden.")
    return image


def align_images_ecc(ref: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Align test image to reference using ECC affine transform."""
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        200,
        1e-6,
    )

    try:
        cc, warp_matrix = cv2.findTransformECC(
            ref_gray,
            test_gray,
            warp_matrix,
            warp_mode,
            criteria,
        )
        aligned = cv2.warpAffine(
            test,
            warp_matrix,
            (ref.shape[1], ref.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned, warp_matrix, f"Alignment erfolgreich (ECC={cc:.4f})."
    except cv2.error as err:
        return test, None, f"Alignment fehlgeschlagen: {err}. Fallback ohne Alignment."


def align_images_orb(ref: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Align test image to reference using ORB keypoints + homography."""
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=4000)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    kp_test, des_test = orb.detectAndCompute(test_gray, None)

    if des_ref is None or des_test is None:
        return test, None, "Alignment fehlgeschlagen: keine ORB-Features gefunden. Fallback ohne Alignment."

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_test, des_ref)
    if len(matches) < 4:
        return test, None, "Alignment fehlgeschlagen: zu wenige Matches. Fallback ohne Alignment."

    matches = sorted(matches, key=lambda m: m.distance)
    keep_n = max(4, int(len(matches) * 0.6))
    good_matches = matches[:keep_n]

    test_pts = np.float32([kp_test[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    ref_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, inlier_mask = cv2.findHomography(test_pts, ref_pts, cv2.RANSAC, 5.0)
    if homography is None:
        return test, None, "Alignment fehlgeschlagen: Homographie nicht berechenbar. Fallback ohne Alignment."

    aligned = cv2.warpPerspective(
        test,
        homography,
        (ref.shape[1], ref.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    return aligned, homography, f"Alignment erfolgreich (ORB, Inliers={inliers}/{len(good_matches)})."


def align_images(
    ref: np.ndarray, test: np.ndarray, method: str
) -> tuple[np.ndarray, np.ndarray | None, str]:
    if method == "ORB (Homography)":
        return align_images_orb(ref, test)
    return align_images_ecc(ref, test)


def compute_ssim_diff(ref_gray: np.ndarray, test_gray: np.ndarray) -> tuple[float, np.ndarray]:
    score, diff = structural_similarity(ref_gray, test_gray, full=True)
    diff_uint8 = (diff * 255).astype(np.uint8)
    return float(score), diff_uint8


def find_regions(mask: np.ndarray, min_area: int) -> tuple[list[np.ndarray], RegionStats]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    total_area = int(sum(cv2.contourArea(cnt) for cnt in filtered))
    return filtered, RegionStats(count=len(filtered), total_area=total_area)


def make_overlay(
    test_bgr: np.ndarray,
    mask: np.ndarray,
    contours: list[np.ndarray],
    show_boxes: bool = True,
    show_heatmap: bool = True,
) -> np.ndarray:
    overlay = test_bgr.copy()

    if show_heatmap:
        heat = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        active = mask > 0
        overlay[active] = cv2.addWeighted(overlay, 0.5, heat, 0.5, 0)[active]

    if show_boxes:
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return overlay


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def encode_png_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG-Encoding fehlgeschlagen.")
    return buf.tobytes()


def make_csv_bytes(
    score: float,
    region_count: int,
    total_area_px: int,
    area_percent: float,
    threshold: int,
    min_area: int,
    alignment_enabled: bool,
    alignment_method: str,
) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "ssim_score",
            "regions_count",
            "total_area_px",
            "total_area_percent",
            "threshold",
            "min_area_px",
            "alignment_enabled",
            "alignment_method",
        ]
    )
    writer.writerow(
        [
            f"{score:.6f}",
            region_count,
            total_area_px,
            f"{area_percent:.6f}",
            threshold,
            min_area,
            alignment_enabled,
            alignment_method if alignment_enabled else "none",
        ]
    )
    return output.getvalue().encode("utf-8")


def make_pdf_report_bytes(
    ref_bgr: np.ndarray,
    test_bgr: np.ndarray,
    diff_gray: np.ndarray,
    mask: np.ndarray,
    overlay_bgr: np.ndarray,
    score: float,
    region_count: int,
    total_area_px: int,
    area_percent: float,
    threshold: int,
    min_area: int,
    alignment_enabled: bool,
    alignment_method: str,
    status_msg: str,
) -> bytes:
    def to_png_reader(image: np.ndarray) -> ImageReader:
        png = encode_png_bytes(image)
        return ImageReader(io.BytesIO(png))

    def draw_image_fit(
        pdf: canvas.Canvas, img_reader: ImageReader, x: float, y: float, w: float, h: float
    ) -> None:
        img_w, img_h = img_reader.getSize()
        scale = min(w / img_w, h / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale
        draw_x = x + (w - draw_w) / 2
        draw_y = y + (h - draw_h) / 2
        pdf.drawImage(img_reader, draw_x, draw_y, width=draw_w, height=draw_h, preserveAspectRatio=True)

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    page_w, page_h = A4
    margin = 36

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(margin, page_h - margin, "SSIM Bildvergleich Report")

    pdf.setFont("Helvetica", 10)
    y_info = page_h - margin - 24
    info_lines = [
        f"SSIM-Score: {score:.4f}",
        f"Abweichungsregionen: {region_count}",
        f"Gesamt-Abweichungsflaeche: {total_area_px} px ({area_percent:.2f}%)",
        f"Threshold: {threshold}, Min. Defektflaeche: {min_area} px",
        (
            f"Alignment: {alignment_method} ({status_msg})"
            if alignment_enabled
            else "Alignment: deaktiviert"
        ),
    ]
    for line in info_lines:
        pdf.drawString(margin, y_info, line)
        y_info -= 14

    cell_gap = 12
    grid_top = y_info - 6
    col_w = (page_w - 2 * margin - cell_gap) / 2
    row_h = 145

    items = [
        ("Referenzbild", to_png_reader(ref_bgr)),
        ("Testbild", to_png_reader(test_bgr)),
        ("SSIM-Diff (invertiert)", to_png_reader(diff_gray)),
        ("Threshold-Maske", to_png_reader(mask)),
        ("Overlay", to_png_reader(overlay_bgr)),
    ]

    for idx, (title, img_reader) in enumerate(items):
        row = idx // 2
        col = idx % 2
        x = margin + col * (col_w + cell_gap)
        y = grid_top - (row + 1) * row_h - row * cell_gap
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(x, y + row_h - 12, title)
        draw_image_fit(pdf, img_reader, x, y, col_w, row_h - 18)

    pdf.showPage()
    pdf.save()
    return buf.getvalue()


def main() -> None:
    st.set_page_config(page_title="SSIM Bildvergleich", layout="wide")
    st.title("SSIM Bildvergleich (OpenCV + scikit-image)")

    st.markdown(
        "Vergleicht ein Golden Image mit einem Testbild und visualisiert Abweichungen als Maske/Overlay."
    )

    col_left, col_right = st.columns(2)
    with col_left:
        ref_file = st.file_uploader("Referenzbild (Golden)", type=["png", "jpg", "jpeg"], key="ref")
    with col_right:
        test_file = st.file_uploader("Testbild", type=["png", "jpg", "jpeg"], key="test")

    align_enabled = st.checkbox("Bilder automatisch ausrichten (Alignment)", value=True)
    alignment_method = "ECC (Affine)"
    if align_enabled:
        alignment_method = st.selectbox(
            "Alignment-Methode",
            options=["ECC (Affine)", "ORB (Homography)"],
            index=0,
        )
    threshold_value = st.slider("Abweichungsschwelle", min_value=0, max_value=255, value=45)
    min_area = st.slider("Min. Defektfläche (px)", min_value=0, max_value=20000, value=120, step=10)
    show_boxes = st.checkbox("Bounding Boxes anzeigen", value=True)
    show_heatmap = st.checkbox("Heatmap Overlay anzeigen", value=True)

    if not ref_file or not test_file:
        st.info("Bitte beide Bilder hochladen, um den Vergleich zu starten.")
        return

    try:
        ref_bgr = load_image(ref_file)
        test_bgr = load_image(test_file)
    except ValueError as err:
        st.error(str(err))
        return

    ref_h, ref_w = ref_bgr.shape[:2]
    test_h, test_w = test_bgr.shape[:2]

    if (ref_w / ref_h) != (test_w / test_h):
        st.warning(
            "Unterschiedliches Seitenverhältnis erkannt. Das Testbild wird auf die Referenzgröße skaliert; "
            "dadurch kann es zu geometrischen Verzerrungen kommen."
        )

    if (ref_h, ref_w) != (test_h, test_w):
        test_bgr = cv2.resize(test_bgr, (ref_w, ref_h), interpolation=cv2.INTER_AREA)

    status_msg = "Alignment deaktiviert."
    if align_enabled:
        test_bgr, _, status_msg = align_images(ref_bgr, test_bgr, method=alignment_method)

    if "fehlgeschlagen" in status_msg.lower():
        st.warning(status_msg)
    else:
        st.info(status_msg)

    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

    score, diff_gray = compute_ssim_diff(ref_gray, test_gray)
    diff_inverted = cv2.bitwise_not(diff_gray)

    _, mask = cv2.threshold(diff_inverted, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, region_stats = find_regions(mask, min_area=min_area)

    filtered_mask = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filtered_mask, contours, contourIdx=-1, color=255, thickness=cv2.FILLED)

    overlay = make_overlay(
        test_bgr=test_bgr,
        mask=filtered_mask,
        contours=contours,
        show_boxes=show_boxes,
        show_heatmap=show_heatmap,
    )

    image_area = ref_h * ref_w
    area_percent = (region_stats.total_area / image_area * 100) if image_area else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("SSIM-Score", f"{score:.4f}")
    m2.metric("Anzahl Abweichungsregionen", str(region_stats.count))
    m3.metric("Gesamt-Abweichungsfläche", f"{region_stats.total_area} px ({area_percent:.2f}%)")

    v1, v2, v3 = st.columns(3)
    with v1:
        st.subheader("Referenzbild")
        st.image(to_rgb(ref_bgr), use_container_width=True)
    with v2:
        st.subheader("Testbild")
        st.image(to_rgb(test_bgr), use_container_width=True)
    with v3:
        st.subheader("SSIM-Diff (grau)")
        st.image(diff_inverted, clamp=True, use_container_width=True)

    v4, v5 = st.columns(2)
    with v4:
        st.subheader("Threshold-Maske")
        st.image(filtered_mask, clamp=True, use_container_width=True)
    with v5:
        st.subheader("Overlay")
        st.image(to_rgb(overlay), use_container_width=True)

    try:
        overlay_png = encode_png_bytes(overlay)
        mask_png = encode_png_bytes(filtered_mask)
        diff_png = encode_png_bytes(diff_inverted)
        csv_bytes = make_csv_bytes(
            score=score,
            region_count=region_stats.count,
            total_area_px=region_stats.total_area,
            area_percent=area_percent,
            threshold=threshold_value,
            min_area=min_area,
            alignment_enabled=align_enabled,
            alignment_method=alignment_method,
        )
        pdf_bytes = make_pdf_report_bytes(
            ref_bgr=ref_bgr,
            test_bgr=test_bgr,
            diff_gray=diff_inverted,
            mask=filtered_mask,
            overlay_bgr=overlay,
            score=score,
            region_count=region_stats.count,
            total_area_px=region_stats.total_area,
            area_percent=area_percent,
            threshold=threshold_value,
            min_area=min_area,
            alignment_enabled=align_enabled,
            alignment_method=alignment_method,
            status_msg=status_msg,
        )
    except ValueError as err:
        st.warning(f"Download-Dateien konnten nicht erzeugt werden: {err}")
        return

    st.subheader("Downloads")
    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        st.download_button(
            label="Overlay PNG",
            data=overlay_png,
            file_name="overlay.png",
            mime="image/png",
        )
    with d2:
        st.download_button(
            label="Maske PNG",
            data=mask_png,
            file_name="mask.png",
            mime="image/png",
        )
    with d3:
        st.download_button(
            label="Diff PNG",
            data=diff_png,
            file_name="ssim_diff.png",
            mime="image/png",
        )
    with d4:
        st.download_button(
            label="Metriken CSV",
            data=csv_bytes,
            file_name="metrics.csv",
            mime="text/csv",
        )
    with d5:
        st.download_button(
            label="Report PDF",
            data=pdf_bytes,
            file_name="vergleich_report.pdf",
            mime="application/pdf",
        )


if __name__ == "__main__":
    main()
