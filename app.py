from __future__ import annotations

import csv
import hashlib
import hmac
import io
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from skimage.metrics import structural_similarity

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DB_PATH = Path("interaktion_results.db")
DEFAULT_ADMIN_EMAIL = os.getenv("APP_BOOTSTRAP_ADMIN_EMAIL", "admin@bildvergleich.local").strip().lower()
DEFAULT_ADMIN_PASSWORD = os.getenv("APP_BOOTSTRAP_ADMIN_PASSWORD", "Admin1234!")


@dataclass
class RegionStats:
    count: int
    total_area: int


def hash_password(password: str, iterations: int = 210_000) -> str:
    salt = os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations).hex()
    return f"pbkdf2_sha256${iterations}${salt}${digest}"


def verify_password(password: str, hashed: str) -> bool:
    try:
        algo, iterations_str, salt, digest = hashed.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        check = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            int(iterations_str),
        ).hex()
        return hmac.compare_digest(check, digest)
    except (ValueError, TypeError):
        return False


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interaction_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                reference_a_png BLOB NOT NULL,
                comparison_b_png BLOB NOT NULL,
                difference_a_to_b_png BLOB NOT NULL,
                difference_b_to_a_png BLOB NOT NULL,
                selected_image TEXT NOT NULL,
                selected_by TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
            """
        )

        admin_row = conn.execute("SELECT id FROM users WHERE is_admin = 1 LIMIT 1").fetchone()
        if admin_row is None:
            conn.execute(
                """
                INSERT INTO users (email, password_hash, is_admin, is_active, created_at)
                VALUES (?, ?, 1, 1, ?)
                """,
                (
                    DEFAULT_ADMIN_EMAIL,
                    hash_password(DEFAULT_ADMIN_PASSWORD),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        conn.commit()


def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    normalized_email = email.strip().lower()
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT id, email, password_hash, is_admin, is_active
            FROM users
            WHERE email = ?
            LIMIT 1
            """,
            (normalized_email,),
        ).fetchone()

    if row is None or int(row["is_active"]) != 1:
        return None
    if not verify_password(password, row["password_hash"]):
        return None

    return {
        "id": int(row["id"]),
        "email": row["email"],
        "is_admin": bool(row["is_admin"]),
    }


def list_users() -> list[dict[str, Any]]:
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, email, is_admin, is_active, created_at
            FROM users
            ORDER BY id ASC
            """
        ).fetchall()

    return [
        {
            "id": int(row["id"]),
            "email": row["email"],
            "is_admin": bool(row["is_admin"]),
            "is_active": bool(row["is_active"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def create_user(email: str, password: str) -> tuple[bool, str]:
    normalized_email = email.strip().lower()
    if not normalized_email or "@" not in normalized_email:
        return False, "Ungültige E-Mail-Adresse."
    if len(password) < 8:
        return False, "Passwort muss mindestens 8 Zeichen haben."

    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (email, password_hash, is_admin, is_active, created_at)
                VALUES (?, ?, 0, 1, ?)
                """,
                (normalized_email, hash_password(password), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        return False, "Benutzer existiert bereits."

    return True, "Benutzer wurde angelegt."


def update_user_password(email: str, new_password: str) -> tuple[bool, str]:
    normalized_email = email.strip().lower()
    if len(new_password) < 8:
        return False, "Passwort muss mindestens 8 Zeichen haben."

    with get_db_connection() as conn:
        row = conn.execute("SELECT id FROM users WHERE email = ? LIMIT 1", (normalized_email,)).fetchone()
        if row is None:
            return False, "Benutzer nicht gefunden."

        conn.execute(
            "UPDATE users SET password_hash = ? WHERE email = ?",
            (hash_password(new_password), normalized_email),
        )
        conn.commit()

    return True, "Passwort aktualisiert."


def set_user_active(email: str, active: bool) -> tuple[bool, str]:
    normalized_email = email.strip().lower()
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT id, is_admin FROM users WHERE email = ? LIMIT 1", (normalized_email,)
        ).fetchone()
        if row is None:
            return False, "Benutzer nicht gefunden."

        if int(row["is_admin"]) == 1 and not active:
            return False, "Admin kann nicht deaktiviert werden."

        conn.execute("UPDATE users SET is_active = ? WHERE email = ?", (1 if active else 0, normalized_email))
        conn.commit()

    return True, "Benutzerstatus aktualisiert."


def load_image(file: Any) -> np.ndarray:
    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Bild konnte nicht geladen werden. Bitte PNG/JPG verwenden.")
    return image


def load_image_from_path(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    return image


def align_images_ecc(ref: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, str]:
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)

    try:
        cc, warp_matrix = cv2.findTransformECC(ref_gray, test_gray, warp_matrix, cv2.MOTION_AFFINE, criteria)
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
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=4000)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    kp_test, des_test = orb.detectAndCompute(test_gray, None)

    if des_ref is None or des_test is None:
        return test, None, "Alignment fehlgeschlagen: keine ORB-Features gefunden. Fallback ohne Alignment."

    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des_test, des_ref)
    if len(matches) < 4:
        return test, None, "Alignment fehlgeschlagen: zu wenige Matches. Fallback ohne Alignment."

    matches = sorted(matches, key=lambda m: m.distance)
    good_matches = matches[: max(4, int(len(matches) * 0.6))]

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


def align_images(ref: np.ndarray, test: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray | None, str]:
    if method == "ORB (Homography)":
        return align_images_orb(ref, test)
    return align_images_ecc(ref, test)


def compute_ssim_diff(ref_gray: np.ndarray, test_gray: np.ndarray) -> tuple[float, np.ndarray]:
    score, diff = structural_similarity(ref_gray, test_gray, full=True)
    return float(score), (diff * 255).astype(np.uint8)


def find_regions(mask: np.ndarray, min_area: int) -> tuple[list[np.ndarray], RegionStats]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    total_area = int(sum(cv2.contourArea(cnt) for cnt in filtered))
    return filtered, RegionStats(count=len(filtered), total_area=total_area)


def make_overlay(
    test_bgr: np.ndarray,
    mask: np.ndarray,
    contours: list[np.ndarray],
    show_boxes: bool,
    show_heatmap: bool,
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


def run_comparison(
    ref_bgr: np.ndarray,
    test_bgr: np.ndarray,
    threshold_value: int,
    min_area: int,
    align_enabled: bool,
    alignment_method: str,
    show_boxes: bool,
    show_heatmap: bool,
) -> dict[str, Any]:
    ref_h, ref_w = ref_bgr.shape[:2]
    test_h, test_w = test_bgr.shape[:2]

    aspect_ratio_mismatch = not np.isclose(ref_w / ref_h, test_w / test_h)
    if (ref_h, ref_w) != (test_h, test_w):
        test_bgr = cv2.resize(test_bgr, (ref_w, ref_h), interpolation=cv2.INTER_AREA)

    status_msg = "Alignment deaktiviert."
    if align_enabled:
        test_bgr, _, status_msg = align_images(ref_bgr, test_bgr, method=alignment_method)

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

    return {
        "score": score,
        "region_stats": region_stats,
        "area_percent": area_percent,
        "status_msg": status_msg,
        "aspect_ratio_mismatch": aspect_ratio_mismatch,
        "diff_inverted": diff_inverted,
        "filtered_mask": filtered_mask,
        "overlay": overlay,
        "ref_bgr": ref_bgr,
        "test_bgr": test_bgr,
    }


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def encode_png_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG-Encoding fehlgeschlagen.")
    return buf.tobytes()


def make_single_csv_bytes(
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


def make_batch_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO()
    if not rows:
        return output.getvalue().encode("utf-8")

    fields = list(rows[0].keys())
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
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
        return ImageReader(io.BytesIO(encode_png_bytes(image)))

    def draw_image_fit(pdf: canvas.Canvas, img_reader: ImageReader, x: float, y: float, w: float, h: float) -> None:
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
        f"Alignment: {alignment_method} ({status_msg})" if alignment_enabled else "Alignment: deaktiviert",
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


def save_interaction_result(
    image_a: np.ndarray,
    image_b: np.ndarray,
    diff_a_to_b: np.ndarray,
    diff_b_to_a: np.ndarray,
    selected_image: str,
    selected_by: str,
) -> None:
    created_at = datetime.now(timezone.utc).isoformat()

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO interaction_results (
                created_at,
                reference_a_png,
                comparison_b_png,
                difference_a_to_b_png,
                difference_b_to_a_png,
                selected_image,
                selected_by
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                sqlite3.Binary(encode_png_bytes(image_a)),
                sqlite3.Binary(encode_png_bytes(image_b)),
                sqlite3.Binary(encode_png_bytes(diff_a_to_b)),
                sqlite3.Binary(encode_png_bytes(diff_b_to_a)),
                selected_image,
                selected_by,
            ),
        )
        conn.commit()


def list_interaction_overview_rows() -> list[dict[str, Any]]:
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                created_at,
                reference_a_png,
                comparison_b_png,
                difference_a_to_b_png,
                selected_image
            FROM interaction_results
            ORDER BY id DESC
            """
        ).fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        selected_label = row["selected_image"]
        selected_letter = "A" if selected_label == "Bild A" else "B" if selected_label == "Bild B" else "?"
        selected_blob = row["reference_a_png"] if selected_letter == "A" else row["comparison_b_png"]
        result.append(
            {
                "id": int(row["id"]),
                "image_a_png": row["reference_a_png"],
                "image_b_png": row["comparison_b_png"],
                "diff_a_to_b_png": row["difference_a_to_b_png"],
                "selected_image_png": selected_blob,
                "selected_letter": selected_letter,
                "timestamp": row["created_at"],
            }
        )
    return result


def collect_images(folder: Path, recursive: bool) -> dict[str, Path]:
    candidates = folder.rglob("*") if recursive else folder.glob("*")
    files: dict[str, Path] = {}
    for path in candidates:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.relative_to(folder).as_posix().lower() if recursive else path.name.lower()
        if key not in files:
            files[key] = path
    return files


def find_logo_path() -> Path | None:
    candidates = [
        Path("evolit-logo.png"),
        Path("evolit_logo.png"),
        Path("logo.png"),
        Path("assets/evolit-logo.png"),
        Path("assets/evolit_logo.png"),
        Path("assets/logo.png"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def render_header() -> bool:
    user = st.session_state.get("auth_user")
    is_admin = bool(user and user.get("is_admin"))

    logo_path = find_logo_path()
    col_logo, col_title, col_admin, col_user, col_logout = st.columns([1, 6, 1.2, 2, 1], vertical_alignment="center")
    with col_logo:
        if logo_path is not None:
            st.image(str(logo_path), use_container_width=True)
    with col_title:
        st.title("Bildvergleich")
    with col_admin:
        admin_clicked = st.button("Admin", disabled=not is_admin, use_container_width=True)
    with col_user:
        if user:
            st.caption(user["email"])
    with col_logout:
        if st.button("Logout", use_container_width=True):
            st.session_state["auth_user"] = None
            st.session_state["selected_mode"] = "Single Vergleich"
            st.rerun()

    st.markdown("Vergleicht ein Golden Image mit einem Testbild und visualisiert Abweichungen als Maske/Overlay.")
    return bool(admin_clicked)


def render_login() -> None:
    st.subheader("Login")
    st.caption("Bitte mit E-Mail-Adresse und Passwort anmelden.")

    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("E-Mail")
        password = st.text_input("Passwort", type="password")
        submitted = st.form_submit_button("Anmelden", type="primary")

    if submitted:
        user = authenticate_user(email=email, password=password)
        if user is None:
            st.error("Ungültige Anmeldedaten oder Benutzer deaktiviert.")
            return
        st.session_state["auth_user"] = user
        st.session_state["selected_mode"] = "Single Vergleich"
        st.rerun()


def render_common_controls(key_prefix: str) -> tuple[bool, str, int, int, bool, bool]:
    align_enabled = st.checkbox("Bilder automatisch ausrichten (Alignment)", value=True, key=f"align_{key_prefix}")
    alignment_method = "ECC (Affine)"
    if align_enabled:
        alignment_method = st.selectbox(
            "Alignment-Methode",
            options=["ECC (Affine)", "ORB (Homography)"],
            index=0,
            key=f"align_method_{key_prefix}",
        )

    threshold_value = st.slider(
        "Abweichungsschwelle", min_value=0, max_value=255, value=45, key=f"threshold_{key_prefix}"
    )
    min_area = st.slider(
        "Min. Defektfläche (px)", min_value=0, max_value=20000, value=120, step=10, key=f"min_area_{key_prefix}"
    )
    show_boxes = st.checkbox("Bounding Boxes anzeigen", value=True, key=f"boxes_{key_prefix}")
    show_heatmap = st.checkbox("Heatmap Overlay anzeigen", value=True, key=f"heatmap_{key_prefix}")

    return align_enabled, alignment_method, threshold_value, min_area, show_boxes, show_heatmap


def render_single_mode() -> None:
    st.subheader("Single Vergleich")
    col_left, col_right = st.columns(2)
    with col_left:
        ref_file = st.file_uploader("Referenzbild (Golden)", type=["png", "jpg", "jpeg"], key="ref_single")
    with col_right:
        test_file = st.file_uploader("Testbild", type=["png", "jpg", "jpeg"], key="test_single")

    align_enabled, alignment_method, threshold_value, min_area, show_boxes, show_heatmap = render_common_controls("single")

    if not ref_file or not test_file:
        st.info("Bitte beide Bilder hochladen, um den Vergleich zu starten.")
        return

    try:
        ref_bgr = load_image(ref_file)
        test_bgr = load_image(test_file)
    except ValueError as err:
        st.error(str(err))
        return

    result = run_comparison(
        ref_bgr=ref_bgr,
        test_bgr=test_bgr,
        threshold_value=threshold_value,
        min_area=min_area,
        align_enabled=align_enabled,
        alignment_method=alignment_method,
        show_boxes=show_boxes,
        show_heatmap=show_heatmap,
    )

    if result["aspect_ratio_mismatch"]:
        st.warning(
            "Unterschiedliches Seitenverhältnis erkannt. Das Testbild wird auf die Referenzgröße skaliert; "
            "dadurch kann es zu geometrischen Verzerrungen kommen."
        )

    status_msg = result["status_msg"]
    if "fehlgeschlagen" in status_msg.lower():
        st.warning(status_msg)
    else:
        st.info(status_msg)

    region_stats: RegionStats = result["region_stats"]
    score = result["score"]
    area_percent = result["area_percent"]

    m1, m2, m3 = st.columns(3)
    m1.metric("SSIM-Score", f"{score:.4f}")
    m2.metric("Anzahl Abweichungsregionen", str(region_stats.count))
    m3.metric("Gesamt-Abweichungsfläche", f"{region_stats.total_area} px ({area_percent:.2f}%)")

    v1, v2, v3 = st.columns(3)
    with v1:
        st.subheader("Referenzbild")
        st.image(to_rgb(result["ref_bgr"]), use_container_width=True)
    with v2:
        st.subheader("Testbild")
        st.image(to_rgb(result["test_bgr"]), use_container_width=True)
    with v3:
        st.subheader("SSIM-Diff (grau)")
        st.image(result["diff_inverted"], clamp=True, use_container_width=True)

    v4, v5 = st.columns(2)
    with v4:
        st.subheader("Threshold-Maske")
        st.image(result["filtered_mask"], clamp=True, use_container_width=True)
    with v5:
        st.subheader("Overlay")
        st.image(to_rgb(result["overlay"]), use_container_width=True)

    try:
        overlay_png = encode_png_bytes(result["overlay"])
        mask_png = encode_png_bytes(result["filtered_mask"])
        diff_png = encode_png_bytes(result["diff_inverted"])
        csv_bytes = make_single_csv_bytes(
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
            ref_bgr=result["ref_bgr"],
            test_bgr=result["test_bgr"],
            diff_gray=result["diff_inverted"],
            mask=result["filtered_mask"],
            overlay_bgr=result["overlay"],
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
        st.download_button("Overlay PNG", data=overlay_png, file_name="overlay.png", mime="image/png")
    with d2:
        st.download_button("Maske PNG", data=mask_png, file_name="mask.png", mime="image/png")
    with d3:
        st.download_button("Diff PNG", data=diff_png, file_name="ssim_diff.png", mime="image/png")
    with d4:
        st.download_button("Metriken CSV", data=csv_bytes, file_name="metrics.csv", mime="text/csv")
    with d5:
        st.download_button("Report PDF", data=pdf_bytes, file_name="vergleich_report.pdf", mime="application/pdf")


def render_batch_mode() -> None:
    st.subheader("Batch Modus")
    st.caption("Ordnerpfade sind lokale Pfade auf dem Rechner, auf dem Streamlit läuft.")

    col_left, col_right = st.columns(2)
    with col_left:
        ref_folder = st.text_input("Referenz-Ordner", placeholder="/path/to/golden", key="ref_folder")
    with col_right:
        test_folder = st.text_input("Test-Ordner", placeholder="/path/to/test", key="test_folder")

    recursive = st.checkbox("Unterordner rekursiv einbeziehen", value=True, key="batch_recursive")
    align_enabled, alignment_method, threshold_value, min_area, show_boxes, show_heatmap = render_common_controls("batch")

    run_batch = st.button("Batch Vergleich starten", type="primary")
    if not run_batch:
        return

    ref_root = Path(ref_folder).expanduser()
    test_root = Path(test_folder).expanduser()

    if not ref_root.is_dir() or not test_root.is_dir():
        st.error("Bitte zwei gültige Ordnerpfade angeben.")
        return

    ref_files = collect_images(ref_root, recursive=recursive)
    test_files = collect_images(test_root, recursive=recursive)

    if not ref_files:
        st.error("Im Referenz-Ordner wurden keine PNG/JPG-Dateien gefunden.")
        return
    if not test_files:
        st.error("Im Test-Ordner wurden keine PNG/JPG-Dateien gefunden.")
        return

    common_keys = sorted(set(ref_files) & set(test_files))
    only_ref = sorted(set(ref_files) - set(test_files))
    only_test = sorted(set(test_files) - set(ref_files))

    st.info(
        f"Gefunden: Referenz={len(ref_files)}, Test={len(test_files)}, gematcht={len(common_keys)}, "
        f"nur Referenz={len(only_ref)}, nur Test={len(only_test)}"
    )

    if not common_keys:
        st.warning("Keine passenden Bildpaare gefunden. Nutze identische Dateinamen/relative Pfade.")
        return

    rows: list[dict[str, Any]] = []
    progress = st.progress(0)

    for idx, key in enumerate(common_keys):
        ref_path = ref_files[key]
        test_path = test_files[key]

        row: dict[str, Any] = {
            "pair_key": key,
            "reference_path": str(ref_path),
            "test_path": str(test_path),
            "status": "ok",
            "alignment_status": "",
            "ssim_score": "",
            "regions_count": "",
            "total_area_px": "",
            "total_area_percent": "",
            "aspect_ratio_mismatch": "",
        }

        try:
            ref_bgr = load_image_from_path(ref_path)
            test_bgr = load_image_from_path(test_path)
            result = run_comparison(
                ref_bgr=ref_bgr,
                test_bgr=test_bgr,
                threshold_value=threshold_value,
                min_area=min_area,
                align_enabled=align_enabled,
                alignment_method=alignment_method,
                show_boxes=show_boxes,
                show_heatmap=show_heatmap,
            )

            region_stats: RegionStats = result["region_stats"]
            row["alignment_status"] = result["status_msg"]
            row["ssim_score"] = f"{result['score']:.6f}"
            row["regions_count"] = region_stats.count
            row["total_area_px"] = region_stats.total_area
            row["total_area_percent"] = f"{result['area_percent']:.6f}"
            row["aspect_ratio_mismatch"] = result["aspect_ratio_mismatch"]
        except ValueError as err:
            row["status"] = f"error: {err}"

        rows.append(row)
        progress.progress((idx + 1) / len(common_keys))

    success_rows = [r for r in rows if r["status"] == "ok"]

    st.subheader("Batch Ergebnisse")
    if success_rows:
        scores = np.array([float(r["ssim_score"]) for r in success_rows], dtype=np.float64)
        areas = np.array([float(r["total_area_px"]) for r in success_rows], dtype=np.float64)
        b1, b2, b3 = st.columns(3)
        b1.metric("Verglichene Paare", str(len(rows)))
        b2.metric("Ø SSIM", f"{scores.mean():.4f}")
        b3.metric("Ø Abweichungsfläche (px)", f"{areas.mean():.1f}")
    else:
        st.warning("Alle Vergleiche sind fehlgeschlagen.")

    st.dataframe(rows, use_container_width=True)

    csv_bytes = make_batch_csv_bytes(rows)
    st.download_button(
        "Batch CSV herunterladen",
        data=csv_bytes,
        file_name="batch_metrics.csv",
        mime="text/csv",
    )

    preview_candidates = [r["pair_key"] for r in success_rows]
    if not preview_candidates:
        return

    preview_key = st.selectbox("Vorschau eines Paares", options=preview_candidates, key="batch_preview")
    ref_bgr = load_image_from_path(ref_files[preview_key])
    test_bgr = load_image_from_path(test_files[preview_key])
    preview_result = run_comparison(
        ref_bgr=ref_bgr,
        test_bgr=test_bgr,
        threshold_value=threshold_value,
        min_area=min_area,
        align_enabled=align_enabled,
        alignment_method=alignment_method,
        show_boxes=show_boxes,
        show_heatmap=show_heatmap,
    )

    p1, p2, p3 = st.columns(3)
    with p1:
        st.image(to_rgb(preview_result["ref_bgr"]), caption="Referenz", use_container_width=True)
    with p2:
        st.image(to_rgb(preview_result["test_bgr"]), caption="Test", use_container_width=True)
    with p3:
        st.image(preview_result["diff_inverted"], caption="SSIM-Diff", clamp=True, use_container_width=True)

    p4, p5 = st.columns(2)
    with p4:
        st.image(preview_result["filtered_mask"], caption="Maske", clamp=True, use_container_width=True)
    with p5:
        st.image(to_rgb(preview_result["overlay"]), caption="Overlay", use_container_width=True)


def render_interaction_mode() -> None:
    st.subheader("Interaktion Modus")
    col_left, col_right = st.columns(2)
    with col_left:
        image_a_file = st.file_uploader("Bild A hochladen", type=["png", "jpg", "jpeg"], key="img_a_interaction")
    with col_right:
        image_b_file = st.file_uploader("Bild B hochladen", type=["png", "jpg", "jpeg"], key="img_b_interaction")

    align_enabled, alignment_method, threshold_value, min_area, show_boxes, show_heatmap = render_common_controls(
        "interaction"
    )

    if not image_a_file or not image_b_file:
        st.info("Bitte Bild A und Bild B hochladen.")
        return

    try:
        image_a_bgr = load_image(image_a_file)
        image_b_bgr = load_image(image_b_file)
    except ValueError as err:
        st.error(str(err))
        return

    result_a_to_b = run_comparison(
        ref_bgr=image_a_bgr,
        test_bgr=image_b_bgr,
        threshold_value=threshold_value,
        min_area=min_area,
        align_enabled=align_enabled,
        alignment_method=alignment_method,
        show_boxes=show_boxes,
        show_heatmap=show_heatmap,
    )
    result_b_to_a = run_comparison(
        ref_bgr=image_b_bgr,
        test_bgr=image_a_bgr,
        threshold_value=threshold_value,
        min_area=min_area,
        align_enabled=align_enabled,
        alignment_method=alignment_method,
        show_boxes=show_boxes,
        show_heatmap=show_heatmap,
    )

    st.markdown("**Vergleich A → B**")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.image(to_rgb(result_a_to_b["ref_bgr"]), caption="Referenz: Bild A", use_container_width=True)
    with a2:
        st.image(to_rgb(result_a_to_b["test_bgr"]), caption="Vergleich: Bild B", use_container_width=True)
    with a3:
        st.image(to_rgb(result_a_to_b["overlay"]), caption="Unterschied A → B", use_container_width=True)

    st.markdown("**Vergleich B → A**")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.image(to_rgb(result_b_to_a["ref_bgr"]), caption="Referenz: Bild B", use_container_width=True)
    with b2:
        st.image(to_rgb(result_b_to_a["test_bgr"]), caption="Vergleich: Bild A", use_container_width=True)
    with b3:
        st.image(to_rgb(result_b_to_a["overlay"]), caption="Unterschied B → A", use_container_width=True)

    m1, m2 = st.columns(2)
    with m1:
        st.metric("SSIM A → B", f"{result_a_to_b['score']:.4f}")
        st.caption(
            f"Regionen: {result_a_to_b['region_stats'].count}, "
            f"Fläche: {result_a_to_b['region_stats'].total_area} px ({result_a_to_b['area_percent']:.2f}%)"
        )
    with m2:
        st.metric("SSIM B → A", f"{result_b_to_a['score']:.4f}")
        st.caption(
            f"Regionen: {result_b_to_a['region_stats'].count}, "
            f"Fläche: {result_b_to_a['region_stats'].total_area} px ({result_b_to_a['area_percent']:.2f}%)"
        )

    selected_image = st.radio(
        "Welches Bild ist korrekt (fehlerfrei)?",
        options=["Bild A", "Bild B"],
        horizontal=True,
        key="interaction_selected_image",
    )

    user = st.session_state.get("auth_user")
    selected_by = user["email"] if user else "unknown"

    if st.button("Auswahl speichern", type="primary", key="interaction_save_button"):
        try:
            save_interaction_result(
                image_a=result_a_to_b["ref_bgr"],
                image_b=result_a_to_b["test_bgr"],
                diff_a_to_b=result_a_to_b["overlay"],
                diff_b_to_a=result_b_to_a["overlay"],
                selected_image=selected_image,
                selected_by=selected_by,
            )
            st.success(f"Gespeichert in {DB_PATH.resolve()}")
        except ValueError as err:
            st.error(f"Speichern fehlgeschlagen: {err}")


def render_overview_mode() -> None:
    st.subheader("Übersicht")
    rows = list_interaction_overview_rows()
    if not rows:
        st.info("Noch keine Einträge aus dem Interaktion Modus vorhanden.")
        return

    st.markdown("**Filter**")
    f1, f2, f3 = st.columns(3)
    with f1:
        id_filter = st.text_input("ID", key="ov_filter_id", placeholder="z. B. 12")
        has_image_a = st.selectbox("Bild A", options=["Alle", "Hat Bild"], key="ov_filter_img_a")
        has_image_b = st.selectbox("Bild B", options=["Alle", "Hat Bild"], key="ov_filter_img_b")
    with f2:
        has_diff = st.selectbox("Difference A to B", options=["Alle", "Hat Bild"], key="ov_filter_diff")
        has_selected_image = st.selectbox(
            "Ausgewähltes Bild (als Bild)", options=["Alle", "Hat Bild"], key="ov_filter_selected_img"
        )
        selected_letter_filter = st.selectbox(
            "Buchstabe des ausgewählten Bildes",
            options=["Alle", "A", "B", "?"],
            key="ov_filter_selected_letter",
        )
    with f3:
        timestamp_filter = st.text_input(
            "Timestamp (enthält)", key="ov_filter_timestamp", placeholder="z. B. 2026-02-20"
        )
        use_ts_from = st.checkbox("Von-Filter aktiv", key="ov_use_ts_from")
        ts_from = st.date_input("Timestamp von", value=date.today(), key="ov_filter_ts_from")
        use_ts_to = st.checkbox("Bis-Filter aktiv", key="ov_use_ts_to")
        ts_to = st.date_input("Timestamp bis", value=date.today(), key="ov_filter_ts_to")

    sort_col, sort_dir_col = st.columns(2)
    with sort_col:
        sort_by = st.selectbox(
            "Sortieren nach",
            options=["ID", "Buchstabe des ausgewählten Bildes", "Timestamp"],
            key="ov_sort_by",
        )
    with sort_dir_col:
        sort_dir = st.selectbox("Richtung", options=["Absteigend", "Aufsteigend"], key="ov_sort_dir")

    filtered_rows = rows

    if id_filter.strip():
        filtered_rows = [r for r in filtered_rows if id_filter.strip() in str(r["id"])]
    if has_image_a == "Hat Bild":
        filtered_rows = [r for r in filtered_rows if bool(r["image_a_png"])]
    if has_image_b == "Hat Bild":
        filtered_rows = [r for r in filtered_rows if bool(r["image_b_png"])]
    if has_diff == "Hat Bild":
        filtered_rows = [r for r in filtered_rows if bool(r["diff_a_to_b_png"])]
    if has_selected_image == "Hat Bild":
        filtered_rows = [r for r in filtered_rows if bool(r["selected_image_png"])]
    if selected_letter_filter != "Alle":
        filtered_rows = [r for r in filtered_rows if r["selected_letter"] == selected_letter_filter]
    if timestamp_filter.strip():
        filtered_rows = [r for r in filtered_rows if timestamp_filter.strip() in r["timestamp"]]

    def parse_iso_date(value: str) -> date | None:
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None

    if use_ts_from:
        filtered_rows = [
            r for r in filtered_rows if (parsed := parse_iso_date(r["timestamp"])) is not None and parsed >= ts_from
        ]
    if use_ts_to:
        filtered_rows = [
            r for r in filtered_rows if (parsed := parse_iso_date(r["timestamp"])) is not None and parsed <= ts_to
        ]

    reverse_sort = sort_dir == "Absteigend"
    if sort_by == "ID":
        filtered_rows = sorted(filtered_rows, key=lambda r: r["id"], reverse=reverse_sort)
    elif sort_by == "Buchstabe des ausgewählten Bildes":
        filtered_rows = sorted(filtered_rows, key=lambda r: r["selected_letter"], reverse=reverse_sort)
    else:
        filtered_rows = sorted(filtered_rows, key=lambda r: r["timestamp"], reverse=reverse_sort)

    st.caption(f"Treffer: {len(filtered_rows)} / {len(rows)}")
    if not filtered_rows:
        st.warning("Keine Zeilen entsprechen den aktuellen Filtern.")
        return

    headers = [
        "ID",
        "Bild A",
        "Bild B",
        "Difference A to B",
        "Ausgewähltes Bild (als Bild)",
        "Buchstabe des Ausgewählten Bildes",
        "Timestamp",
    ]
    widths = [0.7, 1.3, 1.3, 1.3, 1.3, 1.2, 1.4]

    header_cols = st.columns(widths)
    for col, title in zip(header_cols, headers):
        col.markdown(f"**{title}**")

    for row in filtered_rows:
        cols = st.columns(widths)
        cols[0].write(row["id"])
        cols[1].image(row["image_a_png"], use_container_width=True)
        cols[2].image(row["image_b_png"], use_container_width=True)
        cols[3].image(row["diff_a_to_b_png"], use_container_width=True)
        cols[4].image(row["selected_image_png"], use_container_width=True)
        cols[5].write(row["selected_letter"])
        cols[6].write(row["timestamp"])
        st.divider()


def render_admin_mode() -> None:
    st.subheader("Admin")
    st.caption("Benutzerverwaltung")

    users = list_users()
    st.dataframe(users, use_container_width=True)

    with st.form("admin_create_user"):
        st.markdown("**Neuen Benutzer anlegen**")
        new_email = st.text_input("E-Mail", key="new_user_email")
        new_password = st.text_input("Passwort", type="password", key="new_user_password")
        create_submit = st.form_submit_button("Benutzer hinzufügen")
    if create_submit:
        ok, msg = create_user(new_email, new_password)
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

    with st.form("admin_update_password"):
        st.markdown("**Passwort ändern**")
        email_options = [u["email"] for u in users]
        target_email = st.selectbox("Benutzer", options=email_options, key="pw_target_email") if email_options else ""
        updated_password = st.text_input("Neues Passwort", type="password", key="pw_new_password")
        pw_submit = st.form_submit_button("Passwort speichern")
    if pw_submit and target_email:
        ok, msg = update_user_password(target_email, updated_password)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    with st.form("admin_toggle_active"):
        st.markdown("**Benutzer aktiv/deaktivieren**")
        email_options = [u["email"] for u in users]
        toggle_email = st.selectbox("Benutzer auswählen", options=email_options, key="toggle_email") if email_options else ""
        active_choice = st.radio("Status", options=["Aktiv", "Deaktiviert"], horizontal=True, key="toggle_status")
        toggle_submit = st.form_submit_button("Status speichern")
    if toggle_submit and toggle_email:
        ok, msg = set_user_active(toggle_email, active_choice == "Aktiv")
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)


def main() -> None:
    st.set_page_config(page_title="Bildvergleich", layout="wide")
    init_database()

    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = None
    if "selected_mode" not in st.session_state:
        st.session_state["selected_mode"] = "Single Vergleich"

    if st.session_state["auth_user"] is None:
        render_login()
        st.info(
            f"Initialer Admin (nur beim ersten Start): {DEFAULT_ADMIN_EMAIL} / {DEFAULT_ADMIN_PASSWORD}. "
            "Bitte nach dem ersten Login im Admin-Menü Passwort ändern."
        )
        return

    admin_clicked = render_header()
    auth_user = st.session_state["auth_user"]
    is_admin = bool(auth_user.get("is_admin"))

    options = ["Single Vergleich", "Batch Modus", "Interaktion Modus", "Übersicht"]
    if is_admin:
        options.append("Admin")
    if admin_clicked and is_admin:
        st.session_state["selected_mode"] = "Admin"

    current_mode = st.session_state.get("selected_mode", "Single Vergleich")
    if current_mode not in options:
        current_mode = "Single Vergleich"

    mode = st.selectbox("Modus", options=options, index=options.index(current_mode), key="mode_select")
    st.session_state["selected_mode"] = mode

    if mode == "Batch Modus":
        render_batch_mode()
    elif mode == "Interaktion Modus":
        render_interaction_mode()
    elif mode == "Übersicht":
        render_overview_mode()
    elif mode == "Admin":
        render_admin_mode()
    else:
        render_single_mode()


if __name__ == "__main__":
    main()
