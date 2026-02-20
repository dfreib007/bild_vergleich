from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "test_app.db"
    monkeypatch.setattr(app, "DB_PATH", db_path)
    monkeypatch.setattr(app, "DEFAULT_ADMIN_EMAIL", "admin@test.local")
    monkeypatch.setattr(app, "DEFAULT_ADMIN_PASSWORD", "Admin1234!")
    app.init_database()
    return db_path


def make_blank_image(width: int = 120, height: int = 80) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_hash_and_verify_password_roundtrip() -> None:
    hashed = app.hash_password("Secret123!")
    assert hashed.startswith("pbkdf2_sha256$")
    assert app.verify_password("Secret123!", hashed)
    assert not app.verify_password("WrongPwd", hashed)


def test_init_database_bootstraps_single_admin(isolated_db: Path) -> None:
    users = app.list_users()
    admins = [u for u in users if u["is_admin"]]
    assert len(admins) == 1
    assert admins[0]["email"] == "admin@test.local"


def test_create_user_and_authenticate(isolated_db: Path) -> None:
    ok, _ = app.create_user("user@example.com", "StrongPass1")
    assert ok
    auth = app.authenticate_user("user@example.com", "StrongPass1")
    assert auth is not None
    assert auth["email"] == "user@example.com"
    assert auth["is_admin"] is False


def test_create_user_rejects_duplicate_email(isolated_db: Path) -> None:
    ok1, _ = app.create_user("user@example.com", "StrongPass1")
    ok2, msg2 = app.create_user("user@example.com", "StrongPass1")
    assert ok1 is True
    assert ok2 is False
    assert "existiert bereits" in msg2


def test_set_user_active_blocks_login_when_deactivated(isolated_db: Path) -> None:
    app.create_user("user@example.com", "StrongPass1")
    ok, _ = app.set_user_active("user@example.com", active=False)
    assert ok
    assert app.authenticate_user("user@example.com", "StrongPass1") is None


def test_admin_cannot_be_deactivated(isolated_db: Path) -> None:
    ok, msg = app.set_user_active("admin@test.local", active=False)
    assert ok is False
    assert "nicht deaktiviert" in msg


def test_update_user_password_changes_login_secret(isolated_db: Path) -> None:
    app.create_user("user@example.com", "StrongPass1")
    ok, _ = app.update_user_password("user@example.com", "NewStrongPass2")
    assert ok
    assert app.authenticate_user("user@example.com", "StrongPass1") is None
    assert app.authenticate_user("user@example.com", "NewStrongPass2") is not None


def test_run_comparison_identical_images_reports_no_regions() -> None:
    img = make_blank_image()
    result = app.run_comparison(
        ref_bgr=img,
        test_bgr=img.copy(),
        threshold_value=45,
        min_area=10,
        align_enabled=False,
        alignment_method="ECC (Affine)",
        show_boxes=True,
        show_heatmap=True,
    )
    assert result["score"] == pytest.approx(1.0, abs=1e-6)
    assert result["region_stats"].count == 0
    assert result["region_stats"].total_area == 0


def test_run_comparison_detects_modified_region() -> None:
    ref = make_blank_image()
    test = ref.copy()
    test[20:40, 30:55] = (255, 255, 255)

    result = app.run_comparison(
        ref_bgr=ref,
        test_bgr=test,
        threshold_value=20,
        min_area=10,
        align_enabled=False,
        alignment_method="ECC (Affine)",
        show_boxes=True,
        show_heatmap=True,
    )
    assert result["score"] < 1.0
    assert result["region_stats"].count >= 1
    assert result["region_stats"].total_area > 0


def test_save_interaction_result_and_overview_mapping(isolated_db: Path) -> None:
    image_a = make_blank_image()
    image_b = make_blank_image()
    image_b[10:20, 10:20] = (255, 255, 255)

    app.save_interaction_result(
        image_a=image_a,
        image_b=image_b,
        diff_a_to_b=image_b,
        diff_b_to_a=image_a,
        selected_image="Bild B",
        selected_by="user@example.com",
    )

    rows = app.list_interaction_overview_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["selected_letter"] == "B"
    assert row["selected_image_png"] == row["image_b_png"]
