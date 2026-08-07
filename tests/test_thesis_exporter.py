"""Tests for the thesis bundle exporter."""
import zipfile

from src.thesis_exporter import export_thesis_bundle


def test_export_thesis_bundle_creates_zip(tmp_path):
    root = tmp_path / "project"
    tables = root / "results" / "tables"
    viz = root / "results" / "visualizations"
    model = root / "model"
    tables.mkdir(parents=True)
    viz.mkdir(parents=True)
    model.mkdir(parents=True)

    (tables / "table.json").write_text('{"a": 1}')
    (viz / "fig.png").write_text("png")
    (model / "model.pt").write_text("model")
    (model / "scaler.joblib").write_text("scaler")

    out = root / "bundle.zip"
    path = export_thesis_bundle(root, out)
    assert path.exists()

    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        assert "results/tables/table.json" in names
        assert "results/visualizations/fig.png" in names
        assert "model/model.pt" in names
        assert "model/scaler.joblib" in names


def test_export_thesis_bundle_empty_patterns_still_creates_zip(tmp_path):
    root = tmp_path / "project"
    (root / "results").mkdir(parents=True)
    out = root / "bundle.zip"
    path = export_thesis_bundle(root, out, patterns=[])
    assert path.exists()
    with zipfile.ZipFile(path, "r") as zf:
        assert zf.namelist() == []
