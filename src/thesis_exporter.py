"""ALLERT thesis bundle exporter.

Zips all thesis tables, figures, and model artifacts into a single archive.
"""
import zipfile
from pathlib import Path
from typing import List, Optional

from src.config_loader import get_project_root


DEFAULT_PATTERNS = [
    "results/tables/*.json",
    "results/visualizations/*.png",
    "model/*.json",
    "model/*.joblib",
    "model/*.pt",
]


def export_thesis_bundle(
    root: Path,
    out_path: Optional[Path] = None,
    patterns: Optional[List[str]] = None,
) -> Path:
    """Zip thesis artifacts into a single archive.

    Parameters
    ----------
    root:
        Project root directory.
    out_path:
        Destination zip path. Defaults to ``results/thesis_bundle.zip``.
    patterns:
        Glob patterns relative to ``root``. Defaults to ``DEFAULT_PATTERNS``.

    Returns
    -------
    Path
        The written zip file path.
    """
    root = Path(root)
    if out_path is None:
        out_path = root / "results" / "thesis_bundle.zip"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files: List[Path] = []
    for pattern in patterns or DEFAULT_PATTERNS:
        files.extend(root.glob(pattern))

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(set(files)):
            zf.write(f, f.relative_to(root))

    return out_path


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Export ALLERT thesis bundle")
    p.add_argument("--root", default=None, help="Project root (default: repository root)")
    p.add_argument("--out", default=None, help="Output zip path")
    args = p.parse_args(argv)
    root = Path(args.root) if args.root else get_project_root()
    out = Path(args.out) if args.out else None
    path = export_thesis_bundle(root, out)
    print(f"Exported thesis bundle to {path}")
    return path


if __name__ == "__main__":
    main()


__all__ = ["export_thesis_bundle"]
