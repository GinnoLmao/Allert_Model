"""ALLERT latency/timing instrumentation.

Provides a tiny per-phase timing store and JSON export. The store is optional
and low-overhead: if not provided, the ``timed_section`` context manager is a
no-op.
"""
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional


@contextmanager
def timed_section(name: str, store: Optional[Dict[str, Any]] = None):
    """Context manager that records the elapsed time of a code block.

    Parameters
    ----------
    name:
        Phase name (e.g., "preprocessing").
    store:
        Mutable dictionary. If None, the context manager is a no-op.
    """
    if store is None:
        yield
        return
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        store.setdefault(name, []).append(elapsed)


def summarize(store: Dict[str, Any]) -> Dict[str, Any]:
    """Return mean/p95/max per recorded phase."""
    report: Dict[str, Any] = {}
    for name, times in store.items():
        if not times:
            continue
        sorted_times = sorted(times)
        p95_idx = int(0.95 * len(sorted_times))
        p95_idx = min(p95_idx, len(sorted_times) - 1)
        report[name] = {
            "mean_ms": round(1000 * sum(times) / len(times), 2),
            "p95_ms": round(1000 * sorted_times[p95_idx], 2),
            "max_ms": round(1000 * max(times), 2),
            "n_samples": len(times),
        }
    return report


def export_latency_report(
    store: Dict[str, Any],
    out_path: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export the timing store to a JSON file.

    Parameters
    ----------
    store:
        Timing store populated by ``timed_section``.
    out_path:
        Destination JSON path.
    extra:
        Optional metadata to merge into the report.

    Returns
    -------
    Path
        The written JSON path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"phases": summarize(store), "unit": "ms"}
    if extra:
        report["metadata"] = extra
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


def plot_latency_breakdown(
    store: Dict[str, Any],
    out_path: Path,
    title: str = "Fig 4.13 — Pipeline Latency Breakdown",
) -> Path:
    """Generate a horizontal bar chart of mean/p95/max latency per phase."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = summarize(store)
    if not report:
        return None

    phases = list(report.keys())
    means = [report[p]["mean_ms"] for p in phases]
    p95s = [report[p]["p95_ms"] for p in phases]
    maxs = [report[p]["max_ms"] for p in phases]

    y = range(len(phases))
    height = 0.25
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(phases) + 2)))
    ax.barh([i - height for i in y], means, height=height, label="mean", color="steelblue")
    ax.barh(y, p95s, height=height, label="p95", color="coral")
    ax.barh([i + height for i in y], maxs, height=height, label="max", color="firebrick")
    ax.set_yticks(y)
    ax.set_yticklabels(phases)
    ax.set_xlabel("Latency (ms)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


__all__ = [
    "timed_section",
    "summarize",
    "export_latency_report",
    "plot_latency_breakdown",
]
