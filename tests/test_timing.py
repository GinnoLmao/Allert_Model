"""Tests for the timing utility."""
import time

from src.utils.timing import timed_section, summarize, export_latency_report


def test_timed_section_records_elapsed():
    store = {}
    with timed_section("sleep", store):
        time.sleep(0.01)
    assert "sleep" in store
    assert len(store["sleep"]) == 1
    assert store["sleep"][0] > 0


def test_timed_section_noop_without_store():
    # Should not raise
    with timed_section("noop", None):
        pass


def test_summarize_computes_statistics():
    store = {"phase": [0.1, 0.2, 0.3, 0.4, 1.0]}
    report = summarize(store)
    assert "phase" in report
    assert report["phase"]["mean_ms"] > 0
    assert report["phase"]["p95_ms"] >= report["phase"]["mean_ms"]
    assert report["phase"]["max_ms"] >= report["phase"]["p95_ms"]


def test_export_latency_report(tmp_path):
    store = {"phase": [0.1, 0.2]}
    out = tmp_path / "latency.json"
    export_latency_report(store, out, extra={"seed": 42})
    assert out.exists()
    import json
    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["phases"]["phase"]["n_samples"] == 2
    assert data["metadata"]["seed"] == 42
