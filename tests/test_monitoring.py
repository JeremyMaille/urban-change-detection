"""Tests for the inference monitoring and drift detection."""
import json

import numpy as np

from src.serving.monitoring import (
    DriftReport,
    InferenceLogger,
    check_drift,
    image_stats,
    load_reference,
)

# A synthetic training reference: satellite-like images, means ~100, spread ~15
REFERENCE = {
    "channel_mean": [100.0, 105.0, 95.0],
    "channel_mean_std": [15.0, 15.0, 15.0],
}


def test_image_stats_shape_and_values():
    arr = np.full((64, 64, 3), 128, dtype=np.uint8)
    s = image_stats(arr)
    assert s["mean"] == [128.0, 128.0, 128.0]
    assert s["std"] == [0.0, 0.0, 0.0]
    assert s["height"] == 64 and s["width"] == 64


def test_no_drift_for_in_distribution_input():
    stats = {"mean": [102.0, 104.0, 97.0], "std": [30.0] * 3}
    report = check_drift(stats, REFERENCE)
    assert not report.drifted
    assert report.reasons == []


def test_drift_detected_for_out_of_distribution_input():
    # A very bright image (e.g. a vacation photo, mean ~220): 8 std away
    stats = {"mean": [220.0, 225.0, 230.0], "std": [40.0] * 3}
    report = check_drift(stats, REFERENCE)
    assert report.drifted
    assert len(report.reasons) == 3  # all three channels flagged


def test_drift_threshold_is_respected():
    # Exactly 2.9 std away: below the 3.0 threshold, no drift
    stats = {"mean": [100.0 + 2.9 * 15.0, 105.0, 95.0], "std": [30.0] * 3}
    assert not check_drift(stats, REFERENCE).drifted
    # 3.1 std away: drift
    stats = {"mean": [100.0 + 3.1 * 15.0, 105.0, 95.0], "std": [30.0] * 3}
    assert check_drift(stats, REFERENCE).drifted


def test_logger_writes_one_json_line_per_call(tmp_path):
    log_file = tmp_path / "log.jsonl"
    logger = InferenceLogger(str(log_file))
    stats = image_stats(np.zeros((32, 32, 3), dtype=np.uint8))
    for _ in range(3):
        logger.log(0.5, stats, stats, 0.02, DriftReport(drifted=False))
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 3
    record = json.loads(lines[0])
    assert record["latency_s"] == 0.5
    assert record["predicted_change_ratio"] == 0.02
    assert record["drift_detected"] is False


def test_load_reference_returns_none_when_missing(tmp_path):
    assert load_reference(str(tmp_path / "nope.json")) is None
