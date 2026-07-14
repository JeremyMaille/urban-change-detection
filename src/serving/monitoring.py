"""Inference monitoring and input drift detection.

Every prediction served by the demo is logged (latency, input statistics,
prediction statistics) to a JSONL file. Incoming image statistics are
compared to reference statistics computed on the LEVIR-CD+ training set:
if the input distribution deviates too much from what the model saw during
training, the prediction is flagged as potentially unreliable.

The drift check is intentionally simple and dependency-free so it can run on HF Spaces at zero cost.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

# Absolute deviation, in training-set standard deviations, above which
# a per-channel statistic is considered drifted.
Z_THRESHOLD = 3.0


# ---------------------------------------------------------------------------
# Input statistics
# ---------------------------------------------------------------------------

def image_stats(arr: np.ndarray) -> dict:
    """Per-channel statistics of an RGB uint8 image (H, W, 3)."""
    arr = arr.astype(np.float32)
    return {
        "mean": [float(arr[..., c].mean()) for c in range(3)],
        "std": [float(arr[..., c].std()) for c in range(3)],
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
    }


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    drifted: bool
    reasons: list[str] = field(default_factory=list)


def check_drift(stats: dict, reference: dict, z_threshold: float = Z_THRESHOLD) -> DriftReport:
    """Compare per-channel input stats to the training reference.

    `reference` holds, for each channel, the training-set distribution of
    per-image means: {"channel_mean": [m_R, m_G, m_B],
                      "channel_mean_std": [s_R, s_G, s_B], ...}
    An input is drifted when one of its channel means lies more than
    `z_threshold` standard deviations away from the training distribution.
    """
    reasons = []
    names = ["R", "G", "B"]
    for c in range(3):
        ref_mean = reference["channel_mean"][c]
        ref_std = max(reference["channel_mean_std"][c], 1e-6)
        z = abs(stats["mean"][c] - ref_mean) / ref_std
        if z > z_threshold:
            reasons.append(
                f"channel {names[c]}: mean {stats['mean'][c]:.1f} is {z:.1f} std "
                f"from training mean {ref_mean:.1f}"
            )
    return DriftReport(drifted=len(reasons) > 0, reasons=reasons)


def load_reference(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Inference logger
# ---------------------------------------------------------------------------

class InferenceLogger:
    """Appends one JSON line per served prediction."""

    def __init__(self, log_path: str = "inference_log.jsonl"):
        self.log_path = log_path

    def log(
        self,
        latency_s: float,
        stats_t1: dict,
        stats_t2: dict,
        change_ratio: float,
        drift: DriftReport,
    ) -> dict:
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "latency_s": round(latency_s, 3),
            "input_t1": stats_t1,
            "input_t2": stats_t2,
            "predicted_change_ratio": round(change_ratio, 4),
            "drift_detected": drift.drifted,
            "drift_reasons": drift.reasons,
        }
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass  # logging must never break inference
        return record
