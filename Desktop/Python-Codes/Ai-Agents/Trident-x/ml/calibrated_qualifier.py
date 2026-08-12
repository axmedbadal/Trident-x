"""Calibrated tabular qualifier for completed price-action snapshots.

This is a research component, not a price-prediction oracle. The artifact is only
considered eligible when it was fitted, calibrated, and persisted by the
chronological training method in this module.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from research.models import SetupSnapshot


FEATURE_VERSION = "pa-v1"
FEATURE_NAMES = (
    "setup_score",
    "regime_confidence",
    "displacement_atr",
    "relative_volume",
    "direction_buy",
    "family_sweep_reclaim",
    "family_structure_breakout",
    "family_compression_breakout",
    "htf_15m_aligned",
    "htf_1h_aligned",
    "structure_aligned",
    "reclaim_confirmed",
    "location_vwap_zone",
    "reward_risk",
)
MIN_TRAINING_ROWS = 250


@dataclass(frozen=True)
class ModelScore:
    probability: Optional[float]
    status: str
    version: str
    calibration_status: str


def snapshot_features(snapshot: SetupSnapshot) -> List[float]:
    """Create one stable model row from a completed snapshot only."""
    direction = snapshot.direction
    return [
        float(snapshot.setup_score),
        float(snapshot.regime_confidence),
        float(snapshot.displacement_atr),
        float(snapshot.relative_volume),
        float(direction == "BUY"),
        float(snapshot.setup_family == "SWEEP_RECLAIM"),
        float(snapshot.setup_family == "STRUCTURE_BREAKOUT"),
        float(snapshot.setup_family == "COMPRESSION_BREAKOUT"),
        float(snapshot.htf_15m == direction and direction != "NEUTRAL"),
        float(snapshot.htf_1h == direction and direction != "NEUTRAL"),
        float(snapshot.structure_direction == direction and direction != "NEUTRAL"),
        float(snapshot.reclaim_confirmed),
        float(snapshot.location == "VWAP_ZONE"),
        float(snapshot.plan.reward_risk),
    ]


class CalibratedQualifier:
    """Load and score only approved, compatible model artifacts."""

    def __init__(self, artifact_path: str = "models/calibrated_qualifier.pkl"):
        self.artifact_path = artifact_path
        self.artifact: Optional[Dict] = None
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.artifact_path):
            return
        try:
            with open(self.artifact_path, "rb") as handle:
                artifact = pickle.load(handle)
            if artifact.get("feature_version") != FEATURE_VERSION:
                return
            if tuple(artifact.get("feature_names", ())) != FEATURE_NAMES:
                return
            if artifact.get("calibration_status") != "CURRENT":
                return
            self.artifact = artifact
        except (OSError, pickle.PickleError, AttributeError, EOFError):
            self.artifact = None

    @property
    def version(self) -> str:
        return str(self.artifact.get("version", "unavailable")) if self.artifact else "unavailable"

    def score(self, snapshot: SetupSnapshot) -> ModelScore:
        if snapshot.data_quality != "OK":
            return ModelScore(None, "DATA_QUALITY_BLOCK", self.version, "UNAVAILABLE")
        if not self.artifact:
            return ModelScore(None, "MODEL_UNAVAILABLE", "unavailable", "UNAVAILABLE")
        try:
            x = np.asarray([snapshot_features(snapshot)], dtype=float)
            probability = float(self.artifact["model"].predict_proba(x)[0, 1])
            if not 0.0 <= probability <= 1.0:
                raise ValueError("out-of-range probability")
            return ModelScore(probability, "CALIBRATED", self.version, "CURRENT")
        except Exception:
            return ModelScore(None, "MODEL_ERROR", self.version, "STALE")


def train_chronological_baseline(
    snapshots: Iterable[SetupSnapshot],
    labels: Iterable[int],
    artifact_path: str,
    version: str,
) -> Dict[str, float]:
    """Fit then calibrate on a later chronological segment.

    Inputs must already be ordered by timestamp. Labels must represent the declared
    target-before-stop outcome from a future barrier; this function never creates
    labels and therefore cannot hide a labeling-policy decision.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import brier_score_loss, precision_score

    snapshots = list(snapshots)
    y = np.asarray(list(labels), dtype=int)
    if len(snapshots) != len(y):
        raise ValueError("snapshots and labels must have the same length")
    if len(snapshots) < MIN_TRAINING_ROWS:
        raise ValueError(f"at least {MIN_TRAINING_ROWS} chronological labeled setups are required")
    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) < 2:
        raise ValueError("labels must contain both binary outcome classes")

    x = np.asarray([snapshot_features(snapshot) for snapshot in snapshots], dtype=float)
    split = int(len(x) * 0.75)
    if split < 100 or len(x) - split < 50:
        raise ValueError("insufficient later calibration segment")
    base = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=17)
    base.fit(x[:split], y[:split])
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    calibrated.fit(x[split:], y[split:])
    calibration_prob = calibrated.predict_proba(x[split:])[:, 1]
    artifact = {
        "version": version,
        "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
        "calibration_status": "CURRENT",
        "model": calibrated,
        "train_rows": int(split),
        "calibration_rows": int(len(x) - split),
        "last_snapshot_timestamp": int(snapshots[-1].timestamp),
    }
    directory = os.path.dirname(artifact_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(artifact_path, "wb") as handle:
        pickle.dump(artifact, handle)
    binary_pred = (calibration_prob >= 0.55).astype(int)
    return {
        "brier_score": float(brier_score_loss(y[split:], calibration_prob)),
        "precision_at_0_55": float(precision_score(y[split:], binary_pred, zero_division=0)),
        "train_rows": float(split),
        "calibration_rows": float(len(x) - split),
    }


qualifier = CalibratedQualifier()
