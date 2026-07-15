from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config.settings import settings


def _safe(f: Dict[str, float], key: str, default: float = 0.0) -> float:
    v = f.get(key, default)
    return default if v is None or v != v else v


class SignalIntegrityGate:
    def __init__(self):
        pass

    def evaluate(
        self,
        signal: Dict[str, Any],
        features: Dict[str, float],
        mtf_signals: Dict[str, str],
        meta_prob: Optional[float],
        data_age_ms: float,
    ) -> (bool, str):
        if signal.get("direction") == "NEUTRAL" or signal.get("confidence", 0) <= 0:
            return False, "neutral"
        vol_ratio = _safe(features, "vol_sma20_ratio", 0.0)
        if vol_ratio < 1.5:
            return False, f"volume_gate:{vol_ratio:.2f}"
        atr_pct = _safe(features, "atr_pct", 0.0)
        if atr_pct < 0.3:
            return False, f"volatility_gate:{atr_pct:.3f}"
        direction = signal["direction"]
        for tf in ["15m", "1h"]:
            ht = mtf_signals.get(tf, "NEUTRAL")
            if ht != "NEUTRAL" and ht != direction:
                return False, f"mtf_mismatch_{tf}:{ht}"
        regime = signal.get("regime", "MEAN_REVERTING")
        regime_thresholds = {
            "TRENDING_UP": 0.65,
            "TRENDING_DOWN": 0.65,
            "MEAN_REVERTING": 0.75,
            "ACCUMULATION": 0.85,
            "DISTRIBUTION": 0.85,
        }
        if signal["confidence"] < regime_thresholds.get(regime, 0.75):
            return False, f"confidence_gate:{signal['confidence']:.2f}"
        if data_age_ms > 60000:
            return False, f"freshness_gate:{int(data_age_ms)}ms"
        if meta_prob is not None and meta_prob < 0.60:
            return False, f"meta_gate:{meta_prob:.2f}"
        if signal.get("engines_agreeing", 0) < 2:
            return False, "independent_confirmation"
        return True, "passed"


class GateResult:
    passed: bool
    reason: str


gate = SignalIntegrityGate()
