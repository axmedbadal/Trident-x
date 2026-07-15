import logging
from typing import Any, Dict, List

from core.state_manager import state_manager
from strategies.momentum import momentum_engine
from strategies.smc import smc_engine
from strategies.mean_reversion import mean_reversion_engine
from strategies.sniper import sniper_engine

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "sniper": 0.35,
    "smc": 0.30,
    "momentum": 0.20,
    "mean_reversion": 0.15,
}

MIN_WEIGHT = 0.10
MAX_WEIGHT = 0.60
WEIGHT_LEARNING_RATE = 0.05


class ConsensusEngine:
    engines = [momentum_engine, smc_engine, mean_reversion_engine, sniper_engine]
    _dynamic_weights: Dict[str, float] = dict(DEFAULT_WEIGHTS)

    def _get_weight(self, engine_name: str, regime: str) -> float:
        """Fix 3.2: Return regime-aware weight from auto-adjustment."""
        return self._dynamic_weights.get(engine_name, 0.20)

    def combine(self, signals: List[Dict[str, Any]], regime: str) -> Dict[str, Any]:
        valid = [s for s in signals if s.get("direction") != "NEUTRAL" and regime in s.get("permitted_regimes", set())]
        if len(valid) < 2:
            return self._neutral(regime, "insufficient_engine_agreement")
        weights = [self._get_weight(s.get("engine", ""), regime) for s in valid]
        weighted_conf = sum(s["confidence"] * w for s, w in zip(valid, weights)) / max(sum(weights), 1e-9)
        buy_votes = sum(self._get_weight(s.get("engine", ""), regime) for s in valid if s["direction"] == "BUY")
        sell_votes = sum(self._get_weight(s.get("engine", ""), regime) for s in valid if s["direction"] == "SELL")
        if buy_votes > sell_votes * 1.2:
            direction = "BUY"
        elif sell_votes > buy_votes * 1.2:
            direction = "SELL"
        else:
            return self._neutral(regime, "no_majority")
        engines_set = set(s.get("engine") for s in valid)
        if len(engines_set) >= 2:
            weighted_conf += 0.05
        if len(valid) >= 3 and all(s["direction"] == direction for s in valid):
            weighted_conf -= 0.10
        weighted_conf = min(max(weighted_conf, 0.0), 1.0)
        strength = (
            "VERY_STRONG" if weighted_conf >= 0.85
            else "STRONG" if weighted_conf >= 0.70
            else "MODERATE" if weighted_conf >= 0.55
            else "WEAK"
        )
        return {
            "direction": direction,
            "confidence": weighted_conf,
            "strength": strength,
            "engines_agreeing": len(valid),
            "regime": regime,
            "rationale": " | ".join(s.get("rationale", "") for s in valid),
        }

    def adjust_weights(self):
        """Fix 3.2: Auto-adjust engine weights based on win rate performance."""
        for engine_name in ["sniper", "smc", "momentum", "mean_reversion"]:
            total_wr = 0.0
            count = 0
            for regime in ["TRENDING_UP", "TRENDING_DOWN", "MEAN_REVERTING", "ACCUMULATION", "DISTRIBUTION"]:
                perf = state_manager.get_engine_performance(engine_name, regime)
                if perf and perf.get("total_signals", 0) >= 30:
                    total_wr += perf.get("win_rate", 0.5)
                    count += 1
            if count == 0:
                continue
            avg_wr = total_wr / count
            current_w = self._dynamic_weights.get(engine_name, 0.33)
            if avg_wr >= 0.60:
                new_w = current_w + WEIGHT_LEARNING_RATE
            elif avg_wr < 0.45:
                new_w = current_w - WEIGHT_LEARNING_RATE
            else:
                new_w = current_w
            new_w = max(MIN_WEIGHT, min(MAX_WEIGHT, new_w))
            self._dynamic_weights[engine_name] = new_w
        # Normalize to sum to 1.0
        total = sum(self._dynamic_weights.values())
        if total > 0:
            for k in self._dynamic_weights:
                self._dynamic_weights[k] /= total
        logger.debug(f"Engine weights: {self._dynamic_weights}")

    @staticmethod
    def _neutral(regime: str, reason: str) -> Dict[str, Any]:
        return {
            "direction": "NEUTRAL",
            "confidence": 0.0,
            "strength": "NONE",
            "engines_agreeing": 0,
            "regime": regime,
            "rationale": reason,
        }


consensus = ConsensusEngine()
