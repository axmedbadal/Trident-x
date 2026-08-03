import logging
from typing import Any, Dict, List, Optional

from config.settings import settings
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

REGIMES = ["TRENDING_UP", "TRENDING_DOWN", "MEAN_REVERTING", "ACCUMULATION", "DISTRIBUTION"]

MIN_WEIGHT = 0.10
MAX_WEIGHT = 0.60
WEIGHT_LEARNING_RATE = 0.05
MIN_WINRATE = 0.60      # above this -> increase weight
LOW_WINRATE = 0.45      # below this -> decrease weight


class ConsensusEngine:
    engines = [momentum_engine, smc_engine, mean_reversion_engine, sniper_engine]

    # Tier 2: per-regime dynamic weights. Regime -> engine -> weight.
    _regime_weights: Dict[str, Dict[str, float]] = {}
    # Legacy flat view (kept for compatibility); reflects mean of per-regime weights.
    _dynamic_weights: Dict[str, float] = dict(DEFAULT_WEIGHTS)

    def _weights_for(self, regime: str) -> Dict[str, float]:
        if regime not in self._regime_weights or not self._regime_weights[regime]:
            self._regime_weights[regime] = dict(DEFAULT_WEIGHTS)
        return self._regime_weights[regime]

    def _get_weight(self, engine_name: str, regime: str) -> float:
        """Fix 3.2 + Tier 2: regime-aware weight from the per-regime table."""
        return self._weights_for(regime).get(engine_name, 0.20)

    def _normalize(self, weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        if total <= 0:
            return dict(DEFAULT_WEIGHTS)
        return {k: v / total for k, v in weights.items()}

    def _sync_dynamic_weights(self):
        """Keep the legacy flat view equal to the mean of per-regime weights, normalized."""
        out = {}
        for eng in DEFAULT_WEIGHTS:
            vals = [self._weights_for(r).get(eng, 0.20) for r in REGIMES]
            out[eng] = sum(vals) / len(vals)
        self._dynamic_weights = self._normalize(out)

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
        """Tier 2: adapt each regime's engine weights from live engine_performance,
        then keep the legacy flat view consistent."""
        for regime in REGIMES:
            weights = dict(self._weights_for(regime))
            changed = False
            for engine_name in DEFAULT_WEIGHTS:
                perf = state_manager.get_engine_performance(engine_name, regime)
                if perf and perf.get("total_signals", 0) >= settings.ENGINE_EVAL_MIN_SIGNALS:
                    wr = perf.get("win_rate", 0.5)
                    if wr >= MIN_WINRATE:
                        weights[engine_name] = min(MAX_WEIGHT, weights.get(engine_name, 0.2) + WEIGHT_LEARNING_RATE)
                    elif wr < LOW_WINRATE:
                        weights[engine_name] = max(MIN_WEIGHT, weights.get(engine_name, 0.2) - WEIGHT_LEARNING_RATE)
                    changed = True
            if changed:
                self._regime_weights[regime] = self._normalize(weights)
            else:
                self._regime_weights[regime] = self._normalize(weights)
        self._sync_dynamic_weights()

    def apply_backtest_metrics(self, metrics: Dict[str, Dict[str, Dict[str, int]]]):
        """Tier 2: feed backtest produced win/loss counts per engine+regime.

        metrics[engine][regime] = {"wins": int, "losses": int}
        Adapts per-regime weights with the same learning rule as live data.
        """
        for regime in REGIMES:
            weights = dict(self._weights_for(regime))
            changed = False
            for engine_name in DEFAULT_WEIGHTS:
                m = metrics.get(engine_name, {}).get(regime)
                if not m:
                    continue
                wins = m.get("wins", 0)
                losses = m.get("losses", 0)
                n = wins + losses
                if n < 5:  # need enough samples to move a weight
                    continue
                wr = wins / n
                if wr >= MIN_WINRATE:
                    weights[engine_name] = min(MAX_WEIGHT, weights.get(engine_name, 0.2) + WEIGHT_LEARNING_RATE)
                elif wr < LOW_WINRATE:
                    weights[engine_name] = max(MIN_WEIGHT, weights.get(engine_name, 0.2) - WEIGHT_LEARNING_RATE)
                changed = True
            if changed:
                self._regime_weights[regime] = self._normalize(weights)
        self._sync_dynamic_weights()

    def get_weights(self, regime: Optional[str] = None):
        """Expose weights (one regime or all) for transparency / dashboard / tests."""
        if regime is None:
            return {r: dict(self._weights_for(r)) for r in REGIMES}
        return dict(self._weights_for(regime))

    def reset(self):
        self._regime_weights = {}
        self._dynamic_weights = dict(DEFAULT_WEIGHTS)

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