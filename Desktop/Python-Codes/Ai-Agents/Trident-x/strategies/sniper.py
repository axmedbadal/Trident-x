import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List

from config.settings import settings
from features.engineer import features

logger = logging.getLogger(__name__)


class SniperEngine:
    """
    Multi-timeframe momentum-breakout strategy.

    Scores ~20 technical tags across 5m/15m/1h/4h timeframes, fires when
    aggregated score >= activation threshold, and confirms via 5-tick bias
    stability. Includes per-symbol cooldown and asset gating.
    """

    name = "sniper"
    permitted_regimes = {"TRENDING_UP", "TRENDING_DOWN", "MEAN_REVERTING", "ACCUMULATION", "DISTRIBUTION"}
    weight = 0.45

    def __init__(self):
        self._cooldown_until: Dict[str, float] = defaultdict(float)
        self._bias_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self._last_fired: Dict[str, float] = defaultdict(float)

    def generate(self, symbol: str, features_5m: Dict[str, float], funding: float = 0.0) -> Dict[str, Any]:
        if not settings.SNIPER_ENABLED or symbol not in settings.SNIPER_ENABLED_ASSETS:
            return self._neutral("asset_disabled")

        # Cooldown gate
        now = time.time()
        if now < self._cooldown_until[symbol]:
            return self._neutral("cooldown")

        # Multi-timeframe scoring
        tf_scores: Dict[str, Dict[str, Any]] = {}
        for tf in settings.SNIPER_TFS:
            tf_feats = features.compute(symbol, tf)
            if tf_feats:
                tf_scores[tf] = self._score_timeframe(tf_feats)
            else:
                tf_scores[tf] = {"score": 0.0, "bias": "NEUTRAL", "tags": []}

        total_score = sum(s["score"] for s in tf_scores.values())
        max_single_score = max((s["score"] for s in tf_scores.values()), default=0.0)

        # Determine bias from aggregated score
        if total_score >= settings.SNIPER_ACTIVATION_THRESHOLD or max_single_score >= settings.SNIPER_ACTIVATION_THRESHOLD:
            bias = "LONG" if total_score >= 0 else "NEUTRAL"
            if bias == "NEUTRAL" and max_single_score > 0:
                bias = "LONG"
        elif total_score <= -settings.SNIPER_ACTIVATION_THRESHOLD or max_single_score <= -settings.SNIPER_ACTIVATION_THRESHOLD:
            bias = "SHORT"
        else:
            bias = "NEUTRAL"

        # Stability confirmation: 5 consecutive same biases
        history = self._bias_history[symbol]
        if bias != "NEUTRAL":
            history.append(bias)
        else:
            # Reset history on neutral
            history.clear()

        if len(history) < settings.SNIPER_STABILITY_WINDOW:
            return self._neutral("forming")
        if not all(b == bias for b in history):
            return self._neutral("unstable")

        # Entry-missed check: if price moved >2% beyond recent range, skip
        close = features_5m.get("close", 0.0)
        atr = features_5m.get("atr_14", 0.0)
        if close > 0 and atr > 0 and self._entry_missed(symbol, close, bias, atr):
            history.clear()
            return self._neutral("entry_missed")

        # Direction and confidence
        direction = "BUY" if bias == "LONG" else "SELL"

        # Confidence scaling
        conf = min(abs(total_score) / 20.0, 1.0)
        conf += 0.05 * sum(1 for s in tf_scores.values() if s["bias"] == bias)
        if settings.SNIPER_USE_MTF_ALIGNMENT:
            aligned = sum(1 for s in tf_scores.values() if s["bias"] == bias)
            if aligned >= 3:
                conf += 0.10

        # Funding adjustment
        if funding > 0.0001 and direction == "BUY":
            conf -= 0.10
        elif funding < -0.0001 and direction == "SELL":
            conf -= 0.10
        elif abs(funding) > 0.0001:
            conf += 0.05

        conf = min(max(conf, 0.0), 1.0)

        # Cooldown starts now
        self._cooldown_until[symbol] = now + settings.SNIPER_COOLDOWN_SECONDS
        self._last_fired[symbol] = now

        rationale = [
            f"score={total_score:.1f}",
            f"max_tf={max_single_score:.1f}",
            f"aligned={sum(1 for s in tf_scores.values() if s['bias'] == bias)}",
        ]
        top_tags = []
        for tf, s in tf_scores.items():
            for tag in s["tags"]:
                top_tags.append(f"{tf}:{tag}")
        rationale.append(" | ".join(top_tags[:5]))

        return {
            "engine": self.name,
            "direction": direction,
            "confidence": conf,
            "weight": self.weight,
            "rationale": "; ".join(rationale),
        }

    def _score_timeframe(self, f: Dict[str, float]) -> Dict[str, Any]:
        tags: List[str] = []
        score = 0.0

        # Trend / MA
        if f.get("ema_9", 0) > f.get("ema_20", 0) > f.get("ema_50", 0):
            score += 1.0
            tags.append("ema_bull_stack")
        if f.get("ema_9", 0) < f.get("ema_20", 0) < f.get("ema_50", 0):
            score -= 1.0
            tags.append("ema_bear_stack")
        if f.get("sma_aligned_bull", 0):
            score += 0.8
            tags.append("sma_aligned_bull")
        if f.get("sma_aligned_bear", 0):
            score -= 0.8
            tags.append("sma_aligned_bear")
        close = f.get("close", 0.0)
        ema20 = f.get("ema_20", 0.0)
        if close > 0 and ema20 > 0:
            if close > ema20:
                score += 0.3
                tags.append("price_above_ema20")
            else:
                score -= 0.3
                tags.append("price_below_ema20")
        vwap_dev = f.get("vwap_dev", 0.0)
        if vwap_dev > 0:
            score += 0.3
            tags.append("above_vwap")
        elif vwap_dev < 0:
            score -= 0.3
            tags.append("below_vwap")

        # MACD
        macd_hist = f.get("macd_hist", 0.0)
        if macd_hist > 0:
            score += 0.6
            tags.append("macd_hist_positive")
        elif macd_hist < 0:
            score -= 0.6
            tags.append("macd_hist_negative")
        macd = f.get("macd", 0.0)
        macd_sig = f.get("macd_signal", 0.0)
        if macd > macd_sig:
            score += 0.5
            tags.append("macd_bull_cross")
        elif macd < macd_sig:
            score -= 0.5
            tags.append("macd_bear_cross")

        # RSI
        rsi = f.get("rsi_14", 50.0)
        if 50 < rsi < 70:
            score += 0.4
            tags.append("rsi_bull_zone")
        elif 30 < rsi < 50:
            score -= 0.4
            tags.append("rsi_bear_zone")
        elif rsi > 70:
            score -= 0.5
            tags.append("rsi_overbought")
        elif rsi < 30:
            score += 0.5
            tags.append("rsi_oversold")
        if f.get("rsi_divergence", 0):
            score += 1.0
            tags.append("rsi_divergence_bull")

        # ADX
        adx = f.get("adx", 0.0)
        plus_di = f.get("plus_di", 0.0)
        minus_di = f.get("minus_di", 0.0)
        if adx > 25:
            score += 0.5
            tags.append("adx_strong")
            if plus_di > minus_di:
                score += 0.5
                tags.append("plus_di_dom")
            else:
                score -= 0.5
                tags.append("minus_di_dom")

        # Stochastic
        stoch_k = f.get("stoch_k", 50.0)
        stoch_d = f.get("stoch_d", 50.0)
        if stoch_k > stoch_d:
            score += 0.4
            tags.append("stoch_bull_cross")
        else:
            score -= 0.4
            tags.append("stoch_bear_cross")

        # Supertrend
        if f.get("supertrend", 0) > 0:
            score += 1.0
            tags.append("supertrend_bull")
        else:
            score -= 1.0
            tags.append("supertrend_bear")

        # Volatility
        if f.get("ttm_squeeze", 0):
            score += 0.5
            tags.append("ttm_squeeze")
        if f.get("bb_squeeze", 0):
            score += 0.3
            tags.append("bb_squeeze")
        if f.get("atr_expansion", 0):
            score += 0.4
            tags.append("atr_expansion")

        # Volume
        vol_ratio = f.get("vol_sma20_ratio", 1.0)
        if vol_ratio > 1.5:
            score += (0.5 if score > 0 else -0.5)
            tags.append("volume_surge")
        obv = f.get("obv", 0.0)
        if obv != 0:
            # crude rising/falling proxy
            pass
        if f.get("obv_divergence", 0):
            score -= 0.8
            tags.append("obv_divergence_bear")

        # SMC / Price action
        if f.get("fvg_bull", 0):
            score += 0.8
            tags.append("fvg_bull")
        if f.get("fvg_bear", 0):
            score -= 0.8
            tags.append("fvg_bear")
        if f.get("bos_up", 0):
            score += 0.7
            tags.append("bos_up")
        if f.get("bos_down", 0):
            score -= 0.7
            tags.append("bos_down")
        if f.get("engulfing", 0):
            score += (0.6 if score > 0 else -0.6)
            tags.append("engulfing")

        bias = "LONG" if score > 0 else "SHORT" if score < 0 else "NEUTRAL"
        return {"score": score, "bias": bias, "tags": tags}

    def _entry_missed(self, symbol: str, close: float, bias: str, atr: float) -> bool:
        """Check if price moved >2% beyond the recent entry zone."""
        buf = features.buffers.get(f"{symbol}:{settings.PRIMARY_TF}", [])
        if len(buf) < 10:
            return False
        recent = [c["close"] for c in buf[-10:]]
        avg = sum(recent) / len(recent)
        if avg <= 0:
            return False
        if bias == "LONG" and close > avg * (1 + settings.SNIPER_ENTRY_MISS_PCT):
            return True
        if bias == "SHORT" and close < avg * (1 - settings.SNIPER_ENTRY_MISS_PCT):
            return True
        return False

    def _neutral(self, reason: str = "no setup") -> Dict[str, Any]:
        return {
            "engine": self.name,
            "direction": "NEUTRAL",
            "confidence": 0.0,
            "weight": self.weight,
            "rationale": reason,
        }


sniper_engine = SniperEngine()
