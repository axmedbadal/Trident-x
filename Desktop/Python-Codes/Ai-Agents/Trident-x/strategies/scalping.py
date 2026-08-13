"""Scalping engine.

Scans all timeframes for context, then produces a fast 5m executable signal
built for quick entries: volume burst, VWAP reclaim, EMA pullback in a
trending HTF, range breakout, and extreme fade with volatility filters.
"""
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List

from config.settings import settings
from features.engineer import features

logger = logging.getLogger(__name__)


class ScalpingEngine:
    name = "scalping"
    permitted_regimes = {"TRENDING_UP", "TRENDING_DOWN", "MEAN_REVERTING", "ACCUMULATION"}
    weight = 0.25

    def __init__(self):
        self._cooldown_until: Dict[str, float] = defaultdict(float)

    def generate(self, symbol: str, features_5m: Dict[str, float], funding: float = 0.0) -> Dict[str, Any]:
        if not settings.SCALPING_ENABLED:
            return self._neutral("disabled")
        now = time.time()
        if now < self._cooldown_until[symbol]:
            return self._neutral("cooldown")

        close = features_5m.get("close", 0.0)
        atr_pct = features_5m.get("atr_pct", 0.0)
        vol_ratio = features_5m.get("vol_sma20_ratio", 0.0)
        if close <= 0:
            return self._neutral("no_data")

        # Volatility filter: skip dead markets and erratic wide-bar chop
        if atr_pct < settings.SCALPING_MIN_ATR_PCT:
            return self._neutral("too_quiet")
        if atr_pct > settings.SCALPING_MAX_ATR_PCT:
            return self._neutral("too_volatile")

        # HTF trend context across all timeframes
        htf_dir = self._htf_context(symbol)
        if htf_dir == "NEUTRAL":
            return self._neutral("htf_no_context")

        direction = "NEUTRAL"
        conf = 0.0
        rationale: List[str] = []
        f = features_5m
        ema9, ema20 = f.get("ema_9", 0.0), f.get("ema_20", 0.0)
        ema50 = f.get("ema_50", 0.0)
        vwap_dev = f.get("vwap_dev", 0.0)
        rsi = f.get("rsi_14", 50.0)
        stoch_k, stoch_d = f.get("stoch_k", 50.0), f.get("stoch_d", 50.0)
        macd_hist = f.get("macd_hist", 0.0)
        adx = f.get("adx", 0.0)

        # A) Pullback continuation in a trending HTF
        if htf_dir == "LONG":
            if ema9 > 0 and ema20 > 0 and ema9 > ema20 and 45 < rsi < 62 and stoch_k > stoch_d:
                direction = "BUY"
                conf = 0.55
                rationale.append("pullback_continuation_bull")
            elif vol_ratio >= settings.SCALPING_MIN_VOLUME_RATIO and vwap_dev > 0 and macd_hist > 0 and rsi < 68:
                direction = "BUY"
                conf = 0.58
                rationale.append("bull_momentum_burst")
        elif htf_dir == "SHORT":
            if ema9 > 0 and ema20 > 0 and ema9 < ema20 and 38 < rsi < 55 and stoch_k < stoch_d:
                direction = "SELL"
                conf = 0.55
                rationale.append("pullback_continuation_bear")
            elif vol_ratio >= settings.SCALPING_MIN_VOLUME_RATIO and vwap_dev < 0 and macd_hist < 0 and rsi > 32:
                direction = "SELL"
                conf = 0.58
                rationale.append("bear_momentum_burst")

        # B) Trend continuation: price above EMAs with ADX strength + RSI momentum.
        #    Broad trigger with measured edge; fires frequently inside trends.
        if direction == "NEUTRAL":
            if htf_dir == "LONG" and ema9 > ema20 and ema20 > 0 and rsi > 52 and adx > 20 and close > ema20:
                direction = "BUY"
                conf = 0.52
                rationale.append("bull_continuation")
                if vwap_dev > 0:
                    conf += 0.05
                    rationale.append("above_vwap")
                if stoch_k > stoch_d and macd_hist > 0:
                    conf += 0.05
                    rationale.append("stoch+macd_bull")
            elif htf_dir == "SHORT" and ema9 < ema20 and ema9 > 0 and rsi < 48 and adx > 20 and close < ema20:
                direction = "SELL"
                conf = 0.52
                rationale.append("bear_continuation")
                if vwap_dev < 0:
                    conf += 0.05
                    rationale.append("below_vwap")
                if stoch_k < stoch_d and macd_hist < 0:
                    conf += 0.05
                    rationale.append("stoch+macd_bear")

        # C) Range breakout / VWAP reclaim with volume confirmation
        if direction == "NEUTRAL":
            range_high = f.get("donchian_upper", 0.0)
            range_low = f.get("donchian_lower", 0.0)
            if range_high > 0 and vol_ratio >= settings.SCALPING_MIN_VOLUME_RATIO:
                if close > range_high and ema50 > 0 and close > ema50 and rsi < 75:
                    direction = "BUY"
                    conf = 0.55
                    rationale.append("range_breakout_bull")
                elif close < range_low and ema50 > 0 and close < ema50 and rsi > 25:
                    direction = "SELL"
                    conf = 0.55
                    rationale.append("range_breakout_bear")

        # D) Extreme fade, long side only. Short fades measured a negative edge.
        if direction == "NEUTRAL" and vwap_dev < -1.5 and rsi < 28 and stoch_k < 20 and close > f.get("swing_low_20", close):
            direction = "BUY"
            conf = 0.50
            rationale.append("oversold_extreme_fade")

        if direction == "NEUTRAL":
            return self._neutral("no_scalp_setup")

        # Volume + microstructure boosters
        if vol_ratio > 2.0:
            conf += 0.05
            rationale.append(f"vol={vol_ratio:.1f}x")
        vpin = f.get("vpin", 0.0)
        if vpin > 1.5:
            conf += 0.05
            rationale.append("vpin_flow")
        if f.get("spread_approx", 0.0) > 0 and f["spread_approx"] < 0.3:
            conf += 0.05
            rationale.append("tight_spread")

        if funding > 0.0001 and direction == "SELL":
            conf += 0.05
        elif funding < -0.0001 and direction == "BUY":
            conf += 0.05
        elif abs(funding) > 0.0001:
            conf -= 0.10

        conf = min(max(conf, 0.0), 1.0)
        if conf < settings.SCALPING_MIN_CONFIDENCE:
            return self._neutral("low_confidence")
        self._cooldown_until[symbol] = now + settings.SCALPING_COOLDOWN_SECONDS
        return {
            "engine": self.name,
            "direction": direction,
            "confidence": conf,
            "weight": self.weight,
            "rationale": " | ".join(rationale) if rationale else self.name,
        }

    def _htf_context(self, symbol: str) -> str:
        bull = 0
        bear = 0
        for tf in settings.SCALPING_TFS:
            if tf == settings.PRIMARY_TF:
                continue
            feats = features.compute(symbol, tf)
            if not feats:
                continue
            ema9, ema20 = feats.get("ema_9", 0.0), feats.get("ema_20", 0.0)
            adx = feats.get("adx", 0.0)
            rsi = feats.get("rsi_14", 50.0)
            if adx < 18:
                continue
            if ema9 > ema20 and rsi > 50 and feats.get("supertrend", 0) >= 0:
                bull += 1
            elif ema9 < ema20 and rsi < 50 and feats.get("supertrend", 0) <= 0:
                bear += 1
        if bull > bear:
            return "LONG"
        if bear > bull:
            return "SHORT"
        return "NEUTRAL"

    def _neutral(self, reason: str = "no_setup") -> Dict[str, Any]:
        return {
            "engine": self.name,
            "direction": "NEUTRAL",
            "confidence": 0.0,
            "weight": self.weight,
            "rationale": reason,
        }


scalping_engine = ScalpingEngine()
