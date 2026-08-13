"""Price Action engine.

Scans every configured timeframe and candle structure (swing highs/lows,
market structure, break of structure, fair value gaps, liquidity sweeps,
candlestick patterns, support/resistance) and condenses it into an
executable 5m signal with a confidence score.
"""
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List

from config.settings import settings
from features.engineer import features

logger = logging.getLogger(__name__)


class PriceActionEngine:
    name = "price_action"
    permitted_regimes = {"TRENDING_UP", "TRENDING_DOWN", "MEAN_REVERTING", "ACCUMULATION", "DISTRIBUTION"}
    weight = 0.30

    def __init__(self):
        self._cooldown_until: Dict[str, float] = defaultdict(float)

    def generate(self, symbol: str, features_5m: Dict[str, float], funding: float = 0.0) -> Dict[str, Any]:
        if not settings.PRICE_ACTION_ENABLED:
            return self._neutral("disabled")
        now = time.time()
        if now < self._cooldown_until[symbol]:
            return self._neutral("cooldown")

        close = features_5m.get("close", 0.0)
        if close <= 0:
            return self._neutral("no_data")

        # 1) HTF trend context across all timeframes (heavier weight on higher TFs)
        direction = "NEUTRAL"
        conf = 0.0
        rationale: List[str] = []
        htf_bull_votes = 0
        htf_bear_votes = 0

        for tf in settings.PRICE_ACTION_TFS:
            feats = features.compute(symbol, tf)
            if not feats:
                continue
            bias = self._trend_bias(feats)
            if tf == "5m":
                continue
            if bias == "LONG":
                htf_bull_votes += 1
            elif bias == "SHORT":
                htf_bear_votes += 1

        # 2) Market structure + patterns on the 5m entry timeframe
        buf = features.buffers.get(f"{symbol}:{settings.PRIMARY_TF}", [])
        structure_5m = self._structure(symbol, settings.PRIMARY_TF)
        sweep = self._liquidity_sweep(symbol, features_5m, buf)
        fvg = self._fvg_retest(symbol, features_5m)
        pat = self._candle_patterns(buf, features_5m)
        sr = self._sr_proximity(features_5m)
        bos_up = features_5m.get("bos_up", 0.0)
        bos_down = features_5m.get("bos_down", 0.0)

        # Collect bullish/bearish 5m triggers. BOS + structure are the setups with
        # measured edge; patterns/sweep/FVG add extra conviction when present.
        bull_setups: List[str] = []
        bear_setups: List[str] = []
        if structure_5m == "BULLISH":
            bull_setups.append("mss")
        elif structure_5m == "BEARISH":
            bear_setups.append("mss")
        if bos_up:
            bull_setups.append("bos")
        if bos_down:
            bear_setups.append("bos")
        if sweep == "BULL":
            bull_setups.append("sweep")
        elif sweep == "BEAR":
            bear_setups.append("sweep")
        if fvg == "BULL":
            bull_setups.append("fvg")
        elif fvg == "BEAR":
            bear_setups.append("fvg")
        if pat.get("bull"):
            bull_setups.extend(pat["bull"])
        if pat.get("bear"):
            bear_setups.extend(pat["bear"])

        # 3) Directional resolution. Entries require break-of-structure and/or
        #    market structure in the HTF trend direction. Patterns/sweep/FVG
        #    only add conviction. Low-conviction triggers are discarded.
        if htf_bull_votes >= 2 and ("mss" in bull_setups or "bos" in bull_setups) and structure_5m != "BEARISH":
            direction = "BUY"
            conf = 0.55 + 0.05 * (htf_bull_votes - 2)
            rationale.append(f"htf_bull_votes={htf_bull_votes}")
            if "mss" in bull_setups:
                conf += 0.05
                rationale.append("5m_ms_bull")
            if "bos" in bull_setups:
                conf += 0.05
                rationale.append("5m_bos_up")
            if "sweep" in bull_setups:
                conf += 0.05
                rationale.append("bull_sweep")
            if "fvg" in bull_setups:
                conf += 0.05
                rationale.append("fvg_retest_bull")
            extra = [t for t in bull_setups if t not in ("mss", "bos", "sweep", "fvg")]
            if extra:
                conf += min(0.10, 0.03 * len(extra))
                rationale.append(" | ".join(extra[:3]))
            if sr == "SUPPORT":
                conf += 0.05
                rationale.append("at_support")
        elif htf_bear_votes >= 2 and ("mss" in bear_setups or "bos" in bear_setups) and structure_5m != "BULLISH":
            direction = "SELL"
            conf = 0.55 + 0.05 * (htf_bear_votes - 2)
            rationale.append(f"htf_bear_votes={htf_bear_votes}")
            if "mss" in bear_setups:
                conf += 0.05
                rationale.append("5m_ms_bear")
            if "bos" in bear_setups:
                conf += 0.05
                rationale.append("5m_bos_down")
            if "sweep" in bear_setups:
                conf += 0.05
                rationale.append("bear_sweep")
            if "fvg" in bear_setups:
                conf += 0.05
                rationale.append("fvg_retest_bear")
            extra = [t for t in bear_setups if t not in ("mss", "bos", "sweep", "fvg")]
            if extra:
                conf += min(0.10, 0.03 * len(extra))
                rationale.append(" | ".join(extra[:3]))
            if sr == "RESISTANCE":
                conf += 0.05
                rationale.append("at_resistance")

        if direction == "NEUTRAL":
            return self._neutral("no_pa_setup")

        # 4) Volume confirmation + funding edge
        vol_ratio = features_5m.get("vol_sma20_ratio", 1.0)
        if vol_ratio > 1.5:
            conf += 0.05
            rationale.append(f"vol={vol_ratio:.1f}x")
        if funding > 0.0001 and direction == "SELL":
            conf += 0.05
        elif funding < -0.0001 and direction == "BUY":
            conf += 0.05
        elif abs(funding) > 0.0001:
            conf -= 0.10

        conf = min(max(conf, 0.0), 1.0)
        if conf < settings.PRICE_ACTION_MIN_CONFIDENCE:
            return self._neutral("low_confidence")

        self._cooldown_until[symbol] = now + settings.PRICE_ACTION_COOLDOWN_SECONDS
        return {
            "engine": self.name,
            "direction": direction,
            "confidence": conf,
            "weight": self.weight,
            "rationale": " | ".join(rationale) if rationale else self.name,
        }

    def _trend_bias(self, f: Dict[str, float]) -> str:
        ema9, ema20, ema50 = f.get("ema_9", 0.0), f.get("ema_20", 0.0), f.get("ema_50", 0.0)
        adx = f.get("adx", 0.0)
        if adx < 18:
            return "NEUTRAL"
        if ema9 > ema20 > ema50 and f.get("close", 0.0) > ema20:
            return "LONG"
        if ema9 < ema20 < ema50 and f.get("close", 0.0) < ema20:
            return "SHORT"
        if f.get("supertrend", 0) > 0:
            return "LONG"
        if f.get("supertrend", 0) < 0:
            return "SHORT"
        return "NEUTRAL"

    def _structure(self, symbol: str, tf: str) -> str:
        """Classify 5m structure from the most recent swing highs/lows."""
        buf = features.buffers.get(f"{symbol}:{tf}", [])
        if len(buf) < 12:
            return "NEUTRAL"
        highs = [(i, c["high"]) for i, c in enumerate(buf) if self._is_swing_high(buf, i)]
        lows = [(i, c["low"]) for i, c in enumerate(buf) if self._is_swing_low(buf, i)]
        if len(highs) < 2 or len(lows) < 2:
            return "NEUTRAL"
        hh = highs[-1][1] > highs[-2][1]
        hl = lows[-1][1] > lows[-2][1]
        lh = highs[-1][1] < highs[-2][1]
        ll = lows[-1][1] < lows[-2][1]
        if hh and hl:
            return "BULLISH"
        if lh and ll:
            return "BEARISH"
        return "NEUTRAL"

    @staticmethod
    def _is_swing_high(buf: List[Dict], i: int) -> bool:
        if i == 0 or i == len(buf) - 1:
            return False
        return buf[i]["high"] > buf[i - 1]["high"] and buf[i]["high"] > buf[i + 1]["high"]

    @staticmethod
    def _is_swing_low(buf: List[Dict], i: int) -> bool:
        if i == 0 or i == len(buf) - 1:
            return False
        return buf[i]["low"] < buf[i - 1]["low"] and buf[i]["low"] < buf[i + 1]["low"]

    def _liquidity_sweep(self, symbol: str, f: Dict[str, float], buf: List[Dict]) -> str:
        """EQH/EQL sweep: wick through a recent swing level, then close back inside."""
        if len(buf) < 12 or f.get("close", 0.0) <= 0:
            return ""
        last_c = buf[-1]
        high = f.get("high", 0.0) or last_c.get("high", 0.0)
        low = f.get("low", 0.0) or last_c.get("low", 0.0)
        swings_high = [buf[i]["high"] for i in range(len(buf)) if self._is_swing_high(buf, i)]
        swings_low = [buf[i]["low"] for i in range(len(buf)) if self._is_swing_low(buf, i)]
        close = f["close"]
        tol = settings.PRICE_ACTION_SWEEP_TOLERANCE
        if swings_high and abs(high - max(swings_high)) / close < tol and close < max(swings_high):
            return "BEAR"
        if swings_low and abs(low - min(swings_low)) / close < tol and close > min(swings_low):
            return "BULL"
        return ""

    def _fvg_retest(self, symbol: str, f: Dict[str, float]) -> str:
        """Fair value gap formed recently; price currently inside/retesting it."""
        buf = features.buffers.get(f"{symbol}:{settings.PRIMARY_TF}", [])
        n = len(buf)
        if n < settings.PRICE_ACTION_FVG_LOOKBACK + 3:
            return ""
        close = f.get("close", 0.0)
        if close <= 0:
            return ""
        for i in range(n - settings.PRICE_ACTION_FVG_LOOKBACK, n - 1):
            if i < 2:
                continue
            # Bullish FVG: candle i-2 high < candle i low
            if buf[i - 2]["high"] < buf[i]["low"] and buf[i - 1]["close"] > buf[i - 2]["high"]:
                if close >= buf[i - 2]["high"]:
                    return "BULL"
            # Bearish FVG: candle i-2 low > candle i high
            if buf[i - 2]["low"] > buf[i]["high"] and buf[i - 1]["close"] < buf[i - 2]["low"]:
                if close <= buf[i - 2]["low"]:
                    return "BEAR"
        return ""

    def _candle_patterns(self, buf: List[Dict], f: Dict[str, float]) -> Dict[str, List[str]]:
        pat: Dict[str, List[str]] = {"bull": [], "bear": []}
        if len(buf) < 3:
            return pat
        c0, c1, c2 = buf[-1], buf[-2], buf[-3]
        o0, c0c, h0, l0 = c0["open"], c0["close"], c0["high"], c0["low"]
        o1, c1c = c1["open"], c1["close"]
        body0 = c0c - o0
        body1 = c1c - o1

        # Engulfing
        if abs(body0) > abs(body1) and (body0 > 0) != (body1 > 0) and abs(body0) > abs(body1) * 1.1:
            if body0 > 0 and c0c > c1["high"]:
                pat["bull"].append("bull_engulfing")
            elif body0 < 0 and c0c < c1["low"]:
                pat["bear"].append("bear_engulfing")

        # Pin bar / hammer / shooting star
        rng = max(h0 - l0, 1e-9)
        upper_wick = h0 - max(o0, c0c)
        lower_wick = min(o0, c0c) - l0
        if lower_wick > 2 * body0 and lower_wick > 0.5 * rng:
            pat["bull"].append("hammer")
        elif upper_wick > 2 * abs(body0) and upper_wick > 0.5 * rng:
            pat["bear"].append("shooting_star")

        # Inside bar breakout
        if h0 <= c1["high"] and l0 >= c1["low"] and rng < (c1["high"] - c1["low"]):
            pat["bull"].append("inside_bar")
            pat["bear"].append("inside_bar")

        # Doji at level
        if abs(body0) < 0.1 * rng:
            if f.get("swing_high_20", 0.0) > 0 and abs(c0c - f["swing_high_20"]) / c0c < 0.002:
                pat["bear"].append("doji_at_resistance")
            if f.get("swing_low_20", 0.0) > 0 and abs(c0c - f["swing_low_20"]) / c0c < 0.002:
                pat["bull"].append("doji_at_support")

        # Two-candle sequence after a large body (continuation/pullback)
        if body1 < 0 and body0 > 0 and c0c > c1c and c1c < c2["close"]:
            pat["bull"].append("reversal_up")
        if body1 > 0 and body0 < 0 and c0c < c1c and c1c > c2["close"]:
            pat["bear"].append("reversal_down")
        return pat

    def _sr_proximity(self, f: Dict[str, float]) -> str:
        close = f.get("close", 0.0)
        if close <= 0:
            return ""
        sh = f.get("swing_high_20", 0.0)
        sl = f.get("swing_low_20", 0.0)
        if sh > 0 and abs(close - sh) / close < settings.PRICE_ACTION_SWEEP_TOLERANCE * 2:
            return "RESISTANCE"
        if sl > 0 and abs(close - sl) / close < settings.PRICE_ACTION_SWEEP_TOLERANCE * 2:
            return "SUPPORT"
        return ""

    def _neutral(self, reason: str = "no_setup") -> Dict[str, Any]:
        return {
            "engine": self.name,
            "direction": "NEUTRAL",
            "confidence": 0.0,
            "weight": self.weight,
            "rationale": reason,
        }


price_action_engine = PriceActionEngine()
