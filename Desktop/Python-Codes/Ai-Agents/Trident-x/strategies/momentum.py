from typing import Any, Dict


class MomentumEngine:
    name = "momentum"
    permitted_regimes = {"TRENDING_UP", "TRENDING_DOWN"}
    weight = 0.35

    def generate(self, symbol: str, features: Dict[str, float], funding: float = 0.0) -> Dict[str, Any]:
        tags = []
        if features.get("ema_9", 0) > features.get("ema_20", 0) > features.get("ema_50", 0):
            tags.append(("ema_bull_stack", 1.2))
        if features.get("ema_9", 0) < features.get("ema_20", 0) < features.get("ema_50", 0):
            tags.append(("ema_bear_stack", 1.2))
        if features.get("sma_aligned_bull", 0):
            tags.append(("sma_aligned_bull", 1.0))
        if features.get("sma_aligned_bear", 0):
            tags.append(("sma_aligned_bear", 1.0))
        if features.get("adx", 0) > 25 and features.get("plus_di", 0) > features.get("minus_di", 0):
            tags.append(("adx_bull", 1.0))
        if features.get("adx", 0) > 25 and features.get("plus_di", 0) < features.get("minus_di", 0):
            tags.append(("adx_bear", 1.0))
        if features.get("macd_hist", 0) > 0:
            tags.append(("macd_positive", 0.8))
        if features.get("macd_hist", 0) < 0:
            tags.append(("macd_negative", 0.8))
        if features.get("rsi_14", 50) > 50 and features.get("rsi_14", 50) < 70:
            tags.append(("rsi_bull_zone", 0.6))
        if features.get("rsi_14", 50) < 50 and features.get("rsi_14", 50) > 30:
            tags.append(("rsi_bear_zone", 0.6))
        if features.get("stoch_k", 50) > features.get("stoch_d", 50):
            tags.append(("stoch_bull", 0.6))
        else:
            tags.append(("stoch_bear", 0.6))
        if features.get("supertrend", 0) > 0:
            tags.append(("supertrend_bull", 1.0))
        else:
            tags.append(("supertrend_bear", 1.0))
        if features.get("ttm_squeeze", 0):
            tags.append(("ttm_squeeze", 1.0))
        if features.get("vol_sma20_ratio", 0) > 1.5:
            tags.append(("volume_surge", 0.8))
        if features.get("roc_10", 0) > 0:
            tags.append(("roc_positive", 0.5))
        else:
            tags.append(("roc_negative", 0.5))
        if features.get("vwap_dev", 0) > 0:
            tags.append(("above_vwap", 0.5))
        else:
            tags.append(("below_vwap", 0.5))
        if features.get("bb_pct_b", 0.5) > 0.55:
            tags.append(("bb_upper", 0.4))
        if features.get("bb_pct_b", 0.5) < 0.45:
            tags.append(("bb_lower", 0.4))

        if not tags or features.get("adx", 0) < 20:
            return self._neutral()

        bull_score = sum(w for name, w in tags if "bull" in name or "positive" in name)
        bear_score = sum(w for name, w in tags if "bear" in name or "negative" in name)
        total = sum(w for _, w in tags)
        if total < 9.0:
            return self._neutral()

        if bull_score > bear_score * 1.2:
            direction = "BUY"
        elif bear_score > bull_score * 1.2:
            direction = "SELL"
        else:
            return self._neutral()

        conf = max(bull_score, bear_score) / 15.0
        rationale = ["momentum tags"]
        if features.get("rsi_divergence", 0):
            conf += 0.15
            rationale.append("RSI divergence")
        if features.get("ttm_squeeze", 0):
            conf += 0.10
            rationale.append("TTM squeeze")
        # MTF proxy: high timeframe alignment expressed as a feature
        htf_agree = features.get("htf_agree", 0)
        conf += 0.10 * htf_agree
        if htf_agree:
            rationale.append(f"HTF align {htf_agree}")

        if funding > 0.0001 and direction == "BUY":
            conf -= 0.10
        elif funding < -0.0001 and direction == "SELL":
            conf -= 0.10
        elif abs(funding) > 0.0001:
            conf += 0.05

        conf = min(max(conf, 0.0), 1.0)
        return {
            "engine": self.name,
            "direction": direction,
            "confidence": conf,
            "weight": self.weight,
            "rationale": " | ".join(rationale),
        }

    def _neutral(self):
        return {"engine": self.name, "direction": "NEUTRAL", "confidence": 0.0, "weight": self.weight, "rationale": "no setup"}


momentum_engine = MomentumEngine()
