from typing import Any, Dict


class MeanReversionEngine:
    name = "mean_reversion"
    permitted_regimes = {"MEAN_REVERTING", "ACCUMULATION", "DISTRIBUTION"}
    weight = 0.25

    def generate(self, symbol: str, features: Dict[str, float], funding: float = 0.0) -> Dict[str, Any]:
        rsi = features.get("rsi_14", 50)
        bb_pct = features.get("bb_pct_b", 0.5)
        k = features.get("stoch_k", 50)
        d = features.get("stoch_d", 50)
        vwap_dev = features.get("vwap_dev", 0.0)
        direction = "NEUTRAL"
        conf = 0.0
        rationale = []

        if rsi < 30 and bb_pct < 0.05 and k > d and d < 30:
            direction = "BUY"
            conf = 0.55
            rationale.append("RSI oversold + lower BB + stoch cross")
        elif rsi > 70 and bb_pct > 0.95 and k < d and d > 70:
            direction = "SELL"
            conf = 0.55
            rationale.append("RSI overbought + upper BB + stoch cross")
        else:
            return self._neutral()

        if abs(vwap_dev) > 1.0:
            conf += 0.05
            rationale.append("VWAP deviation")
        if features.get("volume_at_node", 0):
            conf += 0.10
            rationale.append("volume at S/R")
        if features.get("obv_divergence", 0):
            conf += 0.08
            rationale.append("OBV divergence")

        if funding > 0.0001 and direction == "SELL":
            conf += 0.05
        elif funding < -0.0001 and direction == "BUY":
            conf += 0.05
        elif abs(funding) > 0.0001:
            conf -= 0.10

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


mean_reversion_engine = MeanReversionEngine()
