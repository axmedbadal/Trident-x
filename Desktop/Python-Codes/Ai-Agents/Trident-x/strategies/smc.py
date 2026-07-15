from typing import Any, Dict

from config.settings import settings


class SMCEngine:
    name = "smc"
    permitted_regimes = {"TRENDING_UP", "TRENDING_DOWN", "MEAN_REVERTING", "ACCUMULATION", "DISTRIBUTION"}
    weight = 0.40

    def generate(self, symbol: str, features: Dict[str, float], funding: float = 0.0) -> Dict[str, Any]:
        params = settings.ASSET_PARAMS.get(symbol, settings.ASSET_PARAMS["SOLUSDT"])
        close = features.get("close", 0.0)
        atr = features.get("atr_14", 0.0)
        if close <= 0 or atr <= 0:
            return self._neutral()
        swing_high = features.get("swing_high_20", 0.0)
        swing_low = features.get("swing_low_20", 0.0)
        fvg_bull = features.get("fvg_bull", 0.0)
        fvg_bear = features.get("fvg_bear", 0.0)
        displacement = features.get("displacement", 0.0)
        bos_up = features.get("bos_up", 0.0)
        bos_down = features.get("bos_down", 0.0)
        atr_pct = features.get("atr_pct", 0.0)

        tol = params["eqh_tolerance"]
        eqh_near = abs(close - swing_high) / close < tol if swing_high > 0 else False
        eql_near = abs(close - swing_low) / close < tol if swing_low > 0 else False

        direction = "NEUTRAL"
        conf = 0.0
        rationale = []

        # sweep + displacement + FVG combo
        if eqh_near and displacement and bos_down:
            direction = "SELL"
            conf = 0.55
            rationale.append("EQH sweep displacement")
        elif eql_near and displacement and bos_up:
            direction = "BUY"
            conf = 0.55
            rationale.append("EQL sweep displacement")
        elif fvg_bull and bos_up and displacement:
            direction = "BUY"
            conf = 0.50
            rationale.append("bullish FVG+BOS")
        elif fvg_bear and bos_down and displacement:
            direction = "SELL"
            conf = 0.50
            rationale.append("bearish FVG+BOS")

        if direction == "NEUTRAL":
            return self._neutral()

        fvg_dist_factor = params["fvg_distance"]
        if atr_pct > fvg_dist_factor * 0.1:
            conf += 0.05

        # bonus modifiers
        if features.get("atr_expansion", 0.0):
            conf += 0.10
            rationale.append("ATR expansion")
        if features.get("vol_sma20_ratio", 0.0) > 3.0:
            conf += 0.05
            rationale.append("volume surge")
        if features.get("engulfing", 0.0):
            conf += 0.08
            rationale.append("engulfing")

        # funding edge
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
            "rationale": " | ".join(rationale) if rationale else self.name,
        }

    def _neutral(self):
        return {"engine": self.name, "direction": "NEUTRAL", "confidence": 0.0, "weight": self.weight, "rationale": "no setup"}


smc_engine = SMCEngine()
