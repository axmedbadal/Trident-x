from typing import Any, Dict

from config.settings import settings


class PartialExitEngine:
    def __init__(self):
        self.levels = [
            ("tp1", 1.0, 0.25, None),
            ("tp2", 2.0, 0.25, None),
            ("tp3", None, 0.20, 2.0),
            ("tp4", None, 0.20, 2.618),
            ("tp5", None, 0.10, 3.618),
        ]

    def check(self, position: Dict[str, Any], features: Dict[str, float]) -> (float, str, Dict):
        entry = position["entry_price"]
        sl = position["stop_loss"]
        atr = features.get("atr_14", 0.0)
        close = features.get("close", 0.0)
        direction = position["direction"]
        if entry <= 0 or close <= 0 or atr <= 0:
            return 0.0, "", position

        r = abs(entry - sl)
        exit_pct = 0.0
        reason = ""
        updated = dict(position)

        for i, (key, rr, pct, atr_mult) in enumerate(self.levels, start=1):
            hit_key = f"tp{i}_hit"
            if updated.get(hit_key, 0):
                continue
            target = entry + (r * rr if direction == "BUY" else -r * rr) if rr else (
                entry + atr * atr_mult if direction == "BUY" else entry - atr * atr_mult
            )
            if direction == "BUY" and close >= target:
                exit_pct += pct
                updated[hit_key] = 1
                if key == "tp1":
                    updated["stop_loss"] = entry
                reason += f"{key},"
            elif direction == "SELL" and close <= target:
                exit_pct += pct
                updated[hit_key] = 1
                if key == "tp1":
                    updated["stop_loss"] = entry
                reason += f"{key},"

        return exit_pct, reason.rstrip(","), updated


partial_exit = PartialExitEngine()
