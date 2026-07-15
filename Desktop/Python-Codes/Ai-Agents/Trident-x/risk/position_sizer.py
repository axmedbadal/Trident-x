import logging
from typing import Dict, List, Optional

import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


class PositionSizer:
    def __init__(self):
        pass

    def size(
        self,
        signal: Dict,
        equity: float,
        features: Dict[str, float],
        regime: str,
        dd_band: str,
        win_rate_30: float,
        open_positions: List[Dict],
    ) -> float:
        sl = features.get("sl_distance", features.get("atr_14", 0.0))
        tp = features.get("tp_distance", sl * 2.0)
        sl = max(sl, 1e-9)
        tp = max(tp, 1e-9)
        p = signal.get("confidence", 0.0)
        b = tp / sl
        kelly = max((p * b - (1 - p)) / b, 0.0) * settings.KELLY_FRACTION

        dd_scale = {"GREEN": 1.0, "YELLOW": 0.75, "ORANGE": 0.5, "RED": 0.25, "BLACK": 0.0}.get(dd_band, 0.0)
        atr_20d_pct = features.get("atr_pct", 0.0) / 100
        vol_mult = 1.0 if atr_20d_pct < 0.05 else (0.75 if atr_20d_pct < 0.10 else 0.5)
        regime_mult = {"TRENDING_UP": 1.2, "TRENDING_DOWN": 1.2, "MEAN_REVERTING": 0.8, "ACCUMULATION": 0.6, "DISTRIBUTION": 0.6}.get(regime, 1.0)

        corr = self._avg_correlation(features.get("_symbol", ""), open_positions)
        corr_penalty = max(0.5, 1.0 - corr * 0.5)

        if win_rate_30 < 0.55:
            wr_mult = 0.5
        elif win_rate_30 < 0.60:
            wr_mult = 0.75
        else:
            wr_mult = 1.0

        fraction = kelly * dd_scale * vol_mult * regime_mult * corr_penalty * wr_mult
        fraction = min(fraction, settings.MAX_POSITION_EQUITY_PCT)
        risk_based_size = equity * fraction
        max_size = equity * settings.MAX_POSITION_EQUITY_PCT
        return min(risk_based_size, max_size)

    def _avg_correlation(self, symbol: str, open_positions: List[Dict]) -> float:
        """Fix 2.2: Compute real rolling correlation if positions exist."""
        if not open_positions or not symbol:
            return 0.0

        from core.state_manager import state_manager
        current_candles = state_manager.load_candles(symbol, "5m", 25)
        if len(current_candles) < 15:
            return 0.0

        current_returns = np.diff([c["close"] for c in current_candles]) / np.array([c["close"] for c in current_candles[:-1]])
        current_returns = current_returns[~np.isnan(current_returns)]
        if len(current_returns) < 10:
            return 0.0

        max_corr = 0.0
        for pos in open_positions:
            pos_sym = pos.get("symbol", "")
            if pos_sym == symbol:
                continue
            pos_candles = state_manager.load_candles(pos_sym, "5m", 25)
            if len(pos_candles) < 15:
                continue
            pos_returns = np.diff([c["close"] for c in pos_candles]) / np.array([c["close"] for c in pos_candles[:-1]])
            pos_returns = pos_returns[~np.isnan(pos_returns)]
            if len(pos_returns) < 10:
                continue

            min_len = min(len(current_returns), len(pos_returns))
            corr = np.corrcoef(current_returns[-min_len:], pos_returns[-min_len:])[0, 1]
            if not np.isnan(corr):
                max_corr = max(max_corr, abs(corr))

        return max_corr


position_sizer = PositionSizer()
