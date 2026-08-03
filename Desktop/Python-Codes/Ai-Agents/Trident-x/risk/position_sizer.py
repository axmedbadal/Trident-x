import logging
import time
from typing import Dict, List, Optional

import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


class PositionSizer:
    def __init__(self):
        self._last_atr_rank_update = 0.0
        self._atr_rankings: Dict[str, int] = {}
        self._global_vol_cap = False

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

        # Tier 3a: volatility targeting — size keeps per-trade risk near target ATR%.
        # Higher ATR% than target shrinks size; lower ATR% grows it back toward the cap.
        vol_target = settings.VOL_TARGET_ATR_PCT / 100.0
        vol_target_mult = max(settings.VOL_SCALE_FLOOR, min(1.0, vol_target / max(atr_20d_pct, 1e-9)))

        corr = self._avg_correlation(features.get("_symbol", ""), open_positions)
        corr_penalty = max(0.5, 1.0 - corr * 0.5)

        # Phase 2: ATR pair-ranking multiplier
        symbol = signal.get("symbol", "")
        atr_rank_mult = self._atr_rank_multiplier(symbol)
        if self._global_vol_cap:
            atr_rank_mult = min(atr_rank_mult, 0.5)

        if win_rate_30 < 0.55:
            wr_mult = 0.5
        elif win_rate_30 < 0.60:
            wr_mult = 0.75
        else:
            wr_mult = 1.0

        fraction = kelly * dd_scale * vol_mult * regime_mult * corr_penalty * wr_mult * atr_rank_mult * vol_target_mult
        fraction = min(fraction, settings.MAX_POSITION_EQUITY_PCT)
        risk_based_size = equity * fraction
        max_size = equity * settings.MAX_POSITION_EQUITY_PCT
        return min(risk_based_size, max_size)

    def _atr_rank_multiplier(self, symbol: str) -> float:
        """Phase 2: Rank pairs by 14-period ATR and return volatility multiplier."""
        now = time.time()
        if now - self._last_atr_rank_update > 86400 or not self._atr_rankings:
            self._recalculate_atr_rankings()
            self._last_atr_rank_update = now
        rank = self._atr_rankings.get(symbol, 2)
        return {1: 1.0, 2: 0.75, 3: 0.5}.get(rank, 0.5)

    def _recalculate_atr_rankings(self):
        """Compute ATR for each pair and rank from lowest (1) to highest (3)."""
        from features.engineer import features, FeatureEngineer
        import pandas as pd
        atr_values: Dict[str, float] = {}
        global_cap = False
        for sym in settings.PAIRS:
            f = features.compute(sym, settings.PRIMARY_TF)
            atr = f.get("atr_14", 0.0) if f else 0.0
            atr_values[sym] = atr
            # Global vol cap: if current ATR > 2x 30-period average
            buf = features.buffers.get(f"{sym}:{settings.PRIMARY_TF}", [])
            if len(buf) >= 30:
                df = pd.DataFrame(buf)
                atr_series = FeatureEngineer._atr(df["high"], df["low"], df["close"], 14)
                atr_30 = atr_series.iloc[-30:].values
                atr_30 = atr_30[~np.isnan(atr_30)]
                if len(atr_30) > 0 and atr > 2.0 * (float(atr_30.mean())):
                    global_cap = True
        self._global_vol_cap = global_cap
        if global_cap:
            logger.info("GLOBAL_VOLATILITY_CAP_ACTIVE")
        sorted_pairs = sorted(atr_values, key=atr_values.get, reverse=False)
        self._atr_rankings = {sym: i + 1 for i, sym in enumerate(sorted_pairs)}

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
