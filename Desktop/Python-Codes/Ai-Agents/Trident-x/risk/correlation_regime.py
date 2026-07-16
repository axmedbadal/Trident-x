import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from config.settings import settings
from core.state_manager import state_manager
from data.rest_client import BinanceRestClient

logger = logging.getLogger(__name__)

# Map our traded pairs to symbols used for correlation
PAIR_TO_SYMBOL = {
    "SOLUSDT": "SOL",
    "XRPUSDT": "XRP",
    "ADAUSDT": "ADA",
}


class CorrelationRegimeFilter:
    """
    Phase 3: Correlation regime filter.

    Detects when SOL decouples from BTC or when all alts decouple from BTC.
    Suppresses signals when correlation regime is YELLOW or RED.
    """

    def __init__(self, lookback: int = 30):
        self.lookback = lookback
        self._last_update = 0.0
        self._cache_ttl = 3600  # Recalc hourly
        self._regime: Dict[str, Any] = {
            "status": "NORMAL",
            "alert_level": "GREEN",
            "suppressed_pairs": [],
            "correlations": {},
        }
        self._rest_client: Optional[BinanceRestClient] = None

    async def is_suppressed(self, symbol: str) -> (bool, str):
        """Return (suppressed, reason) for a given pair."""
        await self._maybe_update()
        if symbol in self._regime["suppressed_pairs"]:
            return True, f"REGIME_FILTER_{self._regime['status']}"
        return False, ""

    async def get_regime(self) -> Dict[str, Any]:
        await self._maybe_update()
        return dict(self._regime)

    async def _maybe_update(self):
        if time.time() - self._last_update < self._cache_ttl:
            return
        try:
            await self._update_regime()
        except Exception as e:
            logger.warning(f"Correlation regime update failed: {e}")

    async def _update_regime(self):
        """Fetch BTC daily data and compute 30-day correlations."""
        btc_returns = await self._get_btc_returns()
        if btc_returns is None or len(btc_returns) < self.lookback:
            self._regime = {
                "status": "UNKNOWN",
                "alert_level": "GREEN",
                "suppressed_pairs": [],
                "correlations": {},
            }
            self._last_update = time.time()
            return

        correlations: Dict[str, float] = {}
        for sym in settings.PAIRS:
            rets = self._get_pair_returns(sym)
            if rets is None or len(rets) < self.lookback:
                continue
            min_len = min(len(btc_returns), len(rets))
            corr = np.corrcoef(btc_returns[-min_len:], rets[-min_len:])[0, 1]
            if not np.isnan(corr):
                correlations[PAIR_TO_SYMBOL.get(sym, sym)] = float(corr)

        sol_corr = correlations.get("SOL", 0.0)
        ada_corr = correlations.get("ADA", 0.0)
        xrp_corr = correlations.get("XRP", 0.0)

        suppressed: List[str] = []
        status = "NORMAL"
        alert = "GREEN"

        # Condition A: SOL decoupling from BTC while ADA/XRP stay correlated
        if sol_corr < 0.40 and ada_corr > 0.60 and xrp_corr > 0.60:
            status = "SOL_DECOUPLING"
            alert = "YELLOW"
            suppressed = ["SOLUSDT"]
        # Condition B: Broad alt decoupling from BTC
        elif sol_corr < 0.30 and ada_corr < 0.30 and xrp_corr < 0.30:
            status = "BROAD_ALT_DECOUPLING"
            alert = "RED"
            suppressed = ["SOLUSDT", "ADAUSDT", "XRPUSDT"]

        self._regime = {
            "status": status,
            "alert_level": alert,
            "suppressed_pairs": suppressed,
            "correlations": correlations,
        }
        self._last_update = time.time()

        if status != "NORMAL":
            logger.warning(
                f"Correlation regime {alert}: {status}, suppressed={suppressed}, "
                f"correlations=SOL:{sol_corr:.2f} ADA:{ada_corr:.2f} XRP:{xrp_corr:.2f}"
            )

    async def _get_btc_returns(self) -> Optional[np.ndarray]:
        """Load BTCUSDT daily candles from state or fetch via REST."""
        candles = state_manager.load_candles("BTCUSDT", "1d", self.lookback + 5)
        if not candles:
            if self._rest_client is None:
                self._rest_client = BinanceRestClient()
            try:
                candles = await self._rest_client.fetch_klines("BTCUSDT", "1d", self.lookback + 5)
                if candles:
                    state_manager.save_candles(candles)
            except Exception as e:
                logger.debug(f"BTC fetch failed: {e}")
                return None
        if not candles:
            return None
        closes = np.array([c["close"] for c in candles])
        return np.diff(closes) / closes[:-1]

    def _get_pair_returns(self, symbol: str) -> Optional[np.ndarray]:
        """Get daily returns for a traded pair from state."""
        candles = state_manager.load_candles(symbol, "1d", self.lookback + 5)
        if not candles:
            return None
        closes = np.array([c["close"] for c in candles])
        return np.diff(closes) / closes[:-1]


correlation_regime = CorrelationRegimeFilter()
