import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import aiohttp

from config.settings import settings

logger = logging.getLogger(__name__)
TF_MAP = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400}


class BinanceRestClient:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def _rate_limited_get(self, url: str, params: Dict[str, Any]) -> Any:
        async with self._lock:
            elapsed = time.time() - self._last_request
            if elapsed < 0.05:
                await asyncio.sleep(0.05 - elapsed)
            self._last_request = time.time()
        session = await self._get_session()
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 429:
                    await asyncio.sleep(5)
                    return None
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            logger.warning(f"REST error: {e}")
            return None

    async def fetch_klines(self, symbol: str, timeframe: str, limit: int = 500) -> List[Dict[str, Any]]:
        url = f"{settings.BINANCE_REST_URL}/api/v3/klines"
        params = {"symbol": symbol.upper(), "interval": timeframe, "limit": limit}
        data = await self._rate_limited_get(url, params)
        candles = []
        if not data or not isinstance(data, list):
            return candles
        for row in data:
            try:
                candles.append({
                    "symbol": symbol.upper(),
                    "timeframe": timeframe,
                    "timestamp": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                })
            except Exception:
                continue
        return candles

    async def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        url = f"{settings.BINANCE_REST_URL}/fapi/v1/fundingRate"
        params = {"symbol": symbol.upper(), "limit": 1}
        data = await self._rate_limited_get(url, params)
        if data and isinstance(data, list) and len(data) > 0:
            try:
                return float(data[0]["fundingRate"])
            except Exception:
                pass
        return 0.0

    async def fetch_prices(self) -> Dict[str, float]:
        url = f"{settings.BINANCE_REST_URL}/api/v3/ticker/price"
        prices = {}
        data = await self._rate_limited_get(url, {})
        if data and isinstance(data, list):
            for item in data:
                sym = item.get("symbol", "").upper()
                if sym in settings.PAIRS:
                    try:
                        prices[sym] = float(item["price"])
                    except Exception:
                        pass
        return prices
