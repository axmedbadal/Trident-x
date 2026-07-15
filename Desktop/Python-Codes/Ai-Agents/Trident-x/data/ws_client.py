import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import websockets

from config.settings import settings

logger = logging.getLogger(__name__)

STREAM_PAIRS = [p.lower() + "@kline_" + settings.PRIMARY_TF for p in settings.PAIRS]


class BinanceWebSocketClient:
    def __init__(self, on_candle: Callable[[Dict], None], on_tick: Optional[Callable[[Dict], None]] = None):
        self.on_candle = on_candle
        self.on_tick = on_tick
        self.ws = None
        self.running = False
        self.last_data_ts = 0
        self.backoff = [1, 2, 4, 8, 16, 30]
        self.connected = False
        self._task = None
        self._keepalive_task = None
        self._last_activity = 0.0
        self._consecutive_empty = 0

    def url(self) -> str:
        return f"{settings.BINANCE_WS_URL}/{'/'.join(STREAM_PAIRS)}"

    async def start(self):
        self.running = True
        self._task = asyncio.create_task(self._run_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive())

    async def stop(self):
        self.running = False
        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self):
        attempt = 0
        while self.running:
            try:
                async with websockets.connect(
                    self.url(),
                    ping_interval=None,
                    close_timeout=10,
                    max_size=2**20,
                ) as ws:
                    self.ws = ws
                    self.connected = True
                    attempt = 0
                    self._last_activity = time.time()
                    logger.info("WebSocket connected")

                    async for msg in ws:
                        if not self.running:
                            break
                        self._last_activity = time.time()

                        # Fix 1.2: Respond to server ping frames
                        if isinstance(msg, bytes):
                            try:
                                await ws.pong(msg)
                            except Exception:
                                pass
                            continue

                        await self._handle(msg)

            except websockets.exceptions.ConnectionClosed as e:
                self.connected = False
                reason = str(e)
                # Fix 1.3: Detect 24h server shutdown
                if "serverShutdown" in reason or getattr(e, "code", 0) == 1001:
                    logger.info("24h server shutdown detected, will reconnect immediately")
                    attempt = 0
                else:
                    wait = self.backoff[min(attempt, len(self.backoff) - 1)]
                    logger.warning(f"WS disconnected: {reason}, reconnect in {wait}s")
                    await asyncio.sleep(wait)
                    attempt += 1
            except Exception as e:
                self.connected = False
                wait = self.backoff[min(attempt, len(self.backoff) - 1)]
                logger.warning(f"WebSocket error: {e}, reconnect in {wait}s")
                await asyncio.sleep(wait)
                attempt += 1

    async def _keepalive(self):
        """Fix 1.2: Send empty pong every 15s if no activity to keep connection alive."""
        while self.running:
            await asyncio.sleep(15)
            if not self.running:
                break
            try:
                if self.ws and self.ws.open:
                    elapsed = time.time() - self._last_activity
                    if elapsed > 60:
                        logger.warning(f"No WS data for {elapsed:.0f}s, sending keepalive pong")
                    await self.ws.pong(b'')
            except Exception:
                pass

    async def _handle(self, raw: str):
        try:
            data = json.loads(raw)

            # Handle combined stream wrapper
            if "stream" in data:
                data = data.get("data", data)

            kline = data.get("k", {})
            if not kline:
                return

            # Fix 1.1: CRITICAL - Only process closed candles
            if not kline.get("x", False):
                return  # Skip forming candles

            sym = kline.get("s", "").upper()
            if not sym:
                return

            candle = {
                "symbol": sym,
                "timeframe": settings.PRIMARY_TF,
                "timestamp": kline["t"],
                "open": float(kline["o"]),
                "high": float(kline["h"]),
                "low": float(kline["l"]),
                "close": float(kline["c"]),
                "volume": float(kline["v"]),
            }

            self.last_data_ts = time.time() * 1000
            asyncio.create_task(self._safe(self.on_candle, candle))

        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.error(f"WS handle error: {e}")

    async def _safe(self, callback, arg):
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(arg)
            else:
                callback(arg)
        except Exception as e:
            logger.error(f"WS callback error: {e}")

    @property
    def data_age_ms(self) -> float:
        """Milliseconds since last data received."""
        if self.last_data_ts == 0:
            return float('inf')
        return time.time() * 1000 - self.last_data_ts
