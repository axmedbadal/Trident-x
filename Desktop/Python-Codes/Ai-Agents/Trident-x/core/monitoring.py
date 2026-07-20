import logging
import os
import time
from collections import deque
from typing import Any, Dict, List


class LogCaptureHandler(logging.Handler):
    """Captures recent log records for dashboard display."""

    def __init__(self, capacity: int = 200):
        super().__init__()
        self.capacity = capacity
        self.records: deque = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord):
        try:
            self.records.append({
                "time": int(time.time() * 1000),
                "level": record.levelname,
                "name": record.name,
                "msg": self.format(record),
            })
        except Exception:
            pass

    def get_recent(self, level: str = None, limit: int = 50) -> List[Dict]:
        result = list(self.records)
        if level:
            result = [r for r in result if r["level"] == level]
        return result[-limit:]

    def get_errors(self, limit: int = 20) -> List[Dict]:
        return [r for r in list(self.records) if r["level"] in ("WARNING", "ERROR", "CRITICAL")][-limit:]


class SystemMonitor:
    """Tracks system health metrics for live dashboard."""

    def __init__(self):
        self.start_time = time.time()
        self._last_candle_timestamps: Dict[str, int] = {}
        self._signal_count = 0
        self._trade_count = 0
        self._rest_errors = 0
        self._ws_reconnects = 0
        self._handler: LogCaptureHandler = None

    def setup_log_capture(self, fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"):
        self._handler = LogCaptureHandler()
        self._handler.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(self._handler)
        return self._handler

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def uptime_str(self) -> str:
        s = int(self.uptime_seconds)
        h, rem = divmod(s, 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m}m {s}s"

    def record_candle(self, symbol: str, timestamp: int):
        self._last_candle_timestamps[symbol] = timestamp

    def increment_signals(self):
        self._signal_count += 1

    def increment_trades(self):
        self._trade_count += 1

    def increment_rest_errors(self):
        self._rest_errors += 1

    def increment_ws_reconnects(self):
        self._ws_reconnects += 1

    def stats(self) -> Dict[str, Any]:
        import psutil
        proc = psutil.Process(os.getpid())
        mem_mb = proc.memory_info().rss / 1024 / 1024
        cpu_pct = proc.cpu_percent(interval=None)
        now = int(time.time() * 1000)
        last_candles = {
            sym: {"age_ms": now - ts if ts else -1}
            for sym, ts in self._last_candle_timestamps.items()
        }
        return {
            "uptime_seconds": int(self.uptime_seconds),
            "uptime": self.uptime_str,
            "memory_mb": round(mem_mb, 1),
            "cpu_pct": round(cpu_pct, 1),
            "signal_count": self._signal_count,
            "trade_count": self._trade_count,
            "rest_errors": self._rest_errors,
            "ws_reconnects": self._ws_reconnects,
            "last_candles": last_candles,
            "timestamp": now,
        }

    def recent_logs(self, level: str = None, limit: int = 50) -> List[Dict]:
        if self._handler is None:
            return []
        return self._handler.get_recent(level, limit)

    def recent_errors(self, limit: int = 20) -> List[Dict]:
        if self._handler is None:
            return []
        return self._handler.get_errors(limit)


system_monitor = SystemMonitor()
