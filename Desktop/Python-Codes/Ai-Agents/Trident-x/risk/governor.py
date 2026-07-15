import logging
import time
from typing import Dict

from config.settings import settings

logger = logging.getLogger(__name__)


class RiskGovernor:
    def __init__(self):
        self.peak_equity = settings.INITIAL_EQUITY
        self.high_watermark = settings.INITIAL_EQUITY
        self.band = "GREEN"
        self.max_dd = 0.0
        self._last_reset_day = int(time.time() // 86400)

    def update(self, equity: float) -> str:
        # Fix 3.6: Reset high watermark at start of each UTC day
        now_day = int(time.time() // 86400)
        if now_day > self._last_reset_day:
            logger.info(f"New UTC day: resetting high watermark (was {self.high_watermark:.2f})")
            self.high_watermark = equity
            self._last_reset_day = now_day

        if equity > self.high_watermark:
            self.high_watermark = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

        dd = (self.high_watermark - equity) / self.high_watermark if self.high_watermark else 0.0
        if dd > self.max_dd:
            self.max_dd = dd

        if dd < 0.05:
            self.band = "GREEN"
        elif dd < 0.10:
            self.band = "YELLOW"
        elif dd < 0.15:
            self.band = "ORANGE"
        elif dd < 0.20:
            self.band = "RED"
        else:
            self.band = "BLACK"
        return self.band

    def scale(self) -> float:
        return {"GREEN": 1.0, "YELLOW": 0.75, "ORANGE": 0.5, "RED": 0.25, "BLACK": 0.0}.get(self.band, 0.0)


governor = RiskGovernor()
