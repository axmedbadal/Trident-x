import logging
import os
import time
from typing import Any, Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)
KILL_FILE = "kill_switch.json"


class CircuitBreaker:
    def __init__(self):
        self.state = "GREEN"
        self.cooldown_until = 0.0
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.max_dd = 0.0
        self.today = self._day()

    def update(self, equity: float, peak_equity: float, daily_pnl: float, consecutive_losses: int):
        self.daily_pnl = daily_pnl
        self.consecutive_losses = consecutive_losses
        self.max_dd = (peak_equity - equity) / peak_equity if peak_equity else 0.0
        now = time.time()
        if now < self.cooldown_until and self.state in ("ORANGE", "RED"):
            return
        if self._kill_file_present():
            self.state = "KILL"
            return
        if self.consecutive_losses >= 5 or self.daily_pnl <= -settings.MAX_DD_PCT or self.max_dd >= settings.MAX_DD_PCT:
            self.state = "RED"
            self.cooldown_until = now + 4 * 3600
        elif self.consecutive_losses >= 3 or self.daily_pnl <= -0.04:
            self.state = "ORANGE"
            self.cooldown_until = now + 3600
        elif self.consecutive_losses >= 2 or self.daily_pnl <= -0.02:
            self.state = "YELLOW"
        else:
            self.state = "GREEN"

    def can_trade(self) -> bool:
        if self.state == "KILL":
            return False
        if self.state in ("ORANGE", "RED") and time.time() < self.cooldown_until:
            return False
        if self.state == "RED":
            return False
        return True

    @staticmethod
    def _kill_file_present() -> bool:
        return os.path.exists(KILL_FILE)

    @staticmethod
    def _day() -> int:
        return int(time.time() // 86400)


circuit_breaker = CircuitBreaker()
