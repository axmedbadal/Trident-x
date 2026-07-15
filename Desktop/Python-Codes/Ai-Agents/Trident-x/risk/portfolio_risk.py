import logging
from typing import Dict, List

import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


class PortfolioRisk:
    def stress_test(self, positions: List[Dict], equity: float) -> float:
        if not positions:
            return 0.0
        loss = 0.0
        for pos in positions:
            size = pos.get("size_usd", 0.0)
            loss += size * 0.10
        return loss / equity if equity else 0.0

    def var95(self, positions: List[Dict], returns: List[float]) -> float:
        if not returns:
            return 0.0
        return np.percentile(returns, 5)

    def exposure(self, positions: List[Dict], equity: float) -> float:
        total = sum(p.get("size_usd", 0.0) for p in positions)
        return total / equity if equity else 0.0


portfolio_risk = PortfolioRisk()
