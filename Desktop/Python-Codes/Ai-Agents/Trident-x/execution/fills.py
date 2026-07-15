import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

from config.settings import settings

logger = logging.getLogger(__name__)
SLIPPAGE = {
    "TRENDING_UP": 0.0005,
    "TRENDING_DOWN": 0.0005,
    "MEAN_REVERTING": 0.0003,
    "ACCUMULATION": 0.0003,
    "DISTRIBUTION": 0.0003,
    "VOLATILE": 0.0015,
}
MAKER_FEE = 0.001
TAKER_FEE = 0.001


@dataclass
class Fill:
    symbol: str
    side: str
    price: float
    quantity: float
    fee: float
    slippage: float
    timestamp: int
    filled: bool = True


@dataclass
class Order:
    id: str
    symbol: str
    side: str
    price: float
    quantity: float
    order_type: str
    status: str = "NEW"
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    created_at: int = 0


class FillProvider(ABC):
    @abstractmethod
    async def fill_entry(self, symbol: str, direction: str, price: float, size_usd: float, regime: str, limit: bool = False, timeout_s: int = 0) -> Dict:
        pass

    @abstractmethod
    async def fill_exit(self, position: Dict, price: float, regime: str, reason: str) -> Dict:
        pass

    @abstractmethod
    async def submit_order(self, symbol: str, side: str, quantity: float, order_type: str, price: float = 0.0) -> Order:
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        pass


class BacktestFill(FillProvider):
    def __init__(self):
        self._order_counter = 0
        self._orders: Dict[str, Order] = {}

    async def fill_entry(self, symbol: str, direction: str, price: float, size_usd: float, regime: str, limit: bool = False, timeout_s: int = 0) -> Dict:
        slippage = 0.0 if limit else SLIPPAGE.get(regime, 0.0005)
        fill_price = price * (1 + slippage) if direction == "BUY" else price * (1 - slippage)
        qty = size_usd / fill_price
        fee = size_usd * TAKER_FEE
        return {"price": fill_price, "quantity": qty, "fee": fee, "slippage": slippage, "filled": True}

    async def fill_exit(self, position: Dict, price: float, regime: str, reason: str) -> Dict:
        slippage = SLIPPAGE.get(regime, 0.0005)
        fill_price = price * (1 - slippage) if position["direction"] == "BUY" else price * (1 + slippage)
        qty = position["size_usd"] / position["entry_price"]
        notional = qty * fill_price
        fee = notional * TAKER_FEE
        return {"price": fill_price, "quantity": qty, "fee": fee, "slippage": slippage, "filled": True}

    async def submit_order(self, symbol: str, side: str, quantity: float, order_type: str, price: float = 0.0) -> Order:
        self._order_counter += 1
        order = Order(
            id=f"BT_{self._order_counter}",
            symbol=symbol, side=side, price=price, quantity=quantity,
            order_type=order_type, created_at=int(time.time() * 1000),
        )
        if order_type == "MARKET":
            order.status = "FILLED"
            order.filled_quantity = quantity
            order.filled_price = price
        self._orders[order.id] = order
        return order

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    async def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.status == "NEW":
            order.status = "CANCELED"
            return True
        return False


class PaperFill(BacktestFill):
    pass


class LiveFill(FillProvider):
    def __init__(self):
        self._order_counter = 0
        self._orders: Dict[str, Order] = {}
        logger.warning("LiveFill initialized — exchange integration stub")

    async def fill_entry(self, symbol: str, direction: str, price: float, size_usd: float, regime: str, limit: bool = False, timeout_s: int = 0) -> Dict:
        return await BacktestFill().fill_entry(symbol, direction, price, size_usd, regime, limit=limit, timeout_s=timeout_s)

    async def fill_exit(self, position: Dict, price: float, regime: str, reason: str) -> Dict:
        return await BacktestFill().fill_exit(position, price, regime, reason)

    async def submit_order(self, symbol: str, side: str, quantity: float, order_type: str, price: float = 0.0) -> Order:
        return await BacktestFill().submit_order(symbol, side, quantity, order_type, price)

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        return await BacktestFill().get_order_status(order_id)

    async def cancel_order(self, order_id: str) -> bool:
        return await BacktestFill().cancel_order(order_id)


def get_fill_provider() -> FillProvider:
    if settings.PAPER_TRADING:
        return PaperFill()
    return LiveFill()
