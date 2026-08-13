import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from config.settings import settings
from core.state_manager import state_manager
from execution.fills import get_fill_provider
from execution.partial_exit import partial_exit
from risk.position_sizer import position_sizer

logger = logging.getLogger(__name__)

LIMIT_ORDER_POLL_INTERVAL = 2


class ExecutionManager:
    def __init__(self):
        self.fill_provider = get_fill_provider()
        self.positions: List[Dict] = []
        self._pending_limit_orders: Dict[str, Dict] = {}

    def load_positions(self):
        self.positions = state_manager.get_open_positions()

    async def open_position(
        self,
        symbol: str,
        signal: Dict[str, Any],
        features: Dict[str, float],
        equity: float,
        regime: str,
        dd_band: str,
        win_rate_30: float,
        signal_id: int = None,
    ) -> bool:
        open_count = len(self.positions)
        if open_count >= settings.MAX_POSITIONS:
            return False
        size_usd = position_sizer.size(signal, equity, features, regime, dd_band, win_rate_30, self.positions)
        if size_usd <= 0:
            return False
        close = features.get("close", 0.0)
        atr = features.get("atr_14", 0.0)
        params = settings.ASSET_PARAMS.get(symbol, settings.ASSET_PARAMS["SOLUSDT"])
        direction = signal["direction"]
        sl = close - params["sl_mult"] * atr if direction == "BUY" else close + params["sl_mult"] * atr
        tp = close + params["tp_mult"] * atr if direction == "BUY" else close - params["tp_mult"] * atr

        confidence = signal.get("confidence", 0.0)
        fill = None
        if confidence >= 0.85:
            fill = await self.fill_provider.fill_entry(symbol, direction, close, size_usd, regime)
        elif confidence >= 0.70:
            fill = await self._limit_order_with_timeout(symbol, direction, close, size_usd, regime, timeout_s=30)
        elif confidence >= 0.55:
            offset = close * 0.001
            limit_price = close - offset if direction == "BUY" else close + offset
            fill = await self._limit_order_with_timeout(symbol, direction, limit_price, size_usd, regime, timeout_s=120)
        else:
            return False

        if fill is None or not fill.get("filled", False):
            logger.info(f"Order not filled for {symbol}, skipping")
            return False

        pos = {
            "symbol": symbol,
            "direction": direction,
            "entry_price": fill["price"],
            "size_usd": size_usd,
            "remaining_size_usd": size_usd,
            "stop_loss": sl,
            "take_profit": tp,
            "status": "OPEN",
            "engine": "consensus",
            "opened_at": int(time.time() * 1000),
            "tp1_hit": 0,
            "tp2_hit": 0,
            "tp3_hit": 0,
            "tp4_hit": 0,
            "tp5_hit": 0,
            "signal_id": signal_id,
        }
        pos["id"] = state_manager.save_position(pos)
        state_manager.insert_trade({
            "position_id": pos["id"], "symbol": symbol, "direction": direction,
            "price": fill["price"], "quantity": fill["quantity"], "fee": fill["fee"],
            "slippage": fill["slippage"], "timestamp": int(time.time() * 1000), "type": "ENTRY",
        })
        self.positions.append(pos)
        logger.info(f"Opened {direction} {symbol} size=${size_usd:.2f}")
        return True

    async def _limit_order_with_timeout(
        self, symbol: str, direction: str, price: float, size_usd: float, regime: str, timeout_s: int
    ) -> Optional[Dict]:
        """Fix 2.4/2.5: Submit limit order, poll until filled or timeout, cancel on timeout."""
        order = await self.fill_provider.submit_order(symbol, direction, size_usd / price, "LIMIT", price)
        if order.status == "FILLED":
            return {"price": order.filled_price, "quantity": order.filled_quantity, "fee": 0.0, "slippage": 0.0, "filled": True}

        self._pending_limit_orders[order.id] = {
            "symbol": symbol, "direction": direction, "price": price,
            "size_usd": size_usd, "regime": regime, "created_at": time.time(),
        }

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            await asyncio.sleep(LIMIT_ORDER_POLL_INTERVAL)
            status = await self.fill_provider.get_order_status(order.id)
            if status and status.status == "FILLED":
                self._pending_limit_orders.pop(order.id, None)
                return {"price": status.filled_price, "quantity": status.filled_quantity, "fee": 0.0, "slippage": 0.0, "filled": True}

        # Timeout: cancel order
        await self.fill_provider.cancel_order(order.id)
        self._pending_limit_orders.pop(order.id, None)
        logger.info(f"Limit order cancelled (timeout {timeout_s}s) {symbol} {direction} @{price}")
        return None

    async def check_exits(self, symbol: str, features: Dict[str, float], regime: str):
        for pos in self.positions[:]:
            if pos["symbol"] != symbol:
                continue
            close = features.get("close", 0.0)
            direction = pos["direction"]
            atr_val = features.get("atr_14", 0.0)

            # Phase 4: Profit-triggered trailing stop
            unrealized_pct = self._unrealized_pct(pos, close)
            rsi_val = features.get("rsi_14", 50)
            if unrealized_pct is not None and atr_val > 0:
                pos["highest_price"] = max(pos.get("highest_price", pos["entry_price"]), close)
                # Activate trailing stop if profitable with RSI confirmation, or mandatory at +5%
                if (
                    (unrealized_pct > 0.02 and 55 < rsi_val < 75) or
                    unrealized_pct > 0.05
                ):
                    pos["trailing_stop_active"] = True

                if pos.get("trailing_stop_active"):
                    highest = pos["highest_price"]
                    trail_dist = 1.5 * atr_val
                    if direction == "BUY":
                        ts_price = highest - trail_dist
                        hard_floor = pos["entry_price"] * 0.95
                        ts_price = max(ts_price, hard_floor)
                        pos["trailing_stop_price"] = max(pos.get("trailing_stop_price", 0.0), ts_price)
                        if close <= pos["trailing_stop_price"]:
                            await self._close(pos, close, regime, "MOMENTUM_TRAILING_STOP")
                            self._remove(pos)
                            continue
                    else:
                        ts_price = highest + trail_dist
                        hard_floor = pos["entry_price"] * 1.05
                        ts_price = min(ts_price, hard_floor)
                        pos["trailing_stop_price"] = min(pos.get("trailing_stop_price", float("inf")), ts_price)
                        if close >= pos["trailing_stop_price"]:
                            await self._close(pos, close, regime, "MOMENTUM_TRAILING_STOP")
                            self._remove(pos)
                            continue

            exit_pct, reason, updated = partial_exit.check(pos, features)
            if exit_pct > 0:
                await self._partial_close(pos, close, regime, reason, exit_pct)
                # Fix 2.4: Update remaining_size_usd
                pos["remaining_size_usd"] = pos.get("remaining_size_usd", pos["size_usd"]) * (1 - exit_pct)
                idx = next((i for i, p in enumerate(self.positions) if p["id"] == pos["id"]), None)
                if idx is not None:
                    if updated.get("tp5_hit"):
                        await self._close(pos, close, regime, "TP5 trailing")
                        self.positions.pop(idx)
                    else:
                        self.positions[idx] = updated
                        self.positions[idx]["remaining_size_usd"] = pos["remaining_size_usd"]
                continue

            # Trailing stop activation based on TP hits
            if atr_val > 0:
                tp_count = sum(1 for i in range(1, 6) if updated.get(f"tp{i}_hit", 0))
                if tp_count >= 3:
                    trail_dist = 0.8 * atr_val
                elif tp_count >= 2:
                    trail_dist = 1.0 * atr_val
                elif tp_count >= 1:
                    trail_dist = 1.5 * atr_val
                else:
                    trail_dist = None
                if trail_dist is not None:
                    if direction == "BUY":
                        new_sl = close - trail_dist
                        if new_sl > pos["stop_loss"]:
                            pos["stop_loss"] = new_sl
                            logger.info(f"Trailing stop {symbol} raised to {new_sl:.4f}")
                    else:
                        new_sl = close + trail_dist
                        if new_sl < pos["stop_loss"]:
                            pos["stop_loss"] = new_sl
                            logger.info(f"Trailing stop {symbol} lowered to {new_sl:.4f}")

            hit_sl = (direction == "BUY" and close <= pos["stop_loss"]) or (direction == "SELL" and close >= pos["stop_loss"])
            hit_tp = (direction == "BUY" and close >= pos["take_profit"]) or (direction == "SELL" and close <= pos["take_profit"])
            hit_time = (int(time.time() * 1000) - pos["opened_at"]) > 20 * 5 * 60 * 1000
            ema9 = features.get("ema_9", 0)
            ema20 = features.get("ema_20", 0)
            rsi_val = features.get("rsi_14", 50)
            momentum_rev = (
                (direction == "BUY" and ema9 < ema20 and rsi_val < 45) or
                (direction == "SELL" and ema9 > ema20 and rsi_val > 55)
            ) if ema9 and ema20 else False
            if hit_sl:
                await self._close(pos, close, regime, "STOP_LOSS")
                self._remove(pos)
            elif hit_tp:
                await self._close(pos, close, regime, "TAKE_PROFIT")
                self._remove(pos)
            elif hit_time:
                await self._close(pos, close, regime, "TIME_STOP")
                self._remove(pos)
            elif momentum_rev:
                await self._close(pos, close, regime, "MOMENTUM_REVERSAL")
                self._remove(pos)

    def _unrealized_pct(self, pos: Dict, close: float) -> Optional[float]:
        if close <= 0 or pos["entry_price"] <= 0:
            return None
        if pos["direction"] == "BUY":
            return (close - pos["entry_price"]) / pos["entry_price"]
        return (pos["entry_price"] - close) / pos["entry_price"]

    async def _partial_close(self, pos: Dict, price: float, regime: str, reason: str, pct: float):
        remaining = pos.get("remaining_size_usd", pos["size_usd"])
        close_size = remaining * pct
        fill = await self.fill_provider.fill_exit({**pos, "size_usd": close_size}, price, regime, reason)
        state_manager.insert_trade({
            "position_id": pos["id"], "symbol": pos["symbol"], "direction": pos["direction"],
            "price": fill["price"], "quantity": fill["quantity"] * pct, "fee": fill["fee"] * pct,
            "slippage": fill["slippage"], "timestamp": int(time.time() * 1000), "type": "PARTIAL",
        })
        logger.info(f"Partial close {pos['symbol']} {pct*100:.0f}% reason {reason}")

    async def _close(self, pos: Dict, price: float, regime: str, reason: str):
        remaining = pos.get("remaining_size_usd", pos["size_usd"])
        pos_for_close = {**pos, "size_usd": remaining}
        fill = await self.fill_provider.fill_exit(pos_for_close, price, regime, reason)
        pnl = (fill["price"] - pos["entry_price"]) / pos["entry_price"] * remaining
        if pos["direction"] == "SELL":
            pnl *= -1
        pnl -= fill["fee"]
        pos["status"] = "CLOSED"
        pos["closed_at"] = int(time.time() * 1000)
        pos["pnl"] = pnl
        pos["exit_reason"] = reason
        pos["remaining_size_usd"] = 0.0
        state_manager.save_position(pos)
        state_manager.insert_trade({
            "position_id": pos["id"], "symbol": pos["symbol"], "direction": pos["direction"],
            "price": fill["price"], "quantity": fill["quantity"], "fee": fill["fee"],
            "slippage": fill["slippage"], "timestamp": int(time.time() * 1000), "type": "EXIT",
        })
        # Attribute P&L to engines that voted for this direction
        signal_id = pos.get("signal_id")
        if signal_id:
            sig = state_manager.get_signal_by_id(signal_id)
            if sig:
                dir_map = {
                    "sniper": sig.get("sniper_vote", "NEUTRAL"),
                    "smc": sig.get("smc_vote", "NEUTRAL"),
                    "momentum": sig.get("momentum_vote", "NEUTRAL"),
                    "mean_reversion": sig.get("mean_reversion_vote", "NEUTRAL"),
                    "price_action": sig.get("price_action_vote", "NEUTRAL"),
                    "scalping": sig.get("scalping_vote", "NEUTRAL"),
                }
                pos_dir = pos["direction"]
                for eng, vote in dir_map.items():
                    if vote == pos_dir:
                        state_manager.record_engine_pnl(eng, regime, pnl)
        logger.info(f"Closed {pos['symbol']} {reason} PnL=${pnl:.2f}")

    def _remove(self, pos: Dict):
        self.positions = [p for p in self.positions if p["id"] != pos["id"]]

    async def flatten_all(self, regime: str, reason: str = "FLATTEN"):
        for pos in self.positions[:]:
            close = pos.get("last_price", pos["entry_price"])
            await self._close(pos, close, regime, reason)
        self.positions.clear()


execution_manager = ExecutionManager()
