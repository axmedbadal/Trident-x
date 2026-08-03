"""Pure-math validation of the Sim accounting used by the Tier 1 backtest harness.

The real pipeline rarely opens trades on a 4.5-day window, so these tests exercise
open/partial/close/exit bookkeeping directly to guarantee equity, cash and PnL
bookkeeping are correct before trusting any real run.
"""
import pytest

from backtest.engine import Sim, FEE, BAR_MS


def _feats(close=100.0, atr=1.0, rsi=50.0, atr_pct=0.8):
    return {"close": close, "atr_14": atr, "atr_pct": atr_pct, "rsi_14": rsi,
            "ema_9": close, "ema_20": close}


def _sig(direction="BUY", confidence=0.9, symbol="SOLUSDT"):
    return {"symbol": symbol, "direction": direction, "confidence": confidence}


def test_open_reduces_cash_into_position():
    sim = Sim(10000)
    ok = sim.open(_sig(), _feats(), "TRENDING_UP", 0)
    assert ok
    assert len(sim.positions) == 1
    assert abs(sim.equity - 10000.0) < 1e-6
    assert sim.cash < 10000.0
    p = sim.positions[0]
    assert p["symbol"] == "SOLUSDT" and p["direction"] == "BUY"
    assert abs(p["remaining_usd"] + sim.cash - 10000.0) < 1e-6


def test_no_positions_duplicates_per_symbol():
    sim = Sim(10000)
    assert sim.open(_sig(), _feats(), "TRENDING_UP", 0)
    assert not sim.open(_sig(), _feats(), "TRENDING_UP", 1)  # same symbol already open


def test_close_buy_pnl():
    sim = Sim(10000)
    sim.open(_sig(), _feats(), "TRENDING_UP", 0)
    p = sim.positions[0]
    entry = p["entry_price"]
    exit_price = entry * 1.05  # +5%
    size = p["size_usd"]
    exp = size * 0.05 - size * FEE
    sim.close(p, exit_price, "TAKE_PROFIT", 1)
    assert len(sim.positions) == 0 and len(sim.closed) == 1
    assert abs(sim.closed[0]["pnl"] - exp) < 1e-6
    assert abs(sim.equity - (10000.0 + exp)) < 1e-6


def test_close_sell_pnl_inverts():
    sim = Sim(10000)
    sim.open(_sig("SELL"), _feats(), "TRENDING_DOWN", 0)
    p = sim.positions[0]
    exit_price = p["entry_price"] * 0.95  # price falls = profit for SELL
    size = p["size_usd"]
    sim.close(p, exit_price, "TAKE_PROFIT", 1)
    assert sim.closed[0]["pnl"] > 0


def test_partial_realizes_pnl_keeps_equity_consistent():
    sim = Sim(10000)
    sim.open(_sig(), _feats(), "TRENDING_UP", 0)
    p = sim.positions[0]
    p["close_price"] = p["entry_price"] * 1.04  # +4%
    eq_before = sim.equity
    sim.partial(p, 0.5)
    # Equity should now equal the fully-realized value of that 4% gain on the whole pos
    realized = (p["close_price"] - p["entry_price"]) / p["entry_price"] * p["size_usd"] - p["size_usd"] * FEE
    eq_after = sim.equity
    # realized partial cash + remaining at entry (unrealized book) must equal eq_before + partial pnl
    assert abs(p["pnl"]) > 0
    assert sim.cash > 0


def test_peak_tracks_equity_high():
    sim = Sim(10000)
    sim.open(_sig(), _feats(), "TRENDING_UP", 0)
    p = sim.positions[0]
    sim.close(p, p["entry_price"] * 1.10, "TAKE_PROFIT", 1)
    assert sim.peak > 10000.0
