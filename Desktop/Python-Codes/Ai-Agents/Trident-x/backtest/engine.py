"""
Tier 1: Backtesting harness. Replays historical candles from trident_x.sqlite
through the same pipeline used live (FeatureEngineer -> strategy engines ->
consensus -> gate -> sizing -> exit) WITHOUT network or live position managers.

Reuses the pure components (strategy engines, consensus, gate, meta_labeler,
regime_detector, partial_exit) and populates the global feature buffers so the
multi-timeframe logic (e.g. Sniper) behaves exactly as it does live.
"""
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from config.settings import settings
from core.state_manager import state_manager
from features.engineer import features
from strategies.momentum import momentum_engine
from strategies.smc import smc_engine
from strategies.mean_reversion import mean_reversion_engine
from strategies.sniper import sniper_engine
from council.gate import gate as gate_engine
from council.consensus import consensus
from ml.regime import regime_detector
from ml.meta_labeler import meta_labeler
from execution.partial_exit import partial_exit

logger = logging.getLogger(__name__)

FEE = 0.001
WARMUP = 60
BAR_MS = 5 * 60 * 1000
TIME_STOP_BARS = 20
ALL_REGIMES = {"TRENDING_UP", "TRENDING_DOWN", "MEAN_REVERTING", "ACCUMULATION", "DISTRIBUTION"}
SLIP = {"TRENDING_UP": 0.0005, "TRENDING_DOWN": 0.0005, "MEAN_REVERTING": 0.0003,
        "ACCUMULATION": 0.0003, "DISTRIBUTION": 0.0003}


class Sim:
    def __init__(self, initial: float):
        self.initial = initial
        self.cash = initial
        self.peak = initial
        self.positions: List[Dict] = []
        self.closed: List[Dict] = []
        # engine -> regime -> {wins, losses}; fed to consensus.apply_backtest_metrics
        self.metrics: Dict[str, Dict[str, Dict[str, int]]] = {}

    @property
    def equity(self) -> float:
        return self.cash + sum(p["remaining_usd"] for p in self.positions)

    def fraction(self, signal: Dict, feats: Dict, regime: str) -> float:
        p = max(0.0, min(1.0, signal.get("confidence", 0.0)))
        atr_pct = feats.get("atr_pct", 0.0) / 100.0
        vol_mult = 1.0 if atr_pct < 0.05 else (0.75 if atr_pct < 0.10 else 0.5)
        regime_mult = {"TRENDING_UP": 1.2, "TRENDING_DOWN": 1.2, "MEAN_REVERTING": 0.8,
                       "ACCUMULATION": 0.6, "DISTRIBUTION": 0.6}.get(regime, 1.0)
        sym = signal.get("symbol", settings.PAIRS[0])
        params = settings.ASSET_PARAMS.get(sym, settings.ASSET_PARAMS["SOLUSDT"])
        rr = params["tp_mult"] / max(params["sl_mult"], 1e-9)
        kelly = max((p * rr - (1 - p)) / max(rr, 1e-9), 0.0) * settings.KELLY_FRACTION
        frac = kelly * vol_mult * regime_mult
        return max(0.0, min(frac, settings.MAX_POSITION_EQUITY_PCT))

    def open(self, signal: Dict, feats: Dict, regime: str, t: int):
        if len(self.positions) >= settings.MAX_POSITIONS:
            return False
        sym = signal.get("symbol")
        if any(p["symbol"] == sym for p in self.positions):
            return False
        size = self.equity * self.fraction(signal, feats, regime)
        if size <= 10:
            return False
        close = feats.get("close", 0.0)
        atr = feats.get("atr_14", 0.0)
        if close <= 0 or atr <= 0:
            return False
        d = signal["direction"]
        params = settings.ASSET_PARAMS.get(sym, settings.ASSET_PARAMS["SOLUSDT"])
        entry = close * (1 + SLIP.get(regime, 0.0005)) if d == "BUY" else close * (1 - SLIP.get(regime, 0.0005))
        sl = entry - params["sl_mult"] * atr if d == "BUY" else entry + params["sl_mult"] * atr
        tp = entry + params["tp_mult"] * atr if d == "BUY" else entry - params["tp_mult"] * atr
        p = {
            "symbol": sym, "direction": d, "entry_price": entry, "size_usd": size,
            "remaining_usd": size, "stop_loss": sl, "take_profit": tp,
            "opened_at": t, "closed_at": None, "pnl": 0.0, "exit_reason": None,
            "tp1_hit": 0, "tp2_hit": 0, "tp3_hit": 0, "tp4_hit": 0, "tp5_hit": 0,
            "regime": regime, "engines": list(signal.get("engines", [])),
        }
        self.positions.append(p)
        self.cash -= size
        return True

    def close(self, p: Dict, exit_price: float, reason: str, t: int):
        proceeds = p["remaining_usd"]
        pnl = (exit_price - p["entry_price"]) / p["entry_price"] * proceeds
        if p["direction"] == "SELL":
            pnl *= -1
        pnl -= proceeds * FEE
        p["pnl"] = pnl
        p["closed_at"] = t
        p["exit_reason"] = reason
        p["remaining_usd"] = 0.0
        if p in self.positions:
            self.positions.remove(p)
        self.closed.append(p)
        self.cash += proceeds + pnl
        self.peak = max(self.peak, self.equity)
        self._attribute(p)

    def _attribute(self, p: Dict):
        """Attribute closed P&L to each engine that voted for the entry direction."""
        regime = p.get("regime", "MEAN_REVERTING")
        wins = 1 if p["pnl"] > 0 else 0
        losses = 1 if p["pnl"] <= 0 else 0
        for eng in p.get("engines", []):
            bucket = self.metrics.setdefault(eng, {}).setdefault(regime, {"wins": 0, "losses": 0})
            bucket["wins"] += wins
            bucket["losses"] += losses

    def partial(self, p: Dict, pct: float):
        close_size = p["remaining_usd"] * pct
        p["remaining_usd"] = p["remaining_usd"] * (1 - pct)
        p["size_usd"] = p["size_usd"] - close_size
        gross = (p["close_price"] - p["entry_price"]) / p["entry_price"] * close_size
        if p["direction"] == "SELL":
            gross *= -1
        pnl_partial = gross - close_size * FEE
        p["pnl"] += pnl_partial
        self.cash += close_size + pnl_partial


def _seed_buffers(sym: str, c5: List[Dict], ctx: Dict[str, List[Dict]], t: int):
    """Point the global feature buffers at history up to time t for this symbol."""
    hist_5m = [c for c in c5 if c["timestamp"] <= t]
    features.buffers[f"{sym}:{settings.PRIMARY_TF}"] = hist_5m[-500:]
    for tf, candles in ctx.items():
        snap = [c for c in candles if c["timestamp"] <= t]
        features.buffers[f"{sym}:{tf}"] = snap[-500:]


def _check_exits(sim: Sim, sym: str, feats: Dict, t: int):
    close = feats.get("close", 0.0)
    for p in list(sim.positions):
        if p["symbol"] != sym:
            continue
        d = p["direction"]
        atr = feats.get("atr_14", 0.0)
        rsi = feats.get("rsi_14", 50)

        # Trailing stop (mirror manager logic)
        if p.get("entry_price", 0) > 0 and close > 0:
            upnl = (close - p["entry_price"]) / p["entry_price"] if d == "BUY" else \
                   (p["entry_price"] - close) / p["entry_price"]
            if upnl > 0.02 and 55 < rsi < 75:
                p["highest"] = max(p.get("highest", p["entry_price"]), close)
                p["trail_on"] = 1
            if p.get("trail_on") and atr > 0:
                trail = 1.5 * atr
                if d == "BUY":
                    ts = max(p.get("highest", p["entry_price"]) - trail, p["entry_price"] * 0.95)
                    if close <= ts:
                        sim.close(p, close, "TRAILING_STOP", t)
                        continue
                else:
                    ts = min(p.get("highest", p["entry_price"]) + trail, p["entry_price"] * 1.05)
                    if close >= ts:
                        sim.close(p, close, "TRAILING_STOP", t)
                        continue

        # Partial scaling via the real partial_exit component
        updated = dict(p)
        exit_pct, _reason, updated = partial_exit.check(updated, feats)
        if exit_pct > 0:
            p["close_price"] = close
            sim.partial(p, exit_pct)
            for i in range(1, 6):
                if updated.get(f"tp{i}_hit"):
                    p[f"tp{i}_hit"] = 1
            if updated.get("tp5_hit"):
                sim.close(p, close, "TP5", t)
                continue

        hit_sl = (d == "BUY" and close <= p["stop_loss"]) or (d == "SELL" and close >= p["stop_loss"])
        hit_tp = (d == "BUY" and close >= p["take_profit"]) or (d == "SELL" and close <= p["take_profit"])
        hit_time = (t - p["opened_at"]) > TIME_STOP_BARS * BAR_MS
        ema9, ema20 = feats.get("ema_9", 0), feats.get("ema_20", 0)
        mom_rev = ((d == "BUY" and ema9 < ema20 and rsi < 45) or
                   (d == "SELL" and ema9 > ema20 and rsi > 55)) if ema9 and ema20 else False

        if hit_sl:
            sim.close(p, close, "STOP_LOSS", t)
        elif hit_tp:
            sim.close(p, close, "TAKE_PROFIT", t)
        elif hit_time:
            sim.close(p, close, "TIME_STOP", t)
        elif mom_rev:
            sim.close(p, close, "MOMENTUM_REVERSAL", t)


def _synthetic_signal(sym: str, feats: Dict, regime: str) -> Dict:
    """Force-mode fallback entry so exits/PnL get exercised on real candles."""
    close = feats.get("close", 0.0)
    ema = feats.get("ema_20", 0.0) or feats.get("ema_9", 0.0) or close
    d = "BUY" if close >= ema else "SELL"
    engines = ["smc", "momentum"] if d == "BUY" else ["momentum", "mean_reversion"]
    return {
        "symbol": sym, "direction": d, "confidence": 0.9, "strength": "STRONG",
        "engines_agreeing": 2, "regime": regime, "engines": engines,
    }


def _install_compute_cache() -> Dict:
    """Wrap features.compute so repeated (symbol, tf, buffer-head) calls are served
    from cache. The backtest drives the buffer; the head timestamp only changes when
    a new candle for that tf is added, so at most one real compute happens per bar."""
    cache: Dict[Any, Any] = {}
    orig = features.compute

    def cached(symbol: str, tf: str):
        buf = features.buffers.get(f"{symbol}:{tf}")
        head = buf[-1]["timestamp"] if buf else None
        key = (symbol, tf, head)
        if key not in cache:
            cache[key] = orig(symbol, tf)
        return cache[key]

    features.compute = cached  # type: ignore[method-assign]
    return cache


def load_data(symbols: Optional[List[str]] = None, days: Optional[int] = None) -> Dict:
    """Load per-symbol candles. days limits the 5m window to the most recent N*288 bars."""
    symbols = symbols or settings.PAIRS
    data = {}
    for sym in symbols:
        c5 = sorted(state_manager.load_candles(sym, settings.PRIMARY_TF, 100000), key=lambda x: x["timestamp"])
        if days:
            bars = days * 288
            c5 = c5[-bars:]
        if c5:
            first_ts = c5[0]["timestamp"]
            data[sym] = {
                "5m": c5,
                "ctx": {tf: sorted([c for c in state_manager.load_candles(sym, tf, 100000)
                                    if c["timestamp"] >= first_ts], key=lambda x: x["timestamp"])
                        for tf in settings.CONTEXT_TFS},
            }
        else:
            data[sym] = {"5m": [], "ctx": {tf: [] for tf in settings.CONTEXT_TFS}}
    return data


def run_backtest(symbols: Optional[List[str]] = None, initial: float = 10000.0, force: bool = False,
                 days: Optional[int] = None):
    """Portfolio-level backtest. Returns (Sim, stats, rejections, eng_stats, consensus_reasons).

    force=True (smoke mode) synthesizes an entry whenever the gate rejects but the
    feature state is valid, so open->partial->exit->PnL bookkeeping gets exercised
    even when real engine setups are rare. days limits the 5m window (e.g. 5 = last 5 days).
    """
    symbols = symbols or settings.PAIRS
    data = load_data(symbols, days=days)
    _install_compute_cache()
    sim = Sim(initial)
    rejections: Dict[str, int] = {}
    eng_stats: Dict[str, Dict[str, int]] = {}
    consensus_reasons: Dict[str, int] = {}

    for sym in symbols:
        c5 = data[sym]["5m"]
        ctx = data[sym]["ctx"]
        if len(c5) < WARMUP + 20:
            continue
        for pos in range(WARMUP, len(c5)):
            candle = c5[pos]
            t = candle["timestamp"]
            _seed_buffers(sym, c5, ctx, t)

            feats5 = features.compute(sym, settings.PRIMARY_TF)
            if feats5 is None:
                continue
            df5 = pd.DataFrame(c5[:pos + 1])
            regime_state = regime_detector.detect(sym, df5)
            regime = regime_state.state if regime_state else "MEAN_REVERTING"

            sigs = [
                smc_engine.generate(sym, feats5, 0.0),
                momentum_engine.generate(sym, feats5, 0.0),
                mean_reversion_engine.generate(sym, feats5, 0.0),
                sniper_engine.generate(sym, feats5, 0.0),
            ]
            for s in sigs:
                s["permitted_regimes"] = ALL_REGIMES
                eng = s.get("engine", "unknown")
                eng_stats.setdefault(eng, {"eval": 0, "fired": 0})
                eng_stats[eng]["eval"] += 1
                if s.get("direction") != "NEUTRAL":
                    eng_stats[eng]["fired"] += 1
            combined = consensus.combine(sigs, regime)
            combined["symbol"] = sym
            if combined.get("direction") == "NEUTRAL":
                consensus_reasons[combined.get("rationale", "unknown")] = \
                    consensus_reasons.get(combined.get("rationale", "unknown"), 0) + 1
            elif combined.get("direction") != "NEUTRAL":
                combined["engines"] = [s.get("engine") for s in sigs if s.get("direction") == combined["direction"]]
                combined["regime"] = regime

            meta_prob = meta_labeler.predict(sym, feats5)
            mtf_signals = {}
            for tf in ["15m", "1h"]:
                f = features.compute(sym, tf)
                if f:
                    mtf_signals[tf] = "BUY" if f.get("rsi_14", 50) > 55 else ("SELL" if f.get("rsi_14", 50) < 45 else "NEUTRAL")

            passed, _reason = gate_engine.evaluate(combined, feats5, mtf_signals, meta_prob, data_age_ms=0)

            # Exits first, then possibly a new entry.
            _check_exits(sim, sym, feats5, t)
            if not passed:
                reason = (_reason or "unknown").split(":")[0]
                if force and len(sim.positions) < settings.MAX_POSITIONS and not any(x["symbol"] == sym for x in sim.positions):
                    synth = _synthetic_signal(sym, feats5, regime)
                    sim.open(synth, feats5, regime, t)
                else:
                    rejections[reason] = rejections.get(reason, 0) + 1
            elif passed and len(sim.positions) < settings.MAX_POSITIONS:
                sim.open(combined, feats5, regime, t)

    return sim, _stats(sim), rejections, eng_stats, consensus_reasons


def _stats(sim: Sim) -> Dict:
    closed = sim.closed
    wins = sum(1 for c in closed if c["pnl"] > 0)
    losses = sum(1 for c in closed if c["pnl"] <= 0)
    by_reason = {}
    for c in closed:
        by_reason[c["exit_reason"]] = by_reason.get(c["exit_reason"], 0) + 1
    return {
        "trades": len(closed),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / len(closed)) if closed else 0.0,
        "total_pnl": sum(c["pnl"] for c in closed),
        "final_equity": sim.equity,
        "peak": sim.peak,
        "open_positions": len(sim.positions),
        "exit_reasons": by_reason,
        "engine_metrics": sim.metrics,
    }


def run_smoke(symbols: Optional[List[str]] = None, initial: float = 10000.0, days: Optional[int] = None):
    """Smoke run: force entries so the full open->partial->exit->PnL path runs."""
    return run_backtest(symbols=symbols, initial=initial, force=True, days=days)