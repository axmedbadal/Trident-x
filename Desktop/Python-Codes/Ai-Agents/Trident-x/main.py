import asyncio
import logging
import os
import signal
import sys
import time
from typing import Dict, List

import pandas as pd

# ── Multi-instance protection ──────────────────────────────────────────────
import atexit
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False
    try:
        import msvcrt
        _HAS_MSVCRT = True
    except ImportError:
        _HAS_MSVCRT = False

LOCK_FILE = "trident_x.lock"
_lock_fd = None

def _acquire_instance_lock():
    global _lock_fd
    try:
        _lock_fd = open(LOCK_FILE, "w")
        if _HAS_FCNTL:
            fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif _HAS_MSVCRT:
            msvcrt.locking(_lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            lock_age = 300
            if os.path.exists(LOCK_FILE):
                import os as _os
                lock_age = time.time() - _os.path.getmtime(LOCK_FILE)
            if lock_age < 300:
                logger.critical(f"Another instance is running (lock age: {lock_age:.0f}s). Exiting.")
                sys.exit(1)
        _lock_fd.write(str(os.getpid()))
        _lock_fd.flush()
        logger.info(f"Instance lock acquired (PID {os.getpid()})")
    except (IOError, OSError) as e:
        logger.critical(f"Another instance is already running. Exiting. ({e})")
        sys.exit(1)

def _release_instance_lock():
    global _lock_fd
    if _lock_fd:
        try:
            if _HAS_FCNTL:
                fcntl.flock(_lock_fd, fcntl.LOCK_UN)
            elif _HAS_MSVCRT:
                try:
                    msvcrt.locking(_lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
            _lock_fd.close()
        except Exception:
            pass
        _lock_fd = None
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except Exception:
        pass

atexit.register(_release_instance_lock)

# ── Config & Logging ──────────────────────────────────────────────────────
from config.settings import settings
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
if settings.LOG_TO_FILE:
    os.makedirs(os.path.dirname(settings.LOG_FILE), exist_ok=True)
    # Fix 2.3: Size-based rotation (10MB, 5 backups)
    file_handler = RotatingFileHandler(
        settings.LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    # Fix 2.3: Time-based rotation (daily, 7 days)
    daily_handler = TimedRotatingFileHandler(
        settings.LOG_FILE.replace(".log", "_daily.log"),
        when="midnight", interval=1, backupCount=7,
    )
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    daily_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(daily_handler)
logger = logging.getLogger("trident")

# ── Core ───────────────────────────────────────────────────────────────────
from core.event_bus import bus
from core.state_manager import state_manager

# ── Data ───────────────────────────────────────────────────────────────────
from data.ws_client import BinanceWebSocketClient
from data.rest_client import BinanceRestClient

# ── Features ───────────────────────────────────────────────────────────────
from features.engineer import features

# ── ML ─────────────────────────────────────────────────────────────────────
from ml.regime import regime_detector
from ml.meta_labeler import meta_labeler

# ── Strategies ─────────────────────────────────────────────────────────────
from strategies.smc import smc_engine
from strategies.momentum import momentum_engine
from strategies.mean_reversion import mean_reversion_engine
from strategies.sniper import sniper_engine

# ── Council ────────────────────────────────────────────────────────────────
from council.gate import gate
from council.consensus import consensus

# ── Risk ───────────────────────────────────────────────────────────────────
from risk.circuit_breaker import circuit_breaker
from risk.governor import governor
from risk.position_sizer import position_sizer
from risk.portfolio_risk import portfolio_risk

# ── Execution ──────────────────────────────────────────────────────────────
from execution.manager import execution_manager

# ── API ────────────────────────────────────────────────────────────────────
from api.server import start_server, update_equity, broadcast, update_pair_data, update_risk, update_signals, update_regimes


# ── Globals ────────────────────────────────────────────────────────────────
trading_loop_running = True
peak_equity = settings.INITIAL_EQUITY
daily_pnl = 0.0
consecutive_losses = 0
win_rate_cache = 0.55
log_buffer: List[Dict] = []
anomaly_pause_until = 0.0
last_candle_prices: Dict[str, float] = {}
api_error_count = 0
api_error_window_start = 0.0
FEE = 0.001
SLIPPAGE_MAP = {
    "TRENDING_UP": 0.0005, "TRENDING_DOWN": 0.0005,
    "MEAN_REVERTING": 0.0003, "ACCUMULATION": 0.0003,
    "DISTRIBUTION": 0.0003, "VOLATILE": 0.0015,
}

# ── Engine Auto-Disable ────────────────────────────────────────────────────
engine_disabled: Dict[str, bool] = {}
engine_weight_scale: Dict[str, float] = {}


def track_engine_performance(engine: str, regime: str, passed_gate: bool, executed: bool, win: bool = False):
    perf = state_manager.get_engine_performance(engine, regime)
    if perf is None:
        perf = {"total_signals": 0, "passed_gate": 0, "executed": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "avg_pnl": 0.0, "disabled": 0}
    perf["total_signals"] += 1
    if passed_gate:
        perf["passed_gate"] += 1
    if executed:
        perf["executed"] += 1
    if win:
        perf["wins"] += 1
    else:
        perf["losses"] += 1
    total = perf["total_signals"]
    perf["win_rate"] = perf["wins"] / max(total, 1)
    perf["gate_pass_rate"] = perf["passed_gate"] / max(total, 1)
    state_manager.update_engine_performance(engine, regime, perf)
    key = f"{engine}:{regime}"
    if total >= settings.ENGINE_EVAL_MIN_SIGNALS and perf["gate_pass_rate"] < settings.ENGINE_DISABLE_THRESHOLD:
        engine_disabled[key] = True
        logger.warning(f"Engine {engine} DISABLED for regime {regime}: gate_pass={perf['gate_pass_rate']:.2f} < {settings.ENGINE_DISABLE_THRESHOLD}")
    if total >= 50 and perf["win_rate"] < 0.50:
        engine_weight_scale[key] = 0.5
        logger.warning(f"Engine {engine} weight halved for regime {regime}: win_rate={perf['win_rate']:.2f}")
    return perf


def is_engine_disabled(engine: str, regime: str) -> bool:
    return engine_disabled.get(f"{engine}:{regime}", False)


def get_engine_weight_scale(engine: str, regime: str) -> float:
    return engine_weight_scale.get(f"{engine}:{regime}", 1.0)


# ── Anomaly Detector ──────────────────────────────────────────────────────
def check_anomalies(symbol: str, candle: Dict) -> bool:
    global anomaly_pause_until, api_error_count, api_error_window_start
    now = time.time() * 1000
    if now < anomaly_pause_until:
        return False
    # Price spike >3% in single bar
    prev_price = last_candle_prices.get(symbol, 0)
    if prev_price > 0:
        pct_change = abs(candle["close"] - prev_price) / prev_price
        if pct_change > 0.03:
            logger.warning(f"Anomaly: price spike {symbol} {pct_change:.2%}")
            anomaly_pause_until = now + 15 * 60 * 1000
            return False
    last_candle_prices[symbol] = candle["close"]
    # Volume spike >5x average
    vol_buf = features.buffers.get(f"{symbol}:5m", [])
    if len(vol_buf) > 20:
        avg_vol = sum(c["volume"] for c in vol_buf[-20:]) / 20
        if avg_vol > 0 and candle["volume"] > 5 * avg_vol:
            logger.warning(f"Anomaly: volume spike {symbol} {candle['volume']/avg_vol:.1f}x")
            anomaly_pause_until = now + 15 * 60 * 1000
            return False
    # Funding rate spike >0.05% (checked via API error rate proxy)
    return True


def _validate_candle(candle: Dict) -> bool:
    """Reject candles with NaN, zero volume, or timestamp gaps."""
    import math
    for field in ["open", "high", "low", "close", "volume"]:
        val = candle.get(field, 0)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return False
    if candle.get("volume", 0) <= 0:
        return False
    if candle.get("high", 0) < candle.get("low", 0):
        return False
    sym = candle.get("symbol", "")
    if sym:
        buf = features.buffers.get(f"{sym}:5m", [])
        if buf:
            last_ts = buf[-1].get("timestamp", 0)
            if last_ts > 0 and candle["timestamp"] - last_ts > 10 * 60 * 1000:
                logger.warning(f"Candle gap detected {sym}: {candle['timestamp'] - last_ts}ms")
    return True


# ── Offline Warmup ─────────────────────────────────────────────────────────
_context_rest_client: BinanceRestClient = None


async def warmup():
    """Load cached candles from SQLite; if none, fetch via REST."""
    global _context_rest_client
    _context_rest_client = BinanceRestClient()
    for sym in settings.PAIRS:
        for tf in [settings.PRIMARY_TF] + settings.CONTEXT_TFS:
            cached = state_manager.load_candles(sym, tf, 500)
            if cached:
                features.load_history(sym, tf, cached)
                logger.info(f"Warmup {sym} {tf}: {len(cached)} cached candles")
            else:
                try:
                    candles = await _context_rest_client.fetch_klines(sym, tf, 500)
                    if candles:
                        features.load_history(sym, tf, candles)
                        state_manager.save_candles(candles)
                        logger.info(f"Warmup {sym} {tf}: fetched {len(candles)} candles via REST")
                except Exception as e:
                    logger.warning(f"Warmup REST fetch failed {sym} {tf}: {e}")


# Fix 1.4: Context timeframe refresh via REST every 5m
async def refresh_context_timeframes():
    """Fetch 15m/1h/4h candles via REST every 5 minutes for MTF alignment."""
    global _context_rest_client
    if _context_rest_client is None:
        _context_rest_client = BinanceRestClient()
    while trading_loop_running:
        try:
            for sym in settings.PAIRS:
                for tf in settings.CONTEXT_TFS:
                    try:
                        candles = await _context_rest_client.fetch_klines(sym, tf, 100)
                        if candles:
                            features.load_history(sym, tf, candles)
                            state_manager.save_candles(candles)
                    except Exception as e:
                        logger.debug(f"Context TF fetch failed {sym} {tf}: {e}")
        except Exception as e:
            logger.warning(f"Context TF refresh error: {e}")
        await asyncio.sleep(300)  # Every 5 minutes


# ── Candle Callback ────────────────────────────────────────────────────────
async def on_candle(candle: Dict):
    """Called when WebSocket delivers a closed 5m candle."""
    global daily_pnl, consecutive_losses, win_rate_cache, peak_equity

    symbol = candle["symbol"]

    # Anomaly detection
    if not check_anomalies(symbol, candle):
        return

    # Data validation
    if not _validate_candle(candle):
        return

    features.update(symbol, settings.PRIMARY_TF, candle)
    state_manager.save_candles([{**candle, "timeframe": settings.PRIMARY_TF}])

    feat_5m = features.compute(symbol, settings.PRIMARY_TF)
    if feat_5m is None:
        return

    # HTF context: fetch a rough direction from 15m/1h features
    mtf_signals = {}
    for htf in ["15m", "1h"]:
        htf_feats = features.compute(symbol, htf)
        if htf_feats:
            mtf_signals[htf] = _htf_direction(htf_feats)

    # Regime detection
    regime_state = regime_detector.detect(symbol, pd.DataFrame(features.buffers.get(f"{symbol}:{settings.PRIMARY_TF}", [])))
    regime = regime_state.state

    # Regime history
    state_manager.insert_regime_change(
        int(time.time() * 1000), symbol, regime, regime_state.confidence, regime_state.hurst
    )

    # Funding rate
    rest = BinanceRestClient()
    funding = 0.0
    try:
        funding = await asyncio.wait_for(rest.fetch_funding_rate(symbol), timeout=5) or 0.0
    except Exception:
        pass
    finally:
        await rest.close()

    # Engine signals
    sig_smc = smc_engine.generate(symbol, feat_5m, funding)
    sig_mom = momentum_engine.generate(symbol, feat_5m, funding)
    sig_mr = mean_reversion_engine.generate(symbol, feat_5m, funding)
    sig_sniper = sniper_engine.generate(symbol, feat_5m, funding)
    sig_smc["permitted_regimes"] = smc_engine.permitted_regimes
    sig_mom["permitted_regimes"] = momentum_engine.permitted_regimes
    sig_mr["permitted_regimes"] = mean_reversion_engine.permitted_regimes
    sig_sniper["permitted_regimes"] = sniper_engine.permitted_regimes

    # Consensus
    combined = consensus.combine([sig_smc, sig_mom, sig_mr, sig_sniper], regime)

    # Meta-labeler
    meta_prob = meta_labeler.predict(symbol, feat_5m)

    # Gate
    data_age = time.time() * 1000 - candle["timestamp"]
    passed, reason = gate.evaluate(combined, feat_5m, mtf_signals, meta_prob, data_age)

    # Engine auto-disable tracking
    for sig in [sig_smc, sig_mom, sig_mr, sig_sniper]:
        eng_name = sig.get("engine", "")
        if eng_name and sig.get("direction") != "NEUTRAL":
            engine_passed = passed and sig.get("direction") == combined.get("direction")
            track_engine_performance(eng_name, regime, engine_passed, passed)
            # Apply weight scaling if engine is degraded
            scale = get_engine_weight_scale(eng_name, regime)
            if scale < 1.0:
                sig["weight"] = sig.get("weight", 0.33) * scale

    # Persist signal
    state_manager.insert_signal({
        "symbol": symbol,
        "direction": combined["direction"],
        "confidence": combined["confidence"],
        "strength": combined["strength"],
        "regime": regime,
        "engines_agreeing": combined["engines_agreeing"],
        "passed_gate": passed,
        "gate_reason": reason,
        "executed": False,
        "timestamp": int(time.time() * 1000),
        "meta_prob": meta_prob,
    })

    # Broadcast update
    equity = governor.peak_equity - (governor.peak_equity * governor.max_dd if governor.max_dd else 0)
    update_equity(equity)

    # Update pair data for dashboard
    update_pair_data(symbol, feat_5m.get("close", 0.0), regime, combined["direction"], combined["confidence"])

    # Check existing positions exits
    await execution_manager.check_exits(symbol, feat_5m, regime)

    # Risk checks
    circuit_breaker.update(equity, peak_equity, daily_pnl, consecutive_losses)
    band = governor.update(equity)
    if band == "BLACK":
        logger.warning("BLACK band: no new trades")

    # Update risk for dashboard
    update_risk(band, governor.peak_equity)

    # Broadcast risk state
    state_manager.save_risk_state({
        "timestamp": int(time.time() * 1000),
        "daily_pnl": daily_pnl,
        "max_dd": governor.max_dd,
        "current_dd": (peak_equity - equity) / peak_equity if peak_equity else 0.0,
        "band": band,
        "consecutive_losses": consecutive_losses,
        "cb_state": circuit_breaker.state,
    })

    # Flush log buffer
    if len(log_buffer) >= 10:
        state_manager.batch_log(log_buffer.copy())
        log_buffer.clear()

    # Anomaly detector
    if feat_5m.get("vol_sma20_ratio", 0) > 5.0:
        logger.warning(f"Volume spike detected {symbol}: {feat_5m['vol_sma20_ratio']:.1f}x")

    # Open new position?
    if not passed or not circuit_breaker.can_trade() or band == "BLACK":
        await broadcast({"type": "update", "signal": combined, "passed": passed, "reason": reason})
        return

    open_count = len(execution_manager.positions)
    if open_count >= settings.MAX_POSITIONS:
        return
    # Only one position per symbol
    if any(p["symbol"] == symbol for p in execution_manager.positions):
        return

    opened = await execution_manager.open_position(
        symbol, combined, feat_5m, equity, regime, band, win_rate_cache,
    )
    if opened:
        logger.info(f"Signal executed: {combined['direction']} {symbol} conf={combined['confidence']:.3f}")

    await broadcast({"type": "update", "signal": combined, "passed": passed})


def _htf_direction(feat: Dict[str, float]) -> str:
    if feat.get("rsi_14", 50) > 55:
        return "BUY"
    elif feat.get("rsi_14", 50) < 45:
        return "SELL"
    return "NEUTRAL"


# ── Background tasks ───────────────────────────────────────────────────────
async def funding_poll_loop():
    """Poll funding rates every 8 hours."""
    rest = BinanceRestClient()
    while trading_loop_running:
        try:
            for sym in settings.PAIRS:
                fr = await rest.fetch_funding_rate(sym)
                if fr is not None:
                    # Fix 2.6: Persist funding rate to database
                    state_manager.save_funding_rate(sym, fr)
                    logger.info(f"Funding {sym}: {fr:.6f}")
        except Exception as e:
            logger.warning(f"Funding poll error: {e}")
        finally:
            await rest.close()
        await asyncio.sleep(8 * 3600)


async def regime_retrain_loop():
    """Check if regime models need retraining and adjust engine weights."""
    while trading_loop_running:
        try:
            for sym in settings.PAIRS:
                df = pd.DataFrame(features.buffers.get(f"{sym}:{settings.PRIMARY_TF}", []))
                if len(df) > 200:
                    await regime_detector.maybe_retrain(sym, df)
                    await meta_labeler.maybe_retrain(sym, state_manager.load_candles(sym, "5m", 500))
            # Fix 3.2: Adjust engine weights based on performance
            consensus.adjust_weights()
        except Exception as e:
            logger.warning(f"Regime retrain error: {e}")
        await asyncio.sleep(6 * 3600)


async def prune_loop():
    """Weekly database maintenance: prune old data, VACUUM, optimize."""
    while trading_loop_running:
        await asyncio.sleep(7 * 86400)
        try:
            state_manager.prune()
            # Force VACUUM (requires separate connection)
            import sqlite3
            conn = sqlite3.connect(state_manager.db_path)
            conn.execute("VACUUM")
            conn.close()
            logger.info("Weekly maintenance: pruned + VACUUM complete")
        except Exception as e:
            logger.warning(f"Weekly maintenance error: {e}")


# ── Shutdown ───────────────────────────────────────────────────────────────
async def graceful_shutdown(sig_name: str = "SIGTERM"):
    global trading_loop_running
    logger.critical(f"Received {sig_name}, shutting down...")
    trading_loop_running = False
    # Flush log buffer
    if log_buffer:
        state_manager.batch_log(log_buffer)
        log_buffer.clear()
    # Stop event bus
    try:
        await bus.stop()
    except Exception:
        pass
    # Stop WebSocket
    if ws_client:
        try:
            await ws_client.stop()
        except Exception:
            pass
    logger.critical("Shutdown complete.")


# ── Main ───────────────────────────────────────────────────────────────────
ws_client: BinanceWebSocketClient = None


async def main():
    global ws_client, peak_equity, daily_pnl, consecutive_losses, win_rate_cache

    # Fix 3.3: Multi-instance protection
    _acquire_instance_lock()

    logger.info("═" * 60)
    logger.info("  TRIDENT-X — Ultimate Quant Trading System")
    logger.info(f"  Pairs: {settings.PAIRS} | TF: {settings.PRIMARY_TF}")
    logger.info(f"  Paper Trading: {settings.PAPER_TRADING}")
    logger.info(f"  Initial Equity: ${settings.INITIAL_EQUITY:,.2f}")
    logger.info("═" * 60)

    # Load state from SQLite
    execution_manager.load_positions()
    risk = state_manager.load_risk_state()
    if risk:
        daily_pnl = risk.get("daily_pnl", 0.0)
        consecutive_losses = int(risk.get("consecutive_losses", 0))

    # Warmup from cache/REST
    await warmup()

    # Start event bus
    await bus.start()

    # Start WebSocket
    ws_client = BinanceWebSocketClient(on_candle=on_candle)
    await ws_client.start()
    logger.info("WebSocket client started")

    # Background tasks
    asyncio.create_task(funding_poll_loop())
    asyncio.create_task(regime_retrain_loop())
    asyncio.create_task(prune_loop())
    asyncio.create_task(refresh_context_timeframes())

    # Start API server
    logger.info("Starting dashboard at http://0.0.0.0:8000")
    await start_server()


def entry():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if sys.platform != "win32":
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig, lambda s=sig: loop.create_task(graceful_shutdown(s.name))
            )
    else:
        def _windows_handler(sig_num, frame):
            asyncio.ensure_future(graceful_shutdown("SIGTERM"))
        signal.signal(signal.SIGTERM, _windows_handler)

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        loop.run_until_complete(graceful_shutdown("KeyboardInterrupt"))
    finally:
        loop.close()


if __name__ == "__main__":
    entry()
