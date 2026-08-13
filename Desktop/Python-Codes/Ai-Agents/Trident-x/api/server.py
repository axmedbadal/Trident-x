import asyncio
import json
import os
import time
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader

from config.settings import settings
from core.event_bus import bus
from core.monitoring import system_monitor
from core.state_manager import state_manager
from execution.manager import execution_manager
from risk.circuit_breaker import circuit_breaker

app = FastAPI(title="TRIDENT-X Dashboard")
_dashboard_html = ""
if os.path.exists("api/dashboard.html"):
    with open("api/dashboard.html", "r", encoding="utf-8") as f:
        _dashboard_html = f.read()

# Fix 3.4: API key authentication
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def _verify_api_key(x_api_key: str = Depends(_api_key_header)):
    if not settings.DASHBOARD_AUTH_ENABLED:
        return
    if not settings.DASHBOARD_API_KEY:
        return
    if x_api_key != settings.DASHBOARD_API_KEY:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Invalid API key")

_clients: list = []


@app.get("/")
def index():
    return HTMLResponse(content=_dashboard_html, status_code=200)


@app.get("/api/health")
def api_health():
    return {
        "status": "ok",
        "paper": settings.PAPER_TRADING,
        "risk_state": circuit_breaker.state,
        "paused": _paused,
        "timestamp": int(time.time() * 1000),
    }


@app.get("/api/monitor/stats")
def api_monitor_stats():
    return system_monitor.stats()


@app.get("/api/monitor/logs")
def api_monitor_logs(level: str = None, limit: int = 50):
    return system_monitor.recent_logs(level, limit)


@app.get("/api/monitor/errors")
def api_monitor_errors(limit: int = 20):
    return system_monitor.recent_errors(limit)


@app.get("/api/state")
def api_state(auth=Depends(_verify_api_key)):
    return _state_payload()


@app.post("/api/override/pause")
def api_pause(auth=Depends(_verify_api_key)):
    global _paused
    _paused = True
    return {"status": "paused"}


@app.post("/api/override/resume")
def api_resume(auth=Depends(_verify_api_key)):
    global _paused
    _paused = False
    return {"status": "resumed"}


@app.post("/api/override/flatten")
def api_flatten(auth=Depends(_verify_api_key)):
    circuit_breaker.state = "KILL"
    asyncio.create_task(execution_manager.flatten_all("KILL", "api_flatten"))
    return {"status": "flattening"}


@app.get("/api/signals")
def api_signals(symbol: str = None, limit: int = 20, auth=Depends(_verify_api_key)):
    return state_manager.get_recent_signals(symbol, limit)


@app.get("/api/positions")
def api_positions(auth=Depends(_verify_api_key)):
    return state_manager.get_open_positions()


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    _clients.append(websocket)
    try:
        await websocket.send_json(_state_payload())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _clients:
            _clients.remove(websocket)
    except Exception:
        if websocket in _clients:
            _clients.remove(websocket)


async def broadcast(payload: Dict[str, Any]):
    dead = []
    for ws in _clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _clients:
            _clients.remove(ws)


def _state_payload():
    positions = state_manager.get_open_positions()
    for p in positions:
        try:
            cur_price = _pair_prices.get(p["symbol"], p["entry_price"])
            if p["direction"] == "BUY":
                p["pnl"] = (cur_price - p["entry_price"]) / p["entry_price"] * p["size_usd"]
            else:
                p["pnl"] = (p["entry_price"] - cur_price) / p["entry_price"] * p["size_usd"]
        except Exception:
            p["pnl"] = 0.0
    pairs = []
    for sym in settings.PAIRS:
        price = _pair_prices.get(sym, 0)
        pairs.append({
            "symbol": sym,
            "price": price,
            "change24h": _pair_24h.get(sym, "0.00%"),
            "regime": _pair_regimes.get(sym, "-"),
            "signal": _pair_signals.get(sym, "NEUTRAL"),
            "confidence": _pair_confidence.get(sym, 0.0),
        })
    engine_perf = []
    for eng in ["smc", "momentum", "mean_reversion", "sniper", "price_action", "scalping"]:
        for regime in ["TRENDING_UP", "TRENDING_DOWN", "MEAN_REVERTING", "ACCUMULATION", "DISTRIBUTION"]:
            perf = state_manager.get_engine_performance(eng, regime)
            if perf and perf.get("total_signals", 0) > 0:
                engine_perf.append({
                    "engine": eng,
                    "regime": regime,
                    "total_signals": perf["total_signals"],
                    "gate_pass_rate": perf.get("passed_gate", 0) / max(perf["total_signals"], 1),
                    "win_rate": perf.get("win_rate", 0),
                    "disabled": perf.get("disabled", 0),
                })
    # Fix 3.5: UTC timestamp
    import datetime
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    return {
        "time": int(utc_now.timestamp() * 1000),
        "equity": _last_equity,
        "peak_equity": _last_peak_equity,
        "positions": positions,
        "risk": circuit_breaker.state,
        "daily_pnl": circuit_breaker.daily_pnl,
        "max_dd": circuit_breaker.max_dd,
        "current_dd": (1 - _last_equity / _last_peak_equity) if _last_peak_equity else 0,
        "band": _last_band,
        "consecutive_losses": circuit_breaker.consecutive_losses,
        "paused": _paused,
        "paper": settings.PAPER_TRADING,
        "pairs": pairs,
        "signals": _last_signals,
        "engine_perf": engine_perf,
        "regimes": _last_regimes,
    }


_last_equity = settings.INITIAL_EQUITY
_last_peak_equity = settings.INITIAL_EQUITY
_last_band = "GREEN"
_last_signals = []
_last_regimes = []
_pair_prices = {}
_pair_24h = {}
_pair_regimes = {}
_pair_signals = {}
_pair_confidence = {}
_paused = False


async def _broadcast_task():
    while True:
        try:
            await broadcast(_state_payload())
        except Exception:
            pass
        await asyncio.sleep(5)


async def start_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    asyncio.create_task(_broadcast_task())
    await server.serve()


def update_equity(equity: float):
    global _last_equity, _last_peak_equity
    _last_equity = equity
    if equity > _last_peak_equity:
        _last_peak_equity = equity


def update_pair_data(symbol: str, price: float, regime: str = "", signal: str = "NEUTRAL", confidence: float = 0.0):
    _pair_prices[symbol] = price
    _pair_regimes[symbol] = regime
    _pair_signals[symbol] = signal
    _pair_confidence[symbol] = confidence


def update_risk(band: str, peak_equity: float):
    global _last_band, _last_peak_equity
    _last_band = band
    _last_peak_equity = peak_equity


def update_signals(signals: list):
    global _last_signals
    _last_signals = signals[-20:]


def update_regimes(regimes: list):
    global _last_regimes
    _last_regimes = regimes[-24:]
