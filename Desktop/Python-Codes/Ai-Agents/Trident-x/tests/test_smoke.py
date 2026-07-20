"""
Smoke tests: verify the whole system can initialise without a network connection.
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest


class TestSystemSmoke:
    """Import-level smoke tests for production safety."""

    def test_import_main_module(self):
        import main
        assert hasattr(main, "entry")
        assert hasattr(main, "graceful_shutdown")

    def test_settings_load(self):
        from config.settings import settings
        assert settings.PAIRS == ["SOLUSDT", "XRPUSDT", "ADAUSDT"]
        assert settings.PRIMARY_TF == "5m"
        assert settings.PAPER_TRADING is True
        assert settings.BINANCE_FUTURES_URL == "https://fapi.binance.com"

    def test_state_manager_initialises(self):
        from core.state_manager import state_manager
        assert state_manager is not None

    def test_feature_engineer(self):
        from features.engineer import features
        assert callable(features.compute)

    def test_regime_detector(self):
        from ml.regime import regime_detector
        assert regime_detector is not None

    def test_meta_labeler(self):
        from ml.meta_labeler import meta_labeler
        assert meta_labeler is not None

    def test_all_strategy_engines_exist(self):
        from strategies.smc import smc_engine
        from strategies.momentum import momentum_engine
        from strategies.mean_reversion import mean_reversion_engine
        from strategies.sniper import sniper_engine
        engines = [smc_engine, momentum_engine, mean_reversion_engine, sniper_engine]
        assert all(e is not None for e in engines)

    def test_council_components(self):
        from council.consensus import consensus
        from council.gate import SignalIntegrityGate
        assert consensus is not None
        gate = SignalIntegrityGate()
        assert gate is not None

    def test_risk_components(self):
        from risk.position_sizer import PositionSizer
        from risk.governor import RiskGovernor
        from risk.correlation_regime import correlation_regime
        assert PositionSizer() is not None
        assert RiskGovernor() is not None
        assert correlation_regime is not None

    def test_execution_components(self):
        from execution.manager import ExecutionManager
        assert ExecutionManager() is not None

    def test_data_clients(self):
        from data.ws_client import BinanceWebSocketClient
        from data.rest_client import BinanceRestClient
        assert BinanceWebSocketClient is not None
        assert BinanceRestClient is not None

    def test_api_server_imports(self):
        from api.server import app, start_server
        assert app is not None
        assert callable(start_server)

    def test_api_health_endpoint(self):
        from api.server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_api_monitor_endpoints(self):
        from api.server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        for path in ["/api/monitor/stats", "/api/monitor/logs", "/api/monitor/errors"]:
            resp = client.get(path)
            assert resp.status_code == 200, f"{path} returned {resp.status_code}"
            assert isinstance(resp.json(), (dict, list))


class TestSystemMonitor:
    def test_uptime(self):
        from core.monitoring import SystemMonitor
        m = SystemMonitor()
        assert m.uptime_seconds >= 0
        assert "h" in m.uptime_str or m.uptime_seconds < 3600

    def test_stats_dict(self):
        from core.monitoring import SystemMonitor
        m = SystemMonitor()
        stats = m.stats()
        assert "uptime_seconds" in stats
        assert "memory_mb" in stats
        assert "cpu_pct" in stats

    def test_log_capture_handler(self):
        import logging
        from core.monitoring import LogCaptureHandler
        handler = LogCaptureHandler(capacity=10)
        logger = logging.getLogger("test_monitor")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.warning("test warning")
        logger.error("test error")
        logs = handler.get_recent()
        assert len(logs) >= 2
        error_logs = handler.get_errors(limit=5)
        assert any("test error" in r["msg"] for r in error_logs)


class TestAsyncSmoke:
    """Async smoke tests requiring an event loop."""

    @pytest.mark.asyncio
    async def test_correlation_regime_async(self):
        from risk.correlation_regime import CorrelationRegimeFilter
        filt = CorrelationRegimeFilter()
        filt._regime = {
            "status": "NORMAL",
            "alert_level": "GREEN",
            "suppressed_pairs": [],
            "correlations": {},
        }
        filt._last_update = 9999999999
        suppressed, reason = await filt.is_suppressed("SOLUSDT")
        assert not suppressed

    @pytest.mark.asyncio
    async def test_rest_client_fetch_without_network_is_graceful(self):
        """Ensure REST client returns empty data gracefully when offline."""
        from data.rest_client import BinanceRestClient
        client = BinanceRestClient()
        result = await client.fetch_klines("SOLUSDT", "1m", 1)
        assert isinstance(result, list)
        await client.close()
