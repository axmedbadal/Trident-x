"""Edge-case and failure-mode tests for TRIDENT-X subsystems."""
import asyncio
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Execution manager edge cases ──────────────────────────────────────────
class TestExecutionManagerEdgeCases:
    def test_unrealized_pct_invalid_close(self):
        from execution.manager import ExecutionManager
        mgr = ExecutionManager()
        pos = {"direction": "BUY", "entry_price": 100.0, "size_usd": 1000.0}
        assert mgr._unrealized_pct(pos, 0.0) is None
        assert mgr._unrealized_pct(pos, -10.0) is None

    def test_unrealized_pct_invalid_entry(self):
        from execution.manager import ExecutionManager
        mgr = ExecutionManager()
        pos = {"direction": "BUY", "entry_price": 0.0, "size_usd": 1000.0}
        assert mgr._unrealized_pct(pos, 110.0) is None

    def test_open_position_rejects_when_max_positions(self):
        from execution.manager import ExecutionManager
        from config.settings import settings
        mgr = ExecutionManager()
        mgr.positions = [{"symbol": f"SYM{i}"} for i in range(settings.MAX_POSITIONS)]
        result = _run(mgr.open_position(
            "SOLUSDT",
            {"direction": "BUY", "confidence": 0.9},
            {"close": 100.0, "atr_14": 1.0, "vol_sma20_ratio": 1.0, "z_score_20": 0.0, "vol_24h_vs_7d": 1.0},
            10000.0,
            "TRENDING_UP",
            "NORMAL",
            0.5,
        ))
        assert result is False


# ── Partial exit engine edge cases ────────────────────────────────────────
class TestPartialExitEdgeCases:
    def test_check_returns_zero_with_invalid_inputs(self):
        from execution.partial_exit import PartialExitEngine
        engine = PartialExitEngine()
        exit_pct, reason, updated = engine.check(
            {"entry_price": 0.0, "stop_loss": 90.0, "direction": "BUY"},
            {"close": 100.0, "atr_14": 1.0},
        )
        assert exit_pct == 0.0
        assert reason == ""
        assert updated["entry_price"] == 0.0

    def test_no_double_tp_hit(self):
        from execution.partial_exit import PartialExitEngine
        engine = PartialExitEngine()
        pos = {"entry_price": 100.0, "stop_loss": 90.0, "direction": "BUY", "tp1_hit": 1}
        exit_pct, reason, updated = engine.check(
            pos,
            {"close": 200.0, "atr_14": 1.0},
        )
        assert "tp1" not in reason
        assert updated["tp1_hit"] == 1


# ── Risk governor edge cases ──────────────────────────────────────────────
class TestRiskGovernorEdgeCases:
    def test_governor_green_band_allows_full_scale(self):
        from risk.governor import RiskGovernor
        gov = RiskGovernor()
        band = gov.update(10000.0)
        assert band == "GREEN"
        assert gov.scale() == 1.0

    def test_governor_blocks_excessive_drawdown(self):
        from risk.governor import RiskGovernor
        gov = RiskGovernor()
        gov.high_watermark = 10000.0
        band = gov.update(7900.0)
        assert band == "BLACK"
        assert gov.scale() == 0.0


# ── Position sizer edge cases ─────────────────────────────────────────────
class TestPositionSizerEdgeCases:
    def test_size_zero_on_missing_features(self):
        from risk.position_sizer import PositionSizer
        sizer = PositionSizer()
        size = sizer.size(
            {"direction": "BUY", "confidence": 0.9},
            10000.0,
            {},
            "TRENDING_UP",
            "NORMAL",
            0.5,
            [],
        )
        assert size == 0.0

    def test_correlation_penalty_with_uncorrelated_positions(self):
        from risk.position_sizer import PositionSizer
        sizer = PositionSizer()
        # Provide one position with no candle data => penalty fallback should handle gracefully
        penalty = sizer._avg_correlation("SOLUSDT", [{"symbol": "XRPUSDT"}])
        assert 0.0 <= penalty <= 1.0


# ── REST client edge cases ────────────────────────────────────────────────
class TestRestClientEdgeCases:
    @pytest.mark.asyncio
    async def test_fetch_prices_returns_dict(self):
        from data.rest_client import BinanceRestClient
        client = BinanceRestClient()
        prices = await client.fetch_prices()
        assert isinstance(prices, dict)
        await client.close()

    @pytest.mark.asyncio
    async def test_fetch_funding_rate_returns_float(self):
        from data.rest_client import BinanceRestClient
        client = BinanceRestClient()
        rate = await client.fetch_funding_rate("SOLUSDT")
        assert isinstance(rate, float)
        await client.close()


# ── Feature engineer edge cases ───────────────────────────────────────────
class TestFeatureEngineerEdgeCases:
    def test_compute_returns_none_with_no_history(self):
        from features.engineer import features
        result = features.compute("SOLUSDT", "5m")
        assert result is None

    def test_compute_with_minimal_candles(self):
        from features.engineer import features
        candles = [
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0}
            for _ in range(50)
        ]
        features.load_history("SOLUSDT", "5m", candles)
        result = features.compute("SOLUSDT", "5m")
        # Some features may not be computable, but it should not crash
        assert isinstance(result, dict)


# ── Meta labeler edge cases ───────────────────────────────────────────────
class TestMetaLabelerEdgeCases:
    def test_predict_with_no_regime(self):
        from ml.meta_labeler import meta_labeler
        prob = meta_labeler.predict("TEST_NO_REGIME", {})
        assert 0.0 <= prob <= 1.0


# ── Correlation regime edge cases ─────────────────────────────────────────
class TestCorrelationRegimeEdgeCases:
    @pytest.mark.asyncio
    async def test_returns_unknown_when_no_data(self):
        from risk.correlation_regime import CorrelationRegimeFilter
        filt = CorrelationRegimeFilter()
        regime = await filt.get_regime()
        assert regime["status"] == "UNKNOWN"
        assert regime["alert_level"] == "GREEN"
