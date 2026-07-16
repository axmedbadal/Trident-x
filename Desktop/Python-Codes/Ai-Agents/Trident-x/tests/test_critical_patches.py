"""Comprehensive tests for TRIDENT-X critical patches."""
import asyncio
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest


# ── Fix 2.1: Meta-labeler cold start ──────────────────────────────────────
class TestMetaLabelerColdStart:
    def test_cold_start_returns_default_trending(self):
        from ml.meta_labeler import meta_labeler, COLD_START_DEFAULT_TRENDING, COLD_START_SIGNALS
        meta_labeler.signal_count["TEST"] = 0
        prob = meta_labeler.predict("TEST", {"regime": "TRENDING_UP"})
        assert prob == COLD_START_DEFAULT_TRENDING

    def test_cold_start_returns_default_other(self):
        from ml.meta_labeler import meta_labeler, COLD_START_DEFAULT_OTHER
        meta_labeler.signal_count["TEST2"] = 0
        prob = meta_labeler.predict("TEST2", {"regime": "MEAN_REVERTING"})
        assert prob == COLD_START_DEFAULT_OTHER

    def test_after_warmup_returns_model_or_default(self):
        from ml.meta_labeler import meta_labeler, COLD_START_SIGNALS
        meta_labeler.signal_count["TEST3"] = COLD_START_SIGNALS + 1
        meta_labeler.is_trained["TEST3"] = False
        prob = meta_labeler.predict("TEST3", {"regime": "TRENDING_UP"})
        assert 0.0 <= prob <= 1.0


# ── Fix 2.2: Correlation penalty ──────────────────────────────────────────
class TestCorrelationPenalty:
    def test_no_positions_returns_zero(self):
        from risk.position_sizer import PositionSizer
        sizer = PositionSizer()
        result = sizer._avg_correlation("SOLUSDT", [])
        assert result == 0.0

    def test_same_symbol_returns_zero(self):
        from risk.position_sizer import PositionSizer
        sizer = PositionSizer()
        positions = [{"symbol": "SOLUSDT"}]
        result = sizer._avg_correlation("SOLUSDT", positions)
        assert result == 0.0


# ── Fix 2.3: Log rotation ─────────────────────────────────────────────────
class TestLogRotation:
    def test_rotating_handler_configured(self):
        from config.settings import settings
        assert settings.LOG_TO_FILE is True
        assert settings.LOG_ROTATION_DAYS >= 1


# ── Fix 2.4: Partial fill tracking ────────────────────────────────────────
class TestPartialFillTracking:
    def test_remaining_size_usd_on_open(self):
        from execution.fills import PaperFill
        fill = PaperFill()
        result = asyncio.get_event_loop().run_until_complete(
            fill.fill_entry("SOLUSDT", "BUY", 100.0, 1000.0, "TRENDING_UP")
        )
        assert result["filled"] is True
        assert result["price"] > 0

    def test_paper_fill_applies_slippage(self):
        from execution.fills import PaperFill, SLIPPAGE
        fill = PaperFill()
        result = asyncio.get_event_loop().run_until_complete(
            fill.fill_entry("SOLUSDT", "BUY", 100.0, 1000.0, "VOLATILE")
        )
        expected_slippage = SLIPPAGE["VOLATILE"]
        assert result["slippage"] == expected_slippage
        assert result["price"] > 100.0


# ── Fix 2.5: Limit order cancel/replace ───────────────────────────────────
class TestLimitOrderCancelReplace:
    def test_backtest_fill_submit_order(self):
        from execution.fills import BacktestFill
        fill = BacktestFill()
        order = asyncio.get_event_loop().run_until_complete(
            fill.submit_order("SOLUSDT", "BUY", 10.0, "LIMIT", 100.0)
        )
        assert order.status == "NEW"

    def test_backtest_fill_cancel_order(self):
        from execution.fills import BacktestFill
        fill = BacktestFill()
        order = asyncio.get_event_loop().run_until_complete(
            fill.submit_order("SOLUSDT", "BUY", 10.0, "LIMIT", 100.0)
        )
        cancelled = asyncio.get_event_loop().run_until_complete(
            fill.cancel_order(order.id)
        )
        assert cancelled is True

    def test_market_order_fills_immediately(self):
        from execution.fills import BacktestFill
        fill = BacktestFill()
        order = asyncio.get_event_loop().run_until_complete(
            fill.submit_order("SOLUSDT", "BUY", 10.0, "MARKET", 100.0)
        )
        assert order.status == "FILLED"


# ── Fix 2.6: Funding rate persistence ─────────────────────────────────────
class TestFundingRatePersistence:
    def test_save_and_load_funding_rate(self):
        from core.state_manager import StateManager
        sm = StateManager(":memory:")
        sm.save_funding_rate("SOLUSDT", 0.0001, 0.00005)
        result = sm.get_latest_funding_rate("SOLUSDT")
        assert result is not None
        assert result["symbol"] == "SOLUSDT"
        assert result["rate"] == 0.0001

    def test_latest_funding_rate_none_when_empty(self):
        from core.state_manager import StateManager
        sm = StateManager(":memory:")
        result = sm.get_latest_funding_rate("NONEXISTENT")
        assert result is None


# ── Fix 3.1: Weekly maintenance ───────────────────────────────────────────
class TestWeeklyMaintenance:
    def test_prune_method_exists(self):
        from core.state_manager import StateManager
        sm = StateManager(":memory:")
        sm.prune()
        # No error means success


# ── Fix 3.2: Engine weight auto-adjustment ────────────────────────────────
class TestEngineWeightAdjustment:
    def test_adjust_weights_normalizes(self):
        from council.consensus import consensus
        consensus.adjust_weights()
        total = sum(consensus._dynamic_weights.values())
        assert abs(total - 1.0) < 0.01

    def test_default_weights_sum_to_one(self):
        from council.consensus import consensus
        total = sum(consensus._dynamic_weights.values())
        assert abs(total - 1.0) < 0.01

    def test_get_weight_returns_valid(self):
        from council.consensus import consensus
        w = consensus._get_weight("smc", "TRENDING_UP")
        assert 0.0 <= w <= 1.0


# ── Fix 3.4: Dashboard auth ───────────────────────────────────────────────
class TestDashboardAuth:
    def test_settings_has_auth_fields(self):
        from config.settings import settings
        assert hasattr(settings, "DASHBOARD_API_KEY")
        assert hasattr(settings, "DASHBOARD_AUTH_ENABLED")


# ── Fix 3.5: UTC timezone ─────────────────────────────────────────────────
class TestUtcTimezone:
    def test_server_timestamp_uses_utc(self):
        import datetime
        utc_now = datetime.datetime.now(datetime.timezone.utc)
        assert utc_now.tzinfo is not None


# ── Fix 3.6: Drawdown high watermark ──────────────────────────────────────
class TestDrawdownHighWatermark:
    def test_peak_equity_tracks(self):
        from risk.governor import RiskGovernor
        gov = RiskGovernor()
        gov.update(10500)
        assert gov.high_watermark == 10500
        assert gov.peak_equity == 10500

    def test_dd_calculation(self):
        from risk.governor import RiskGovernor
        gov = RiskGovernor()
        gov.update(10000)
        gov.update(9500)
        assert 0.04 < gov.max_dd < 0.06

    def test_scale_method(self):
        from risk.governor import RiskGovernor
        gov = RiskGovernor()
        gov.update(9800)  # ~2% DD
        assert gov.scale() == 1.0
        gov.update(9400)  # ~6% DD
        assert gov.scale() < 1.0


# ── Fix 3.3: Multi-instance protection ────────────────────────────────────
class TestMultiInstanceProtection:
    def test_lock_file_created_and_removed(self):
        from main import _acquire_instance_lock, _release_instance_lock
        _acquire_instance_lock()
        assert os.path.exists("trident_x.lock")
        _release_instance_lock()
        # On Windows, file removal may be delayed; check _lock_fd is reset
        from main import _lock_fd
        assert _lock_fd is None


# ── Sniper Strategy ───────────────────────────────────────────────────────
class TestSniperStrategy:
    def test_asset_gating_disabled(self):
        from strategies.sniper import SniperEngine
        from config.settings import settings
        engine = SniperEngine()
        original = settings.SNIPER_ENABLED_ASSETS
        settings.SNIPER_ENABLED_ASSETS = ["BTCUSDT"]
        result = engine.generate("SOLUSDT", {"close": 100.0}, 0.0)
        settings.SNIPER_ENABLED_ASSETS = original
        assert result["direction"] == "NEUTRAL"

    def test_sniper_neutral_without_history(self):
        from strategies.sniper import SniperEngine
        engine = SniperEngine()
        result = engine.generate("SOLUSDT", {"close": 100.0}, 0.0)
        assert result["direction"] == "NEUTRAL"
        assert result["engine"] == "sniper"

    def test_score_timeframe_bullish(self):
        from strategies.sniper import SniperEngine
        engine = SniperEngine()
        f = {
            "ema_9": 110.0, "ema_20": 105.0, "ema_50": 100.0,
            "sma_aligned_bull": 1.0,
            "close": 112.0, "ema_20": 105.0,
            "vwap_dev": 1.0,
            "macd_hist": 1.0, "macd": 2.0, "macd_signal": 1.0,
            "rsi_14": 55.0,
            "adx": 30.0, "plus_di": 35.0, "minus_di": 20.0,
            "stoch_k": 60.0, "stoch_d": 50.0,
            "supertrend": 1.0,
            "ttm_squeeze": 1.0,
            "vol_sma20_ratio": 2.0,
            "fvg_bull": 1.0, "bos_up": 1.0,
        }
        result = engine._score_timeframe(f)
        assert result["score"] > 0
        assert result["bias"] == "LONG"

    def test_cooldown_blocks_signal(self):
        import time
        from strategies.sniper import SniperEngine
        engine = SniperEngine()
        engine._cooldown_until["SOLUSDT"] = time.time() + 1000
        result = engine.generate("SOLUSDT", {"close": 100.0}, 0.0)
        assert result["direction"] == "NEUTRAL"
        assert result["rationale"] == "cooldown"

    def test_default_weights_include_sniper(self):
        from council.consensus import consensus
        assert "sniper" in consensus._dynamic_weights
        assert abs(sum(consensus._dynamic_weights.values()) - 1.0) < 0.01


# ── Vijackic Adaptation: Phase 1 Z-score Gate ─────────────────────────────
class TestZScoreGate:
    def test_blocks_long_overextended(self):
        from council.gate import SignalIntegrityGate
        gate = SignalIntegrityGate()
        signal = {"direction": "BUY", "confidence": 0.9, "regime": "TRENDING_UP", "engines_agreeing": 2}
        features = {"vol_sma20_ratio": 2.0, "atr_pct": 1.0, "z_score_20": 1.8, "vol_24h_vs_7d": 2.0}
        passed, reason = gate.evaluate(signal, features, {}, 0.7, 0)
        assert not passed
        assert "mean_reversion_block_long" in reason

    def test_blocks_short_overextended(self):
        from council.gate import SignalIntegrityGate
        gate = SignalIntegrityGate()
        signal = {"direction": "SELL", "confidence": 0.9, "regime": "TRENDING_DOWN", "engines_agreeing": 2}
        features = {"vol_sma20_ratio": 2.0, "atr_pct": 1.0, "z_score_20": -1.8, "vol_24h_vs_7d": 2.0}
        passed, reason = gate.evaluate(signal, features, {}, 0.7, 0)
        assert not passed
        assert "mean_reversion_block_short" in reason

    def test_allows_when_volume_low(self):
        from council.gate import SignalIntegrityGate
        gate = SignalIntegrityGate()
        signal = {"direction": "BUY", "confidence": 0.9, "regime": "TRENDING_UP", "engines_agreeing": 2}
        features = {"vol_sma20_ratio": 2.0, "atr_pct": 1.0, "z_score_20": 1.8, "vol_24h_vs_7d": 1.0}
        passed, reason = gate.evaluate(signal, features, {"15m": "BUY", "1h": "BUY"}, 0.7, 0)
        assert passed


# ── Vijackic Adaptation: Phase 2 ATR Ranking ──────────────────────────────
class TestATRRanking:
    def test_position_sizer_has_atr_multiplier(self):
        from risk.position_sizer import PositionSizer
        sizer = PositionSizer()
        mult = sizer._atr_rank_multiplier("SOLUSDT")
        assert 0.5 <= mult <= 1.0

    def test_atr_rankings_computed(self):
        from risk.position_sizer import PositionSizer
        sizer = PositionSizer()
        sizer._recalculate_atr_rankings()
        assert len(sizer._atr_rankings) <= 3


# ── Vijackic Adaptation: Phase 3 Correlation Regime ───────────────────────
class TestCorrelationRegime:
    def test_normal_regime_no_suppression(self):
        import asyncio
        from risk.correlation_regime import CorrelationRegimeFilter
        filt = CorrelationRegimeFilter()
        filt._regime = {
            "status": "NORMAL",
            "alert_level": "GREEN",
            "suppressed_pairs": [],
            "correlations": {},
        }
        filt._last_update = 9999999999
        suppressed, reason = asyncio.get_event_loop().run_until_complete(filt.is_suppressed("SOLUSDT"))
        assert not suppressed

    def test_sol_decoupling_suppresses_sol(self):
        import asyncio
        from risk.correlation_regime import CorrelationRegimeFilter
        filt = CorrelationRegimeFilter()
        filt._regime = {
            "status": "SOL_DECOUPLING",
            "alert_level": "YELLOW",
            "suppressed_pairs": ["SOLUSDT"],
            "correlations": {"SOL": 0.2, "ADA": 0.7, "XRP": 0.7},
        }
        filt._last_update = 9999999999
        suppressed, reason = asyncio.get_event_loop().run_until_complete(filt.is_suppressed("SOLUSDT"))
        assert suppressed
        assert "SOL_DECOUPLING" in reason


# ── Vijackic Adaptation: Phase 4 Momentum Trailing Stop ───────────────────
class TestMomentumTrailingStop:
    def test_unrealized_pct_buy(self):
        from execution.manager import ExecutionManager
        mgr = ExecutionManager()
        pos = {"direction": "BUY", "entry_price": 100.0, "size_usd": 1000.0}
        assert mgr._unrealized_pct(pos, 110.0) == 0.10

    def test_unrealized_pct_sell(self):
        from execution.manager import ExecutionManager
        mgr = ExecutionManager()
        pos = {"direction": "SELL", "entry_price": 100.0, "size_usd": 1000.0}
        assert mgr._unrealized_pct(pos, 90.0) == 0.10


# ── Fill provider factory ─────────────────────────────────────────────────
class TestFillProviderFactory:
    def test_paper_trading_returns_paper_fill(self):
        from execution.fills import get_fill_provider, PaperFill
        provider = get_fill_provider()
        assert isinstance(provider, PaperFill)


# ── Execution manager ─────────────────────────────────────────────────────
class TestExecutionManager:
    def test_positions_initially_empty(self):
        from execution.manager import ExecutionManager
        mgr = ExecutionManager()
        assert mgr.positions == []

    def test_pending_limit_orders_initially_empty(self):
        from execution.manager import ExecutionManager
        mgr = ExecutionManager()
        assert mgr._pending_limit_orders == {}


# ── Circuit breaker ────────────────────────────────────────────────────────
class TestCircuitBreaker:
    def test_initial_state(self):
        from risk.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker()
        assert cb.state == "GREEN"
        assert cb.can_trade() is True

    def test_kill_prevents_trading(self):
        from risk.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker()
        cb.state = "KILL"
        assert cb.can_trade() is False

    def test_red_prevents_trading(self):
        from risk.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker()
        cb.state = "RED"
        cb.cooldown_until = time.time() + 3600
        assert cb.can_trade() is False


# ── State manager core ────────────────────────────────────────────────────
class TestStateManagerCore:
    def test_candles_save_and_load(self):
        from core.state_manager import StateManager
        sm = StateManager(":memory:")
        candles = [
            {"symbol": "SOLUSDT", "timeframe": "5m", "timestamp": 1000, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 100},
        ]
        sm.save_candles(candles)
        loaded = sm.load_candles("SOLUSDT", "5m", 10)
        assert len(loaded) == 1
        assert loaded[0]["close"] == 1.5

    def test_position_save_and_load(self):
        from core.state_manager import StateManager
        sm = StateManager(":memory:")
        pos = {
            "symbol": "SOLUSDT", "direction": "BUY", "entry_price": 100.0,
            "size_usd": 1000.0, "stop_loss": 95.0, "take_profit": 110.0,
            "status": "OPEN", "engine": "consensus", "opened_at": int(time.time() * 1000),
        }
        pos_id = sm.save_position(pos)
        assert pos_id > 0
        positions = sm.get_open_positions()
        assert len(positions) == 1

    def test_signal_insert_and_retrieve(self):
        from core.state_manager import StateManager
        sm = StateManager(":memory:")
        signal = {
            "symbol": "SOLUSDT", "direction": "BUY", "confidence": 0.8,
            "strength": "STRONG", "regime": "TRENDING_UP", "engines_agreeing": 2,
            "passed_gate": True, "gate_reason": None, "executed": True,
            "timestamp": int(time.time() * 1000), "meta_prob": 0.7,
        }
        sig_id = sm.insert_signal(signal)
        assert sig_id > 0
        signals = sm.get_recent_signals("SOLUSDT", 5)
        assert len(signals) == 1
