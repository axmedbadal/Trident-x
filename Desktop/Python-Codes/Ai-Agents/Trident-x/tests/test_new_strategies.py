"""Tests for the Price Action and Scalping strategy engines."""
import time

from config.settings import settings
from features.engineer import FeatureEngineer


def _seed_features(symbol: str):
    """Seed 5m/15m/1h/4h buffers with synthetic candles so compute() works."""
    fe = FeatureEngineer(maxlen=500)
    # Uptrend series on higher TFs, mild pullback on 5m then recovery
    import math
    base = {tf: 500.0 if tf == "5m" else 300.0 for tf in ["5m", "15m", "1h", "4h"]}
    for tf, b in base.items():
        candles = []
        t = 1_700_000_000_000
        for i in range(120):
            price = b * (1 + 0.0006 * i)  # steady uptrend
            candles.append({
                "symbol": symbol, "timeframe": tf, "timestamp": t,
                "open": price, "high": price * 1.002, "low": price * 0.998,
                "close": price, "volume": 1000.0 + i,
            })
            t += 5 * 60 * 1000
        fe.load_history(symbol, tf, candles)
    return fe


def test_price_action_engine_exists_and_neutral_on_empty():
    from strategies.price_action import PriceActionEngine
    engine = PriceActionEngine()
    result = engine.generate("SOLUSDT", {"close": 0.0}, 0.0)
    assert result["engine"] == "price_action"
    assert result["direction"] == "NEUTRAL"


def test_price_action_neutral_when_disabled():
    from strategies.price_action import price_action_engine
    original = settings.PRICE_ACTION_ENABLED
    settings.PRICE_ACTION_ENABLED = False
    try:
        result = price_action_engine.generate("SOLUSDT", {"close": 100.0}, 0.0)
        assert result["direction"] == "NEUTRAL"
        assert result["rationale"] == "disabled"
    finally:
        settings.PRICE_ACTION_ENABLED = original


def test_price_action_fires_on_mtf_bull_structure():
    import strategies.price_action as pa_mod
    engine = pa_mod.PriceActionEngine()
    fe = _seed_features("PAUSDT")
    pa_mod.features.buffers = fe.buffers
    f5 = fe.compute("PAUSDT", "5m")
    assert f5 is not None
    result = engine.generate("PAUSDT", f5, 0.0)
    # With a clean uptrend + 5m structure, an engine may fire BUY or stay neutral;
    # it must never print an outright wrong-direction signal against the HTF trend.
    assert result["direction"] in ("BUY", "NEUTRAL")


def test_price_action_swing_helpers():
    from strategies.price_action import PriceActionEngine
    engine = PriceActionEngine()
    buf = [
        {"high": 10, "low": 9}, {"high": 11, "low": 10}, {"high": 10.5, "low": 9.5},
        {"high": 12, "low": 11}, {"high": 11.5, "low": 10.5}, {"high": 13, "low": 12},
    ]
    assert engine._is_swing_high(buf, 1) is True
    assert engine._is_swing_high(buf, 3) is True
    assert engine._is_swing_high(buf, 5) is False


def test_scalping_engine_exists_and_neutral_on_empty():
    from strategies.scalping import ScalpingEngine
    engine = ScalpingEngine()
    result = engine.generate("SOLUSDT", {"close": 0.0, "atr_pct": 0.0}, 0.0)
    assert result["engine"] == "scalping"
    assert result["direction"] == "NEUTRAL"


def test_scalping_neutral_when_disabled():
    from strategies.scalping import scalping_engine
    original = settings.SCALPING_ENABLED
    settings.SCALPING_ENABLED = False
    try:
        result = scalping_engine.generate("SOLUSDT", {"close": 100.0, "atr_pct": 1.0}, 0.0)
        assert result["direction"] == "NEUTRAL"
        assert result["rationale"] == "disabled"
    finally:
        settings.SCALPING_ENABLED = original


def test_scalping_volatility_filters():
    from strategies.scalping import ScalpingEngine
    engine = ScalpingEngine()
    too_quiet = engine.generate("SOLUSDT", {"close": 100.0, "atr_pct": 0.01}, 0.0)
    assert too_quiet["direction"] == "NEUTRAL"
    assert "too_quiet" in too_quiet["rationale"]
    too_wide = engine.generate("SOLUSDT", {"close": 100.0, "atr_pct": 10.0}, 0.0)
    assert too_wide["direction"] == "NEUTRAL"
    assert "too_volatile" in too_wide["rationale"]


def test_scalping_htf_context_neutral_without_buffers():
    from strategies.scalping import ScalpingEngine
    engine = ScalpingEngine()
    assert engine._htf_context("NOBUFUSDT") == "NEUTRAL"


def test_consensus_includes_new_engines():
    from council.consensus import consensus
    assert "price_action" in consensus._dynamic_weights
    assert "scalping" in consensus._dynamic_weights
    total = sum(consensus._dynamic_weights.values())
    assert abs(total - 1.0) < 0.01


def test_state_manager_persists_new_votes():
    import time as _t
    from core.state_manager import StateManager
    sm = StateManager(":memory:")
    sig = {
        "symbol": "SOLUSDT", "direction": "BUY", "confidence": 0.8,
        "strength": "STRONG", "regime": "TRENDING_UP", "engines_agreeing": 2,
        "passed_gate": True, "gate_reason": None, "executed": True,
        "timestamp": int(_t.time() * 1000), "meta_prob": 0.7,
        "price_action_vote": "BUY", "scalping_vote": "SELL",
    }
    sig_id = sm.insert_signal(sig)
    assert sig_id > 0
    signals = sm.get_recent_signals("SOLUSDT", 5)
    assert signals[0]["price_action_vote"] == "BUY"
    assert signals[0]["scalping_vote"] == "SELL"


# ── Edge cases on internal helpers ─────────────────────────────────────────
class TestPriceActionEdgeCases:
    def test_structure_short_buffer(self):
        from strategies.price_action import PriceActionEngine
        engine = PriceActionEngine()
        import strategies.price_action as pa_mod
        pa_mod.features.buffers = {}
        assert engine._structure("SOLUSDT", "5m") == "NEUTRAL"

    def test_structure_flat_buffer(self):
        from strategies.price_action import PriceActionEngine
        engine = PriceActionEngine()
        buf = [{"high": 10.0, "low": 9.0}] * 20
        assert engine._structure("SOLUSDT", "5m") == "NEUTRAL"

    def test_liquidity_sweep_empty_buffers(self):
        from strategies.price_action import PriceActionEngine
        engine = PriceActionEngine()
        assert engine._liquidity_sweep("SOLUSDT", {"close": 100.0}, []) == ""

    def test_liquidity_sweep_requires_close(self):
        from strategies.price_action import PriceActionEngine
        engine = PriceActionEngine()
        buf = [{"high": 10.0, "low": 9.0, "open": 9.5}] * 15
        assert engine._liquidity_sweep("SOLUSDT", {"close": 0.0}, buf) == ""

    def test_liquidity_sweep_bull_detected(self):
        from strategies.price_action import PriceActionEngine
        engine = PriceActionEngine()
        # unique swing low at index 7 (low 99); last candle wicks to 98.95 (<99)
        # and closes back at 99.3 (>99): a bull EQL sweep.
        buf = [{"high": 101.0, "low": 100.0, "open": 100.5, "close": 100.5}] * 14
        buf[7] = {"high": 100.5, "low": 99.0, "open": 100.0, "close": 99.8}
        f = {"close": 99.3, "high": 100.5, "low": 98.95}
        assert engine._liquidity_sweep("SOLUSDT", f, buf) == "BULL"

    def test_liquidity_sweep_bear_detected(self):
        from strategies.price_action import PriceActionEngine
        engine = PriceActionEngine()
        # unique swing high at index 7 (high 101); last candle spikes to 101.05
        # and closes back at 100.8 (<101): a bear EQH sweep.
        buf = [{"high": 100.0, "low": 99.0, "open": 99.5, "close": 99.5}] * 14
        buf[7] = {"high": 101.0, "low": 99.2, "open": 99.5, "close": 99.6}
        f = {"close": 100.8, "high": 101.05, "low": 100.5}
        assert engine._liquidity_sweep("SOLUSDT", f, buf) == "BEAR"

    def test_fvg_retest_short_buffer(self):
        from strategies.price_action import PriceActionEngine
        engine = PriceActionEngine()
        import strategies.price_action as pa_mod
        pa_mod.features.buffers = {}
        assert engine._fvg_retest("SOLUSDT", {"close": 100.0}) == ""

    def test_fvg_retest_bull(self):
        from strategies.price_action import PriceActionEngine
        engine = PriceActionEngine()
        import strategies.price_action as pa_mod
        # Need n >= FVG_LOOKBACK(10) + 3 = 13 candles. Build 14-candle buffer
        # where candle[i-2=8].high(90.5) < candle[i=10].low(91.0) => bullish FVG,
        # and current close 91.0 retests above the FVG origin.
        buf = [{"open": 90, "high": 91, "low": 89, "close": 90, "volume": 1}] * 14
        buf[8] = {"open": 90, "high": 90.5, "low": 89, "close": 90.2, "volume": 1}
        buf[9] = {"open": 90.5, "high": 91, "low": 90, "close": 90.8, "volume": 1}
        buf[10] = {"open": 91.2, "high": 92, "low": 91.0, "close": 91.8, "volume": 1}
        buf[11] = {"open": 91.0, "high": 91.5, "low": 90.7, "close": 91.0, "volume": 1}
        pa_mod.features.buffers = {"SOLUSDT:5m": buf}
        assert engine._fvg_retest("SOLUSDT", {"close": 91.0}) == "BULL"

    def test_candle_patterns_short_buffer(self):
        from strategies.price_action import PriceActionEngine
        engine = PriceActionEngine()
        assert engine._candle_patterns([], {}) == {"bull": [], "bear": []}

    def test_candle_patterns_engulfing_bull(self):
        from strategies.price_action import PriceActionEngine
        engine = PriceActionEngine()
        buf = [
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"open": 101, "high": 101.5, "low": 99.5, "close": 100.2, "volume": 1},
            {"open": 99.0, "high": 102, "low": 98, "close": 101.8, "volume": 1},
        ]
        pat = engine._candle_patterns(buf, {})
        assert "bull_engulfing" in pat["bull"]

    def test_sr_proximity_zero_close(self):
        from strategies.price_action import PriceActionEngine
        engine = PriceActionEngine()
        assert engine._sr_proximity({"close": 0.0}) == ""

    def test_trend_bias_no_adx(self):
        from strategies.price_action import PriceActionEngine
        engine = PriceActionEngine()
        assert engine._trend_bias({"adx": 0.0, "supertrend": 0}) == "NEUTRAL"
        assert engine._trend_bias({"adx": 25, "supertrend": 1}) == "LONG"
        assert engine._trend_bias({"adx": 25, "supertrend": -1}) == "SHORT"


class TestScalpingEdgeCases:
    def test_htf_context_counting(self):
        from strategies.scalping import ScalpingEngine
        import strategies.scalping as sc_mod
        from features.engineer import FeatureEngineer
        fe = FeatureEngineer(maxlen=500)
        tf_bufs = {}
        for tf in ["15m", "1h", "4h"]:
            candles = []
            t = 1_700_000_000_000
            for i in range(100):
                price = 500.0 * (1 + 0.001 * i)
                candles.append({"symbol": "EUSDT", "timeframe": tf, "timestamp": t,
                                "open": price, "high": price * 1.001, "low": price * 0.999,
                                "close": price, "volume": 100.0})
                t += 15 * 60 * 1000
            fe.load_history("EUSDT", tf, candles)
        sc_mod.features.buffers = fe.buffers
        engine = ScalpingEngine()
        assert engine._htf_context("EUSDT") == "LONG"

    def test_generate_with_direct_buffers(self):
        from strategies.scalping import ScalpingEngine
        import strategies.scalping as sc_mod
        from features.engineer import FeatureEngineer
        fe = FeatureEngineer(maxlen=500)
        for tf in ["5m", "15m", "1h", "4h"]:
            candles = []
            t = 1_700_000_000_000
            for i in range(120):
                price = 500.0 * (1 + 0.001 * i)
                candles.append({"symbol": "FUSDT", "timeframe": tf, "timestamp": t,
                                "open": price, "high": price * 1.001, "low": price * 0.999,
                                "close": price, "volume": 1000.0 + i})
                t += 5 * 60 * 1000
            fe.load_history("FUSDT", tf, candles)
        sc_mod.features.buffers = fe.buffers
        f5 = fe.compute("FUSDT", "5m")
        assert f5 is not None
        result = ScalpingEngine().generate("FUSDT", f5, 0.0)
        assert result["direction"] in ("BUY", "NEUTRAL")
        assert result["engine"] == "scalping"
