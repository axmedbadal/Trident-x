"""Tests for Tier 2 (per-regime consensus weights) and Tier 3 (volatility
scaling + walk-forward retraining) of the TRIDENT-X upgrade roadmap."""
import types

import numpy as np
import pytest

from config.settings import settings


# ── Tier 2: per-regime dynamic consensus weights ───────────────────────────
class TestConsensusTier2:
    def _fresh_consensus(self):
        from council.consensus import consensus
        consensus.reset()
        return consensus

    def test_default_regime_weights_exist_for_all_engines(self):
        c = self._fresh_consensus()
        for regime, weights in c.get_weights().items():
            for eng in ("sniper", "smc", "momentum", "mean_reversion", "price_action"):
                assert eng in weights
                assert weights[eng] > 0

    def test_get_weight_is_regime_aware(self):
        c = self._fresh_consensus()
        assert c._get_weight("sniper", "TRENDING_UP") == 0.30
        assert c._get_weight("smc", "TRENDING_UP") == 0.25

    def test_backtest_metrics_boost_winning_engine(self):
        c = self._fresh_consensus()
        metrics = {"sniper": {"TRENDING_UP": {"wins": 50, "losses": 10}}}
        before = c.get_weights("TRENDING_UP")["sniper"]
        c.apply_backtest_metrics(metrics)
        after = c.get_weights("TRENDING_UP")["sniper"]
        assert after > before
        assert abs(sum(c.get_weights("TRENDING_UP").values()) - 1.0) < 1e-6

    def test_backtest_metrics_penalize_losing_engine(self):
        c = self._fresh_consensus()
        metrics = {"sniper": {"TRENDING_UP": {"wins": 5, "losses": 45}}}
        before = c.get_weights("TRENDING_UP")["sniper"]
        c.apply_backtest_metrics(metrics)
        after = c.get_weights("TRENDING_UP")["sniper"]
        assert after < before

    def test_backtest_metrics_ignore_small_samples(self):
        c = self._fresh_consensus()
        metrics = {"sniper": {"TRENDING_UP": {"wins": 1, "losses": 0}}}
        before = c.get_weights("TRENDING_UP")["sniper"]
        c.apply_backtest_metrics(metrics)
        assert c.get_weights("TRENDING_UP")["sniper"] == before

    def test_adjust_weights_keeps_legacy_view_normalized(self):
        c = self._fresh_consensus()
        c.adjust_weights()
        assert "sniper" in c._dynamic_weights
        assert abs(sum(c._dynamic_weights.values()) - 1.0) < 0.01

    def test_adjust_weights_keeps_per_regime_normalized(self):
        c = self._fresh_consensus()
        c.adjust_weights()
        for regime, weights in c.get_weights().items():
            assert abs(sum(weights.values()) - 1.0) < 1e-6


# ── Tier 3a: volatility-targeted sizing ────────────────────────────────────
class TestVolatilityScaling:
    def _size(self, atr_pct):
        from risk.position_sizer import PositionSizer
        sizer = PositionSizer()
        signal = {"symbol": "SOLUSDT", "direction": "BUY", "confidence": 0.8}
        feats = {"atr_pct": atr_pct, "atr_14": 1.0, "close": 100.0, "sl_distance": 1.5}
        return sizer.size(signal, 10000.0, feats, "TRENDING_UP", "GREEN", 0.6, [])

    def test_high_vol_shrinks_position(self):
        assert self._size(0.12) < self._size(0.03)

    def test_size_stays_positive_at_extreme_vol(self):
        assert self._size(0.50) > 0

    def test_config_present(self):
        assert settings.VOL_TARGET_ATR_PCT > 0
        assert 0 < settings.VOL_SCALE_FLOOR <= 1.0


# ── Tier 3b: walk-forward retraining gates ─────────────────────────────────
class TestWalkForwardRetraining:
    def test_regime_validate_rejects_nan_model(self):
        from ml.regime import RegimeDetector
        bad = types.SimpleNamespace(
            startprob_=np.array([0.5, 0.5]),
            transmat_=np.array([[np.nan, 0.5], [0.5, 0.5]]),
            means_=np.array([[0.0], [1.0]]),
        )
        assert not RegimeDetector._validate_model(bad)

    def test_regime_validate_accepts_valid_model(self):
        from ml.regime import RegimeDetector
        good = types.SimpleNamespace(
            startprob_=np.array([0.5, 0.5]),
            transmat_=np.array([[0.5, 0.5], [0.5, 0.5]]),
            means_=np.array([[0.0], [1.0]]),
        )
        assert RegimeDetector._validate_model(good)

    def test_meta_labeler_skips_walk_forward_with_insufficient_data(self):
        from ml.meta_labeler import MetaLabeler
        ml = MetaLabeler()
        candles = [{"timestamp": i, "open": 100.0, "high": 101.0, "low": 99.0,
                    "close": 100.0 + i * 0.01, "volume": 1000.0} for i in range(100)]
        ml._train("WFTEST", candles)
        assert not ml.is_trained.get("WFTEST", False)

    def test_walk_forward_retrain_returns_false_without_data(self):
        from ml.meta_labeler import MetaLabeler
        ml = MetaLabeler()
        assert ml.walk_forward_retrain("WFTEST2", []) is False


# ── Tier 1: Sim accounting with engine attribution ─────────────────────────
class TestSimAttribution:
    def test_close_attributes_pnl_to_engines(self):
        from backtest.engine import Sim
        sim = Sim(10000)
        feats = {"close": 100.0, "atr_14": 1.0, "atr_pct": 0.8, "rsi_14": 50.0}
        sig = {"symbol": "SOLUSDT", "direction": "BUY", "confidence": 0.9,
               "engines": ["smc", "momentum"]}
        assert sim.open(sig, feats, "TRENDING_UP", 0)
        p = sim.positions[0]
        sim.close(p, p["entry_price"] * 1.05, "TAKE_PROFIT", 1)
        assert sim.metrics["smc"]["TRENDING_UP"]["wins"] == 1
        assert sim.metrics["momentum"]["TRENDING_UP"]["wins"] == 1

    def test_close_attributes_loss_to_engines(self):
        from backtest.engine import Sim
        sim = Sim(10000)
        feats = {"close": 100.0, "atr_14": 1.0, "atr_pct": 0.8, "rsi_14": 50.0}
        sig = {"symbol": "XRPUSDT", "direction": "BUY", "confidence": 0.9,
               "engines": ["sniper"]}
        assert sim.open(sig, feats, "MEAN_REVERTING", 0)
        p = sim.positions[0]
        sim.close(p, p["entry_price"] * 0.95, "STOP_LOSS", 1)
        assert sim.metrics["sniper"]["MEAN_REVERTING"]["losses"] == 1