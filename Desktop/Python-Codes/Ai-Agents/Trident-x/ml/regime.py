import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False

logger = logging.getLogger(__name__)

STATES = ["TRENDING_UP", "TRENDING_DOWN", "MEAN_REVERTING", "ACCUMULATION", "DISTRIBUTION"]


@dataclass
class RegimeState:
    state: str = "MEAN_REVERTING"
    confidence: float = 0.5
    hurst: float = 0.5


class RegimeDetector:
    def __init__(self, n_states: int = 5):
        self.n_states = n_states
        self.models: Dict[str, object] = {}
        self._last_train: Dict[str, float] = {}
        self._low_conf_count: Dict[str, int] = {}
        self._current: Dict[str, RegimeState] = {}

    def detect(self, symbol: str, df: pd.DataFrame) -> RegimeState:
        try:
            state = self._hmm_detect(symbol, df) if HMM_AVAILABLE else self._rule_detect(df)
        except Exception as e:
            logger.warning(f"HMM detect failed {symbol}: {e}, using rule fallback")
            state = self._rule_detect(df)
        self._current[symbol] = state
        return state

    async def maybe_retrain(self, symbol: str, df: pd.DataFrame):
        now = time.time()
        last = self._last_train.get(symbol, 0)
        if now - last > 7 * 86400 or self._low_conf_count.get(symbol, 0) > 288:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._train, symbol, df),
                    timeout=10,
                )
                self._last_train[symbol] = now
                self._low_conf_count[symbol] = 0
            except asyncio.TimeoutError:
                logger.warning(f"HMM retrain timeout {symbol}")
            except Exception as e:
                logger.warning(f"HMM retrain failed {symbol}: {e}")

    def _train(self, symbol: str, df: pd.DataFrame):
        if len(df) < 200:
            return
        feats = self._regime_features(df).dropna()
        if len(feats) < 150:
            return
        # Dynamically reduce states for small datasets
        n_states = min(self.n_states, max(2, len(feats) // 100))
        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, random_state=42,
                            tol=1e-4, verbose=False)
        try:
            model.fit(feats.values)
        except Exception as e:
            logger.warning(f"HMM fit failed {symbol}: {e}")
            return
        # Validate: reject degenerate models (NaN params or zero-sum transmat rows)
        if np.any(np.isnan(model.startprob_)) or np.any(np.isnan(model.transmat_)):
            logger.warning(f"HMM model degenerate for {symbol} (NaN params), keeping previous")
            return
        if np.any(np.isnan(model.means_)):
            logger.warning(f"HMM model degenerate for {symbol} (NaN means), keeping previous")
            return
        if np.any(model.transmat_.sum(axis=1) == 0):
            logger.warning(f"HMM model degenerate for {symbol} (zero-sum transmat rows), keeping previous")
            return
        self.models[symbol] = model
        logger.info(f"Retrained HMM for {symbol} ({n_states} states, {len(feats)} samples)")

    def _hmm_detect(self, symbol: str, df: pd.DataFrame) -> RegimeState:
        model = self.models.get(symbol)
        feats = self._regime_features(df)
        if model is None or len(feats) < 1:
            return self._rule_detect(df)
        x = feats.iloc[[-1]].values
        h = self._hurst(df["close"].values[-100:])
        score = model.score_samples(x)[0]
        state_idx = int(model.predict(x)[0])
        label = self._map_state(state_idx, df, h)
        conf = max(0.1, min(1.0, 1.0 / (1.0 + np.exp(-score / 100.0))))
        if conf < 0.60:
            self._low_conf_count[symbol] = self._low_conf_count.get(symbol, 0) + 1
        return RegimeState(label, conf, h)

    def _rule_detect(self, df: pd.DataFrame) -> RegimeState:
        c = df["close"]
        rsi = self._rsi(c)
        adx = self._adx_simple(df)
        ema_slope = c.ewm(span=50).mean().diff().iloc[-1]
        h = self._hurst(c.values[-100:])
        rsi_val = rsi.iloc[-1] if len(rsi) else 50
        adx_val = adx.iloc[-1] if len(adx) else 20
        if adx_val > 25:
            label = "TRENDING_UP" if ema_slope > 0 else "TRENDING_DOWN"
        elif 30 <= rsi_val <= 70:
            label = "MEAN_REVERTING"
        else:
            vol_slope = df["volume"].tail(20).mean() - df["volume"].tail(50).mean()
            label = "ACCUMULATION" if vol_slope > 0 else "DISTRIBUTION"
        return RegimeState(label, 0.60, h)

    @staticmethod
    def _regime_features(df: pd.DataFrame) -> pd.DataFrame:
        c, v = df["close"], df["volume"]
        ret = c.pct_change().dropna()
        atr = (df["high"] - df["low"]).rolling(14).mean()
        vol_ratio = v / v.rolling(20).mean().replace(0, 1e-9)
        feats = pd.DataFrame({
            "ret": ret,
            "vol": ret.rolling(20).std(),
            "vol_ratio": vol_ratio,
            "atr": atr,
        }).dropna()
        return feats.replace([np.inf, -np.inf], 0).fillna(0)

    @staticmethod
    def _map_state(idx: int, df: pd.DataFrame, hurst: float) -> str:
        c = df["close"]
        slope = c.ewm(span=50).mean().diff().iloc[-1]
        if hurst > 0.55:
            return "TRENDING_UP" if slope >= 0 else "TRENDING_DOWN"
        if hurst < 0.45:
            return "MEAN_REVERTING"
        vol_change = df["volume"].tail(20).mean() - df["volume"].tail(50).mean()
        return "ACCUMULATION" if vol_change >= 0 else "DISTRIBUTION"

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-9)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _adx_simple(df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus = (df["high"] - df["high"].shift(1)).clip(lower=0)
        minus = (df["low"].shift(1) - df["low"]).clip(lower=0)
        plus_di = 100 * plus.rolling(period).mean() / atr.replace(0, 1e-9)
        minus_di = 100 * minus.rolling(period).mean() / atr.replace(0, 1e-9)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
        return dx.rolling(period).mean()

    @staticmethod
    def _hurst(ts: np.ndarray, max_lag: int = 50) -> float:
        if len(ts) < 30:
            return 0.5
        lags = range(2, min(max_lag, len(ts) // 2))
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        tau = np.array(tau)
        tau[tau == 0] = 1e-9
        slope, _, _, _, _ = linregress(np.log(lags), np.log(tau))
        return max(0.0, min(1.0, slope * 2.0))


regime_detector = RegimeDetector()
