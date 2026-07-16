import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, maxlen: int = 500):
        self.buffers: Dict[str, List[Dict]] = {}
        self.maxlen = maxlen

    def update(self, symbol: str, tf: str, candle: Dict):
        key = f"{symbol}:{tf}"
        self.buffers.setdefault(key, [])
        self.buffers[key].append(candle)
        if len(self.buffers[key]) > self.maxlen:
            self.buffers[key].pop(0)

    def load_history(self, symbol: str, tf: str, candles: List[Dict]):
        key = f"{symbol}:{tf}"
        self.buffers[key] = list(candles[-self.maxlen:])

    def _df(self, symbol: str, tf: str) -> Optional[pd.DataFrame]:
        buf = self.buffers.get(f"{symbol}:{tf}")
        if not buf or len(buf) < 50:
            return None
        df = pd.DataFrame(buf)
        df["symbol"] = symbol
        df["timeframe"] = tf
        return df

    def compute(self, symbol: str, tf: str) -> Optional[Dict[str, float]]:
        df = self._df(symbol, tf)
        if df is None:
            return None
        try:
            feats = self._compute_features(df)
            return feats
        except Exception as e:
            logger.error(f"feature compute error {symbol} {tf}: {e}")
            return None

    def _compute_features(self, df: pd.DataFrame) -> Dict[str, float]:
        o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
        hl2 = (h + l) / 2
        last = {}

        def _set(name, series):
            val = series.iloc[-1] if hasattr(series, "iloc") else series
            last[name] = 0.0 if (pd.isna(val) or np.isinf(val)) else float(val)

        # Trend
        for period in [9, 20, 50, 200]:
            if len(c) >= period:
                _set(f"ema_{period}", c.ewm(span=period, adjust=False).mean())
                _set(f"sma_{period}", c.rolling(period).mean())
            else:
                last[f"ema_{period}"] = 0.0
                last[f"sma_{period}"] = 0.0
        smas = [last["sma_9"], last["sma_20"], last["sma_50"], last["sma_200"]]
        last["sma_aligned_bull"] = float(smas[0] > smas[1] > smas[2] > smas[3])
        last["sma_aligned_bear"] = float(smas[0] < smas[1] < smas[2] < smas[3])

        # Supertrend
        atr = self._atr(h, l, c, 14)
        upper = hl2 + 3 * atr
        lower = hl2 - 3 * atr
        _set("supertrend_upper", upper)
        _set("supertrend_lower", lower)
        last["supertrend"] = 1.0 if c.iloc[-1] > lower.iloc[-1] else -1.0

        # ADX
        plus_di, minus_di, adx = self._adx(h, l, c, 14)
        _set("adx", adx)
        _set("plus_di", plus_di)
        _set("minus_di", minus_di)

        # MACD
        macd_line = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        macd_sig = macd_line.ewm(span=9, adjust=False).mean()
        _set("macd", macd_line)
        _set("macd_signal", macd_sig)
        _set("macd_hist", macd_line - macd_sig)

        # TTM Squeeze
        bb_lower = c.rolling(20).mean() - 2 * c.rolling(20).std()
        kc_lower = c.rolling(20).mean() - 1.5 * self._atr(h, l, c, 20)
        _set("ttm_squeeze", (bb_lower > kc_lower).astype(int))

        # Momentum
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        _set("rsi_14", 100 - 100 / (1 + rs))

        low_min = l.rolling(14).min()
        high_max = h.rolling(14).max()
        k = 100 * (c - low_min) / (high_max - low_min).replace(0, 1e-9)
        d = k.rolling(3).mean()
        _set("stoch_k", k)
        _set("stoch_d", d)
        _set("williams_r", (high_max - c) / (high_max - low_min).replace(0, 1e-9) * -100)
        tp = (h + l + c) / 3
        ma_tp = tp.rolling(20).mean()
        _set("cci", (tp - ma_tp) / (0.015 * tp.rolling(20).std().replace(0, 1e-9)))
        mf = ((c - c.shift(1)) / c.shift(1).replace(0, 1e-9))
        _set("mfi", 100 - 100 / (1 + (v * mf).rolling(14).mean()))
        _set("roc_10", (c - c.shift(10)) / c.shift(10).replace(0, 1e-9))

        # Volatility
        _set("atr_14", atr)
        _set("atr_pct", atr / c.replace(0, 1e-9) * 100)
        bb_m = c.rolling(20).mean()
        bb_std = c.rolling(20).std().replace(0, 1e-9)
        _set("bb_width", (4 * bb_std) / bb_m.replace(0, 1e-9))
        _set("bb_pct_b", (c - (bb_m - 2 * bb_std)) / (4 * bb_std).replace(0, 1e-9))
        kc_m = c.ewm(span=20).mean()
        kc = 1.5 * self._atr(h, l, c, 20)
        last["bb_squeeze"] = float((bb_m - 2 * bb_std).iloc[-1] > (kc_m - kc).iloc[-1] and (bb_m + 2 * bb_std).iloc[-1] < (kc_m + kc).iloc[-1]) if len(bb_m) > 0 else 0.0
        _set("keltner_upper", c.ewm(span=20).mean() + 2 * self._atr(h, l, c, 20))
        _set("keltner_lower", c.ewm(span=20).mean() - 2 * self._atr(h, l, c, 20))
        _set("donchian_upper", h.rolling(20).max())
        _set("donchian_lower", l.rolling(20).min())

        # Volume
        vol_sma20 = v.rolling(20).mean().replace(0, 1e-9)
        _set("vol_sma20_ratio", v / vol_sma20)
        # Phase 1: 24h vs 7-day volume SMA (approximate using 5m candles)
        periods_24h = min(288, len(v))
        periods_7d = min(2016, len(v))
        if periods_24h > 0 and periods_7d > 0:
            vol_24h = v.iloc[-periods_24h:].sum()
            vol_7d_sma = v.iloc[-periods_7d:].mean() * periods_24h
            vol_7d_sma = float(vol_7d_sma) if not hasattr(vol_7d_sma, "replace") else vol_7d_sma.replace(0, 1e-9)
            _set("vol_24h_vs_7d", vol_24h / max(vol_7d_sma, 1e-9))
        else:
            last["vol_24h_vs_7d"] = 1.0
        obv = ((np.sign(c.diff()) * v).cumsum())
        _set("obv", obv)
        vwap = (v * (h + l + c) / 3).cumsum() / v.cumsum().replace(0, 1e-9)
        _set("vwap_dev", (c - vwap) / vwap.replace(0, 1e-9) * 100)
        _set("cvd", (np.sign(c.diff()) * v).cumsum())
        _set("volume_vah", (v * h).rolling(20).sum() / v.rolling(20).sum().replace(0, 1e-9))
        _set("volume_val", (v * l).rolling(20).sum() / v.rolling(20).sum().replace(0, 1e-9))

        # Phase 1: Z-score for mean reversion blocker
        sma20 = c.rolling(20).mean().replace(0, 1e-9)
        std20 = c.rolling(20).std().replace(0, 1e-9)
        _set("z_score_20", (c - sma20) / std20)

        # SMC: lagged swing highs/lows, FVG, displacement
        rh = h.rolling(20, center=False).max().shift(1)
        rl = l.rolling(20, center=False).min().shift(1)
        _set("swing_high_20", rh.iloc[-1] if len(rh) else 0.0)
        _set("swing_low_20", rl.iloc[-1] if len(rl) else 0.0)
        last["fvg_bull"] = float(l.iloc[-1] > h.iloc[-3] and l.iloc[-2] > h.iloc[-3])
        last["fvg_bear"] = float(h.iloc[-1] < l.iloc[-3] and h.iloc[-2] < l.iloc[-3])
        c3range = (h - l).rolling(3).sum()
        last["displacement"] = float(c3range.iloc[-1] > 3 * atr.iloc[-1]) if len(atr) else 0.0
        last["bos_up"] = float(c.iloc[-1] > rh.iloc[-1]) if len(rh) else 0.0
        last["bos_down"] = float(c.iloc[-1] < rl.iloc[-1]) if len(rl) else 0.0

        # Microstructure
        last["vpin"] = float((v.iloc[-1] if len(v) else 0.0) / max(vol_sma20.iloc[-1], 1e-9))
        last["spread_approx"] = float((h.iloc[-1] - l.iloc[-1]) / c.iloc[-1] * 100) if c.iloc[-1] else 0.0

        # Extra signals used by strategies
        last["close"] = float(c.iloc[-1])
        last["atr_expansion"] = float(atr.iloc[-1] > 2 * atr.shift(3).iloc[-1]) if len(atr) > 3 else 0.0
        body = (c - o).abs()
        prev_body = body.shift(1)
        last["engulfing"] = float(body.iloc[-1] > prev_body.iloc[-1] and (c.iloc[-1] - o.iloc[-1]) * (c.iloc[-2] - o.iloc[-2]) < 0) if len(body) > 1 else 0.0
        last["volume_at_node"] = float(abs(last.get("vwap_dev", 0.0)) < 0.5 and last.get("vol_sma20_ratio", 0.0) > 1.2)
        # RSI divergence: price makes new low but RSI makes higher low = bullish div
        rsi_series = 100 - 100 / (1 + rs)
        if len(rsi_series.dropna()) >= 20:
            price_low_20 = c.iloc[-20:].min()
            price_low_5 = c.iloc[-5:].min()
            rsi_low_20 = rsi_series.iloc[-20:].min()
            rsi_low_5 = rsi_series.iloc[-5:].min()
            last["rsi_divergence"] = float(
                (price_low_5 <= price_low_20 * 1.001) and (rsi_low_5 > rsi_low_20 + 2)
            ) if rsi_low_20 < 40 else 0.0
        else:
            last["rsi_divergence"] = 0.0
        # OBV divergence: price makes new high but OBV makes lower high = bearish div
        if len(obv.dropna()) >= 20:
            price_high_20 = c.iloc[-20:].max()
            price_high_5 = c.iloc[-5:].max()
            obv_high_20 = obv.iloc[-20:].max()
            obv_high_5 = obv.iloc[-5:].max()
            last["obv_divergence"] = float(
                (price_high_5 >= price_high_20 * 0.999) and (obv_high_5 < obv_high_20 * 0.98)
            ) if obv_high_20 > 0 else 0.0
        else:
            last["obv_divergence"] = 0.0
        # HTF agreement: check if higher timeframe features align with 5m direction
        htf_bull = 0
        sym = df["symbol"].iloc[-1] if "symbol" in df.columns else ""
        for htf_key in ["15m", "1h"]:
            htf_buf = self.buffers.get(f"{sym}:{htf_key}", [])
            if htf_buf and len(htf_buf) > 50:
                htf_f = htf_buf[-1]
                htf_rsi = htf_f.get("rsi_14", 50)
                htf_adx = htf_f.get("adx", 20)
                if htf_rsi > 55 and htf_adx > 25:
                    htf_bull += 1
                elif htf_rsi < 45 and htf_adx > 25:
                    htf_bull -= 1
        last["htf_agree"] = float(htf_bull) / 2.0
        # Clean
        clean = {}
        for k, val in last.items():
            if isinstance(val, (np.floating, np.integer)):
                val = float(val)
            if val is None or pd.isna(val) or np.isinf(val):
                val = 0.0
            clean[k] = float(val)
        return clean

    @staticmethod
    def _atr(high, low, close, period: int) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _adx(high, low, close, period: int):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        plus_dm = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)
        plus_dm[plus_dm <= minus_dm] = 0
        minus_dm[minus_dm <= plus_dm] = 0
        atr = tr.rolling(period).mean().replace(0, 1e-9)
        plus_di = 100 * plus_dm.rolling(period).mean() / atr
        minus_di = 100 * minus_dm.rolling(period).mean() / atr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
        adx = dx.rolling(period).mean()
        return plus_di, minus_di, adx


features = FeatureEngineer()
