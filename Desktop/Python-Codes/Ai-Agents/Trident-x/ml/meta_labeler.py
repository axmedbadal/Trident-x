import asyncio
import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

FEATURE_COLS = ["rsi_14", "adx", "macd_hist", "atr_pct", "vol_sma20_ratio", "vwap_dev", "bb_pct_b", "roc_10"]

COLD_START_SIGNALS = 100
COLD_START_DEFAULT_TRENDING = 0.65
COLD_START_DEFAULT_OTHER = 0.60
MIN_TRAIN_SAMPLES = 50
MIN_AUC_THRESHOLD = 0.55


class MetaLabeler:
    def __init__(self):
        self.models: Dict[str, object] = {}
        self._last_train: Dict[str, float] = {}
        self.is_trained: Dict[str, bool] = {}
        self.signal_count: Dict[str, int] = {}
        self.trained = False

    def predict(self, symbol: str, features: Dict[str, float]) -> float:
        """Return P(win) for current signal. Uses cold start default until trained."""
        self.signal_count[symbol] = self.signal_count.get(symbol, 0) + 1
        count = self.signal_count[symbol]

        model = self.models.get(symbol)
        if model is not None and self.is_trained.get(symbol, False) and count >= COLD_START_SIGNALS:
            try:
                x = np.array([[features.get(c, 0.0) for c in FEATURE_COLS]])
                proba = model.predict_proba(x)[0]
                return float(proba[1])
            except Exception as e:
                logger.warning(f"MetaLabeler predict error: {e}")

        # Cold start: regime-aware default
        if count < COLD_START_SIGNALS:
            regime = features.get("regime", "MEAN_REVERTING")
            if regime in ("TRENDING_UP", "TRENDING_DOWN"):
                return COLD_START_DEFAULT_TRENDING
            return COLD_START_DEFAULT_OTHER
        return 0.55

    async def maybe_retrain(self, symbol: str, candles: List[Dict]):
        if len(candles) < 200 or (time.time() - self._last_train.get(symbol, 0)) < 7 * 86400:
            return
        try:
            await asyncio.to_thread(self._train, symbol, candles)
        except Exception as e:
            logger.warning(f"Meta labeler retrain failed {symbol}: {e}")

    def _train(self, symbol: str, candles: List[Dict]):
        df = pd.DataFrame(candles)
        if len(df) < 200:
            return
        df["returns"] = df["close"].pct_change().shift(-3)
        df["label"] = (df["returns"] > 0).astype(int)

        from features.engineer import FeatureEngineer
        fe = FeatureEngineer()
        fe.load_history(symbol, "5m", candles)
        X, y = [], []
        for i in range(50, len(candles) - 3):
            f = fe.compute(symbol, "5m")
            if f is None:
                continue
            X.append([f.get(c, 0.0) for c in FEATURE_COLS])
            y.append(df["label"].iloc[i])

        if len(X) < MIN_TRAIN_SAMPLES or not XGB_AVAILABLE:
            return

        split = int(len(X) * 0.7)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        model = XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            n_jobs=2, random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Validate AUC
        try:
            from sklearn.metrics import roc_auc_score
            val_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_proba)
            if auc < MIN_AUC_THRESHOLD:
                logger.warning(f"Meta-labeler AUC {auc:.3f} too low for {symbol}, staying cold start")
                return
            logger.info(f"Meta-labeler trained for {symbol}, AUC: {auc:.3f}")
        except Exception:
            logger.info(f"Meta-labeler trained for {symbol} (AUC check skipped)")

        self.models[symbol] = model
        self._last_train[symbol] = time.time()
        self.is_trained[symbol] = True
        self.trained = True


meta_labeler = MetaLabeler()
