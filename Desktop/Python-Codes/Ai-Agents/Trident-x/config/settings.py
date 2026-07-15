from pydantic_settings import BaseSettings
from typing import List, Dict


class TridentConfig(BaseSettings):
    PAIRS: List[str] = ["SOLUSDT", "XRPUSDT", "ADAUSDT"]
    PRIMARY_TF: str = "5m"
    CONTEXT_TFS: List[str] = ["15m", "1h", "4h"]

    INITIAL_EQUITY: float = 10000.0

    MAX_POSITIONS: int = 2
    MAX_RISK_PER_TRADE_PCT: float = 0.02
    MAX_POSITION_EQUITY_PCT: float = 0.25
    MAX_DAILY_LOSS_PCT: float = 0.05
    MAX_DD_PCT: float = 0.10

    KELLY_FRACTION: float = 0.25

    ASSET_PARAMS: Dict = {
        "SOLUSDT": {
            "eqh_tolerance": 0.005,
            "fvg_distance": 2.5,
            "sl_mult": 1.5,
            "tp_mult": 2.5,
            "volatility_threshold": 0.08,
        },
        "XRPUSDT": {
            "eqh_tolerance": 0.004,
            "fvg_distance": 2.0,
            "sl_mult": 1.5,
            "tp_mult": 2.0,
            "volatility_threshold": 0.06,
        },
        "ADAUSDT": {
            "eqh_tolerance": 0.003,
            "fvg_distance": 3.5,
            "sl_mult": 1.5,
            "tp_mult": 3.0,
            "volatility_threshold": 0.05,
        },
    }

    BINANCE_WS_URL: str = "wss://stream.binance.com:9443/ws"
    BINANCE_REST_URL: str = "https://api.binance.com"
    BINANCE_API_KEY: str = ""
    BINANCE_SECRET: str = ""

    PAPER_TRADING: bool = True
    SLIPPAGE_MODEL: str = "regime_adjusted"

    # Fix 3.4: Dashboard auth
    DASHBOARD_API_KEY: str = ""
    DASHBOARD_AUTH_ENABLED: bool = False

    LOG_LEVEL: str = "INFO"
    LOG_TO_FILE: bool = True
    LOG_FILE: str = "logs/trident_x.log"
    LOG_ROTATION_DAYS: int = 7

    REGIME_RETRAIN_DAYS: int = 7
    META_LABELER_RETRAIN_DAYS: int = 7
    HMM_TIMEOUT_SECONDS: int = 10

    FUNDING_RATE_THRESHOLD: float = 0.0001

    ENGINE_DISABLE_THRESHOLD: float = 0.68
    ENGINE_EVAL_MIN_SIGNALS: int = 30

    # Sniper strategy
    SNIPER_ENABLED: bool = True
    SNIPER_ENABLED_ASSETS: List[str] = ["SOLUSDT", "XRPUSDT", "ADAUSDT"]
    SNIPER_TFS: List[str] = ["5m", "15m", "1h", "4h"]
    SNIPER_ACTIVATION_THRESHOLD: float = 6.0
    SNIPER_COOLDOWN_SECONDS: int = 300
    SNIPER_STABILITY_WINDOW: int = 5
    SNIPER_ENTRY_MISS_PCT: float = 0.02
    SNIPER_USE_MTF_ALIGNMENT: bool = True

    class Config:
        env_file = ".env"


settings = TridentConfig()
