import asyncio
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from config.settings import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (symbol, timeframe, timestamp)
);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('BUY','SELL')),
    entry_price REAL NOT NULL,
    size_usd REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN','CLOSED')),
    engine TEXT NOT NULL,
    opened_at INTEGER NOT NULL,
    closed_at INTEGER,
    pnl REAL,
    exit_reason TEXT,
    tp1_hit INTEGER DEFAULT 0,
    tp2_hit INTEGER DEFAULT 0,
    tp3_hit INTEGER DEFAULT 0,
    tp4_hit INTEGER DEFAULT 0,
    tp5_hit INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    fee REAL NOT NULL DEFAULT 0.0,
    slippage REAL NOT NULL DEFAULT 0.0,
    timestamp INTEGER NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('ENTRY','EXIT','PARTIAL')),
    FOREIGN KEY (position_id) REFERENCES positions(id)
);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL NOT NULL,
    strength TEXT NOT NULL,
    regime TEXT NOT NULL,
    engines_agreeing INTEGER NOT NULL DEFAULT 0,
    passed_gate INTEGER NOT NULL DEFAULT 0,
    gate_reason TEXT,
    executed INTEGER NOT NULL DEFAULT 0,
    timestamp INTEGER NOT NULL,
    meta_prob REAL
);
CREATE TABLE IF NOT EXISTS risk_state (
    timestamp INTEGER PRIMARY KEY,
    daily_pnl REAL NOT NULL DEFAULT 0.0,
    max_dd REAL NOT NULL DEFAULT 0.0,
    current_dd REAL NOT NULL DEFAULT 0.0,
    band TEXT NOT NULL DEFAULT 'GREEN',
    consecutive_losses INTEGER NOT NULL DEFAULT 0,
    cb_state TEXT NOT NULL DEFAULT 'GREEN'
);
CREATE TABLE IF NOT EXISTS engine_performance (
    engine TEXT NOT NULL,
    regime TEXT NOT NULL,
    total_signals INTEGER NOT NULL DEFAULT 0,
    passed_gate INTEGER NOT NULL DEFAULT 0,
    executed INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    win_rate REAL DEFAULT 0.0,
    avg_pnl REAL DEFAULT 0.0,
    disabled INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (engine, regime)
);
CREATE TABLE IF NOT EXISTS regime_history (
    timestamp INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    regime TEXT NOT NULL,
    confidence REAL NOT NULL,
    hurst REAL,
    PRIMARY KEY (timestamp, symbol)
);
CREATE TABLE IF NOT EXISTS system_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    level TEXT NOT NULL,
    module TEXT NOT NULL,
    message TEXT NOT NULL,
    correlation_id TEXT
);
CREATE TABLE IF NOT EXISTS funding_rates (
    symbol TEXT NOT NULL,
    rate REAL NOT NULL,
    predicted_rate REAL DEFAULT 0.0,
    timestamp INTEGER NOT NULL,
    PRIMARY KEY (symbol, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf_ts ON candles(symbol, timeframe, timestamp);
"""


class StateManager:
    def __init__(self, db_path: str = "trident_x.sqlite"):
        self.db_path = db_path
        self._mem_conn = None
        if db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.executescript(SCHEMA)
            self._mem_conn.commit()
        self._init_db()
        self._migrate()

    def _connection(self):
        if self._mem_conn is not None:
            return self._mem_conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        with self._connection() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _migrate(self):
        """Add columns for individual engine votes and signal_id tracking."""
        migrations = [
            "ALTER TABLE signals ADD COLUMN sniper_vote TEXT DEFAULT 'NEUTRAL'",
            "ALTER TABLE signals ADD COLUMN smc_vote TEXT DEFAULT 'NEUTRAL'",
            "ALTER TABLE signals ADD COLUMN momentum_vote TEXT DEFAULT 'NEUTRAL'",
            "ALTER TABLE signals ADD COLUMN mean_reversion_vote TEXT DEFAULT 'NEUTRAL'",
            "ALTER TABLE signals ADD COLUMN price_action_vote TEXT DEFAULT 'NEUTRAL'",
            "ALTER TABLE signals ADD COLUMN scalping_vote TEXT DEFAULT 'NEUTRAL'",
            "ALTER TABLE positions ADD COLUMN signal_id INTEGER DEFAULT NULL",
        ]
        with self._connection() as conn:
            for sql in migrations:
                try:
                    conn.execute(sql)
                    conn.commit()
                except sqlite3.OperationalError:
                    pass  # column already exists

    def save_candles(self, candles: List[Dict[str, Any]]):
        rows = [
            (c["symbol"], c["timeframe"], c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"])
            for c in candles
        ]
        with self._connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?,?)", rows
            )
            conn.commit()

    def load_candles(self, symbol: str, timeframe: str, limit: int = 500) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM candles WHERE symbol=? AND timeframe=? ORDER BY timestamp DESC LIMIT ?",
                (symbol, timeframe, limit),
            )
            rows = cur.fetchall()
        return [dict(r) for r in reversed(rows)]

    def insert_signal(self, signal: Dict[str, Any]) -> int:
        with self._connection() as conn:
            cur = conn.execute(
                "INSERT INTO signals(symbol,direction,confidence,strength,regime,engines_agreeing,passed_gate,gate_reason,executed,timestamp,meta_prob,sniper_vote,smc_vote,momentum_vote,mean_reversion_vote,price_action_vote,scalping_vote) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    signal["symbol"], signal["direction"], signal["confidence"], signal["strength"],
                    signal["regime"], signal["engines_agreeing"], int(signal["passed_gate"]),
                    signal.get("gate_reason"), int(signal["executed"]), signal["timestamp"], signal.get("meta_prob"),
                    signal.get("sniper_vote", "NEUTRAL"), signal.get("smc_vote", "NEUTRAL"),
                    signal.get("momentum_vote", "NEUTRAL"), signal.get("mean_reversion_vote", "NEUTRAL"),
                    signal.get("price_action_vote", "NEUTRAL"), signal.get("scalping_vote", "NEUTRAL"),
                ),
            )
            conn.commit()
            return cur.lastrowid

    def get_recent_signals(self, symbol: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            if symbol:
                cur = conn.execute(
                    "SELECT * FROM signals WHERE symbol=? ORDER BY timestamp DESC LIMIT ?",
                    (symbol, limit),
                )
            else:
                cur = conn.execute(
                    "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", (limit,)
                )
            return [dict(r) for r in cur.fetchall()]

    def get_open_positions(self) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM positions WHERE status='OPEN'")
            return [dict(r) for r in cur.fetchall()]

    def save_funding_rate(self, symbol: str, rate: float, predicted_rate: float = 0.0):
        ts = int(time.time() * 1000)
        with self._connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO funding_rates(symbol,rate,predicted_rate,timestamp) VALUES (?,?,?,?)",
                (symbol, rate, predicted_rate, ts),
            )
            conn.commit()

    def get_latest_funding_rate(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM funding_rates WHERE symbol=? ORDER BY timestamp DESC LIMIT 1", (symbol,)
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def save_position(self, position: Dict[str, Any]) -> int:
        with self._connection() as conn:
            if position.get("id"):
                conn.execute(
                    """UPDATE positions SET status=?,closed_at=?,pnl=?,exit_reason=?,
                       tp1_hit=?,tp2_hit=?,tp3_hit=?,tp4_hit=?,tp5_hit=? WHERE id=?""",
                    (
                        position["status"], position.get("closed_at"), position.get("pnl"),
                        position.get("exit_reason"), int(position.get("tp1_hit", 0)),
                        int(position.get("tp2_hit", 0)), int(position.get("tp3_hit", 0)),
                        int(position.get("tp4_hit", 0)), int(position.get("tp5_hit", 0)),
                        position["id"],
                    ),
                )
            else:
                cur = conn.execute(
                    "INSERT INTO positions(symbol,direction,entry_price,size_usd,stop_loss,take_profit,status,engine,opened_at,signal_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        position["symbol"], position["direction"], position["entry_price"],
                        position["size_usd"], position["stop_loss"], position["take_profit"],
                        position.get("status", "OPEN"), position.get("engine", "consensus"),
                        position["opened_at"], position.get("signal_id"),
                    ),
                )
                position["id"] = cur.lastrowid
            conn.commit()
            return position["id"]

    def insert_trade(self, trade: Dict[str, Any]):
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO trades(position_id,symbol,direction,price,quantity,fee,slippage,timestamp,type) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    trade["position_id"], trade["symbol"], trade["direction"], trade["price"],
                    trade["quantity"], trade.get("fee", 0.0), trade.get("slippage", 0.0),
                    trade["timestamp"], trade["type"],
                ),
            )
            conn.commit()

    def save_risk_state(self, state: Dict[str, Any]):
        state = {k: (0.0 if v is None else v) for k, v in state.items()}
        with self._connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO risk_state(timestamp,daily_pnl,max_dd,current_dd,band,consecutive_losses,cb_state) VALUES (?,?,?,?,?,?,?)",
                (
                    state["timestamp"], state["daily_pnl"], state["max_dd"], state["current_dd"],
                    state["band"], state["consecutive_losses"], state["cb_state"],
                ),
            )
            conn.commit()

    def load_risk_state(self) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM risk_state ORDER BY timestamp DESC LIMIT 1")
            row = cur.fetchone()
        return dict(row) if row else None

    def update_engine_performance(self, engine: str, regime: str, metrics: Dict[str, Any]):
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO engine_performance(engine,regime,total_signals,passed_gate,executed,
                    wins,losses,win_rate,avg_pnl,disabled)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(engine,regime) DO UPDATE SET
                    total_signals=excluded.total_signals,
                    passed_gate=excluded.passed_gate,
                    executed=excluded.executed,
                    wins=excluded.wins,
                    losses=excluded.losses,
                    win_rate=excluded.win_rate,
                    avg_pnl=excluded.avg_pnl,
                    disabled=excluded.disabled""",
                (
                    engine, regime, metrics["total_signals"], metrics["passed_gate"], metrics["executed"],
                    metrics["wins"], metrics["losses"], metrics["win_rate"], metrics["avg_pnl"],
                    int(metrics["disabled"]),
                ),
            )
            conn.commit()

    def get_signal_by_id(self, signal_id: int) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM signals WHERE id=?", (signal_id,))
            row = cur.fetchone()
        return dict(row) if row else None

    def record_engine_pnl(self, engine: str, regime: str, pnl: float):
        """Called after position close to attribute P&L to engine+regime."""
        perf = self.get_engine_performance(engine, regime)
        if perf is None:
            perf = {"total_signals": 0, "passed_gate": 0, "executed": 0,
                    "wins": 0, "losses": 0, "win_rate": 0.0, "avg_pnl": 0.0, "disabled": 0}
        perf["executed"] = perf.get("executed", 0) + 1
        if pnl > 0:
            perf["wins"] = perf.get("wins", 0) + 1
        else:
            perf["losses"] = perf.get("losses", 0) + 1
        total_decisions = perf["wins"] + perf["losses"]
        perf["win_rate"] = perf["wins"] / max(total_decisions, 1)
        # Rolling avg PnL
        prev_avg = perf.get("avg_pnl", 0.0)
        prev_count = total_decisions - 1
        perf["avg_pnl"] = (prev_avg * prev_count + pnl) / total_decisions if total_decisions > 1 else pnl
        self.update_engine_performance(engine, regime, perf)

    def get_engine_performance(self, engine: str, regime: str) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM engine_performance WHERE engine=? AND regime=?", (engine, regime)
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def insert_regime_change(self, ts: int, symbol: str, regime: str, confidence: float, hurst: Optional[float]):
        with self._connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO regime_history(timestamp,symbol,regime,confidence,hurst) VALUES (?,?,?,?,?)",
                (ts, symbol, regime, confidence, hurst),
            )
            conn.commit()

    def batch_log(self, entries: List[Dict[str, Any]]):
        rows = [
            (e["timestamp"], e["level"], e["module"], e["message"], e.get("correlation_id"))
            for e in entries
        ]
        with self._connection() as conn:
            conn.executemany(
                "INSERT INTO system_log(timestamp,level,module,message,correlation_id) VALUES (?,?,?,?,?)", rows
            )
            conn.commit()

    def prune(self):
        cutoff_candles = int(time.time() * 1000) - 90 * 24 * 60 * 60 * 1000
        cutoff_log = int(time.time()) - settings.LOG_ROTATION_DAYS * 86400
        with self._connection() as conn:
            conn.execute("DELETE FROM candles WHERE timestamp < ?", (cutoff_candles,))
            conn.execute("DELETE FROM system_log WHERE timestamp < ?", (cutoff_log,))
            conn.commit()
            conn.execute("VACUUM")
            conn.commit()


state_manager = StateManager()
