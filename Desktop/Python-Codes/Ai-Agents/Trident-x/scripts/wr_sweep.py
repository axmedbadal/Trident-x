"""Win-rate sweep — optimized: pre-compute all features, then sweep via lookups."""
import sys, os, time, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from features.engineer import features
from strategies.price_action import price_action_engine
from strategies.scalping import scalping_engine

TP_ATR_LEVELS = [0.5, 0.75, 1.0, 1.5, 2.5]
SL_ATR_LEVELS = [1.0, 1.5]  # compare both
LOOK_FORWARD = 30
CONTEXT_CANDLES = 500
STEP = 3
SYMBOLS = ["SOLUSDT", "XRPUSDT", "ADAUSDT"]

def load_candles(symbol, tf, limit):
    conn = sqlite3.connect("trident_x.sqlite")
    try:
        rows = pd.read_sql(
            "SELECT timestamp,open,high,low,close,volume FROM candles "
            "WHERE symbol=? AND timeframe=? ORDER BY timestamp DESC LIMIT ?",
            conn, params=(symbol, tf, limit))
    finally:
        conn.close()
    rows = rows.iloc[::-1].reset_index(drop=True)
    return [dict(r) for _, r in rows.iterrows()]

def check_outcome(direction, entry, atr, sl_atr, tp_atr, future_bars):
    if atr <= 0:
        return "SKIP", 0.0
    sl_dist = sl_atr * atr
    tp_dist = tp_atr * atr
    for bar in future_bars:
        h, l = bar["high"], bar["low"]
        if direction == "BUY":
            if h >= entry + tp_dist: return "WIN", tp_atr / sl_atr
            if l <= entry - sl_dist: return "LOSS", -1.0
        else:
            if l <= entry - tp_dist: return "WIN", tp_atr / sl_atr
            if h >= entry + sl_dist: return "LOSS", -1.0
    return "TIMEOUT", 0.0

# ── Load data ──────────────────────────────────────────────
print("Loading data...")
for symbol in SYMBOLS:
    for tf in ["5m", "15m", "1h", "4h"]:
        features.buffers[f"{symbol}:{tf}"] = load_candles(symbol, tf, 100000)
        features.buffers[f"{symbol}:{tf}"].sort(key=lambda x: x["timestamp"])
    print(f"  {symbol}: {len(features.buffers[f'{symbol}:5m'])} bars")

# ── Pre-warm context TF cache ─────────────────────────────
print("Caching context TFs...")
for symbol in SYMBOLS:
    for tf in ["15m", "1h", "4h"]:
        features.compute(symbol, tf)
    print(f"  {symbol} done")

# ── Pre-compute 5m features for every window ──────────────
print("Pre-computing 5m features (the slow part)...")
feat_cache = {}  # (symbol, timestamp) -> features dict
original_bufs = {}  # save original full buffers before pre-computation overwrites them
for symbol in SYMBOLS:
    original_bufs[symbol] = features.buffers[f"{symbol}:5m"]
t0 = time.time()
total_windows = 0
for symbol in SYMBOLS:
    c5 = features.buffers[f"{symbol}:5m"]
    n = len(c5)
    count = 0
    for i in range(CONTEXT_CANDLES, n - LOOK_FORWARD, STEP):
        features.buffers[f"{symbol}:5m"] = c5[max(0, i - CONTEXT_CANDLES + 1): i + 1]
        f5 = features.compute(symbol, "5m")
        if f5:
            ts = c5[i]["timestamp"]
            atr_val = f5.get("atr_14", 0)
            feat_cache[(symbol, ts)] = (f5, atr_val)
            count += 1
    total_windows += count
    print(f"  {symbol}: {count} windows cached  ({time.time()-t0:.0f}s elapsed)")
print(f"Total windows: {total_windows}  ({time.time()-t0:.0f}s)")

# ── Sweep: fast — just look up pre-computed features ──────
print("\nSweeping signals...")
rows = []
t1 = time.time()

# Monkey-patch features.compute to return cached5m features (avoids 80ms recomputation)
_orig_compute = features.compute
def _cached_compute(symbol, tf):
    if tf == "5m":
        buf = features.buffers.get(f"{symbol}:5m")
        if buf:
            ts = buf[-1].get("timestamp", 0)
            hit = feat_cache.get((symbol, ts))
            if hit:
                return hit[0]
    return _orig_compute(symbol, tf)
features.compute = _cached_compute

for symbol in SYMBOLS:
    c5 = original_bufs[symbol]
    n = len(c5)
    fires = 0
    for i in range(CONTEXT_CANDLES, n - LOOK_FORWARD, STEP):
        price_action_engine._cooldown_until.clear()
        scalping_engine._cooldown_until.clear()
        ts = c5[i]["timestamp"]
        cached = feat_cache.get((symbol, ts))
        if not cached:
            continue
        f5, atr = cached
        if atr <= 0:
            continue
        # Set buffer for raw candle patterns the engine accesses directly
        features.buffers[f"{symbol}:5m"] = c5[max(0, i - CONTEXT_CANDLES + 1): i + 1]
        entry = c5[i]["close"]
        future = c5[i + 1: i + 1 + LOOK_FORWARD]
        for engine in (price_action_engine, scalping_engine):
            try:
                sig = engine.generate(symbol, f5, 0.0)
            except Exception as e:
                print(f"  ERR {engine.name} {symbol}@{ts}: {type(e).__name__}: {e}")
                continue
            if sig["direction"] in ("NEUTRAL", "WAIT"):
                continue
            fires += 1
            direction = sig["direction"]
            setup = (sig.get("rationale") or "").split("|")[0].split(",")[0].strip()
            for sl_atr in SL_ATR_LEVELS:
                for tp_atr in TP_ATR_LEVELS:
                    outcome, trade_r = check_outcome(direction, entry, atr, sl_atr, tp_atr, future)
                    rows.append(dict(
                        engine=engine.name, symbol=symbol, sl_atr=sl_atr,
                        tp_atr=tp_atr, tp_r=round(tp_atr / sl_atr, 3),
                        direction=direction, setup=setup,
                        outcome=outcome, trade_r=trade_r))
    print(f"  {symbol}: {fires} signals")
features.compute = _orig_compute
print(f"Sweep done in {time.time()-t1:.0f}s")

df = pd.DataFrame(rows)
df = df[df["outcome"] != "SKIP"]

# ── Results ────────────────────────────────────────────────
print("\n=== ENGINE x SL x TP (all symbols) ===")
g = df.groupby(["engine","sl_atr","tp_atr"]).agg(
    n=("outcome","size"), wins=("outcome", lambda x: (x=="WIN").sum()),
    losses=("outcome", lambda x: (x=="LOSS").sum()),
    timeouts=("outcome", lambda x: (x=="TIMEOUT").sum()),
    total_r=("trade_r","sum")).reset_index()
g["wr%"] = (g["wins"]/g["n"]*100).round(1)
g["be_wr%"] = (g["sl_atr"]/(g["sl_atr"]+g["tp_atr"])*100).round(1)
g["avg_r"] = (g["total_r"]/g["n"]).round(4)
g["edge_pp"] = (g["wr%"]-g["be_wr%"]).round(1)
print(g.to_string(index=False))

print("\n=== TOP SETUPS: SL=1.0 x TP=0.5 ===")
sub = df[(df["sl_atr"]==1.0) & (df["tp_atr"]==0.5) & (df["outcome"].isin(["WIN","LOSS"]))]
gs = sub.groupby(["engine","direction","setup"]).agg(
    n=("outcome","size"), wins=("outcome", lambda x: (x=="WIN").sum()),
    total_r=("trade_r","sum")).reset_index()
gs["wr%"] = (gs["wins"]/gs["n"]*100).round(1)
gs["be_wr%"] = round(1.0/(1.0+0.5)*100, 1)
gs["avg_r"] = (gs["total_r"]/gs["n"]).round(4)
gs["edge_pp"] = (gs["wr%"]-gs["be_wr%"]).round(1)
gs = gs[gs["n"]>=10].sort_values(["engine","edge_pp"], ascending=[True,False])
print(gs.head(20).to_string(index=False))

print("\n=== TOP SETUPS: SL=1.0 x TP=0.75 ===")
sub = df[(df["sl_atr"]==1.0) & (df["tp_atr"]==0.75) & (df["outcome"].isin(["WIN","LOSS"]))]
gs = sub.groupby(["engine","direction","setup"]).agg(
    n=("outcome","size"), wins=("outcome", lambda x: (x=="WIN").sum()),
    total_r=("trade_r","sum")).reset_index()
gs["wr%"] = (gs["wins"]/gs["n"]*100).round(1)
gs["be_wr%"] = round(1.0/(1.0+0.75)*100, 1)
gs["avg_r"] = (gs["total_r"]/gs["n"]).round(4)
gs["edge_pp"] = (gs["wr%"]-gs["be_wr%"]).round(1)
gs = gs[gs["n"]>=10].sort_values(["engine","edge_pp"], ascending=[True,False])
print(gs.head(20).to_string(index=False))

print("\n=== SYMBOL x ENGINE: SL=1.0 x TP=0.5 ===")
sub = df[(df["sl_atr"]==1.0) & (df["tp_atr"]==0.5) & (df["outcome"].isin(["WIN","LOSS"]))]
s = sub.groupby(["engine","symbol"]).agg(
    n=("outcome","size"), wins=("outcome", lambda x: (x=="WIN").sum()),
    total_r=("trade_r","sum")).reset_index()
s["wr%"] = (s["wins"]/s["n"]*100).round(1)
s["avg_r"] = (s["total_r"]/s["n"]).round(4)
print(s.to_string(index=False))

df.to_csv("wr_sweep_full.csv", index=False)
print("\nSaved wr_sweep_full.csv")
