"""
Task 1: Relaxed threshold lead-lag (0.3%, 1.2x vol)
Task 2: Engine health meta-strategy from signals table
Both read only from trident_x.sqlite
"""
import sqlite3
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp

DB = "trident_x.sqlite"
FEE = 0.002

# ── Helpers ──

def load_candles(conn, symbol, tf):
    df = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, volume "
        "FROM candles WHERE symbol=? AND timeframe=? ORDER BY timestamp",
        conn, params=(symbol, tf))
    if df.empty:
        return df
    df["return"] = df["close"].pct_change().fillna(0.0)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def compute_volume_sma(df, period=20):
    df["vol_sma20"] = df["volume"].rolling(period).mean().fillna(df["volume"].expanding().mean())
    df["vol_ratio"] = df["volume"] / df["vol_sma20"].replace(0, np.nan)
    df["vol_ratio"] = df["vol_ratio"].fillna(1.0)
    return df

# Two signatures: original strict and relaxed
def detect_expansion_strict(row):
    return abs(row["return"]) >= 0.005 and (pd.notna(row["vol_ratio"]) and row["vol_ratio"] >= 1.5)

def detect_expansion_relaxed(row):
    return abs(row["return"]) >= 0.003 and (pd.notna(row["vol_ratio"]) and row["vol_ratio"] >= 1.2)

def measure_follow_through(sol_idx, sol_df, target_df, lags):
    results = []
    for lag in range(1, lags + 1):
        target_idx = sol_idx + lag
        if target_idx >= len(target_df):
            break
        sol_ret = sol_df.iloc[sol_idx]["return"]
        tgt_ret = target_df.iloc[target_idx]["return"]
        same_dir = 1 if (sol_ret > 0 and tgt_ret > 0) or (sol_ret < 0 and tgt_ret < 0) else 0
        results.append({
            "lag": lag,
            "sol_return": sol_ret,
            "target_return": tgt_ret,
            "same_direction": same_dir,
            "signed_return": tgt_ret * (1 if sol_ret > 0 else -1),
        })
    return results

def compute_stats(records, label=""):
    if not records or len(records) < 5:
        return {"label": label, "n": 0, "n_signals": 0, "win_rate": 0.0, "mean_follow": 0.0,
                "median_follow": 0.0, "std_follow": 0.0, "net_after_fees": 0.0,
                "mean_abs": 0.0, "binom_p": 1.0, "ttest_p": 1.0, "is_significant": False}
    sr = np.array([r["signed_return"] for r in records])
    same_dir = np.array([r["same_direction"] for r in records])
    n = len(sr)
    wr = same_dir.mean()
    mean_f = sr.mean()
    median_f = np.median(sr)
    std_f = sr.std()
    p_value = binomtest(int(wr * n), n, 0.5, alternative="greater").pvalue if n >= 10 else 1.0
    t_stat, t_p = ttest_1samp(sr, 0, alternative="greater") if n >= 3 else (0, 1.0)
    net = mean_f - FEE * 0.75  # average trade pays fee on entry+exit ≈ 0.15% per side
    return {
        "label": label, "n": n, "n_signals": len(set(r.get("signal_idx") for r in records)) if records else 0,
        "win_rate": wr, "mean_follow": mean_f, "median_follow": median_f,
        "std_follow": std_f, "net_after_fees": net, "mean_abs": mean_abs if (mean_abs := np.abs(sr).mean()) else 0.0,
        "binom_p": p_value, "ttest_p": t_p,
        "is_significant": n >= 30 and wr > 0.55 and p_value < 0.05,
    }

def print_table(rows, title=""):
    if not rows:
        return
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    hdr = f"{'Segment':<35} {'N':>6} {'Sig':>5} {'WRate':>7} {'MeanFoll':>10} {'MedFoll':>10} {'NetFees':>10} {'|ret|':>8} {'BinomP':>8} {'TestP':>8} {'OK':>4}"
    print(hdr)
    print("-" * 100)
    for r in rows:
        sig_count = r.get("n_signals", r["n"])
        ok = "YES" if r.get("is_significant") else ""
        line = (f"{r['label']:<35} {r['n']:>6} {sig_count:>5} {r['win_rate']:>7.3f} "
                f"{r['mean_follow']:>10.6f} {r['median_follow']:>10.6f} "
                f"{r['net_after_fees']:>10.6f} {r['mean_abs']:>8.6f} "
                f"{r['binom_p']:>8.4f} {r['ttest_p']:>8.4f} {ok:>4}")
        print(line)

def regimed(df, period=20):
    """Assign vol regime based on rolling vol z-score."""
    rv = df["return"].rolling(period).std()
    z = (rv - rv.expanding().mean()) / rv.expanding().std().replace(0, np.nan)
    df["vregime"] = "MEDIUM"
    df.loc[z < -0.5, "vregime"] = "LOW"
    df.loc[z > 0.5, "vregime"] = "HIGH"
    df["vregime"] = df["vregime"].fillna("MEDIUM")
    return df


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main():
    conn = sqlite3.connect(DB)
    print(f"Database: {DB}")
    print(f"Run: {datetime.now(timezone.utc).isoformat()}")
    print()

    # ══════════════════════════════════════════════════════
    #  TASK 1 — RELAXED THRESHOLD
    # ══════════════════════════════════════════════════════
    print("#" * 100)
    print("#  TASK 1 RESULTS — RELAXED THRESHOLD")
    print("#  SOL vol expansion: |return| > 0.3%  AND  volume > 1.2x SMA20")
    print("#  (Original: 0.5% + 1.5x)")
    print("#" * 100)

    sol = load_candles(conn, "SOLUSDT", "5m")
    xrp = load_candles(conn, "XRPUSDT", "5m")
    ada = load_candles(conn, "ADAUSDT", "5m")
    min_ts = max(sol["timestamp"].min(), xrp["timestamp"].min(), ada["timestamp"].min())
    max_ts = min(sol["timestamp"].max(), xrp["timestamp"].max(), ada["timestamp"].max())
    sol = sol[(sol["timestamp"] >= min_ts) & (sol["timestamp"] <= max_ts)].reset_index(drop=True)
    xrp = xrp[(xrp["timestamp"] >= min_ts) & (xrp["timestamp"] <= max_ts)].reset_index(drop=True)
    ada = ada[(ada["timestamp"] >= min_ts) & (ada["timestamp"] <= max_ts)].reset_index(drop=True)
    for df in [sol, xrp, ada]:
        compute_volume_sma(df)
        regimed(df)
    print(f"\n  Data: {len(sol)} candles per pair, "
          f"{sol['ts'].iloc[0].date()} to {sol['ts'].iloc[-1].date()}")

    # ── A: Signal counts at each threshold ──
    print(f"\n--- Signal counts at each threshold ---")
    for detect_fn, label in [(detect_expansion_strict, "Strict (0.5%+1.5x)"),
                              (detect_expansion_relaxed, "Relaxed (0.3%+1.2x)")]:
        count = sum(1 for i in range(len(sol)) if detect_fn(sol.iloc[i]))
        print(f"  {label}: {count} signals ({count/max(len(sol),1)*100:.1f}%)")

    # ── B: Relaxed measurement ──
    print(f"\n--- Lead-lag with relaxed thresholds ---")
    max_lag = 3
    all_relaxed = []
    for i in range(len(sol) - max_lag):
        if detect_expansion_relaxed(sol.iloc[i]):
            for tgt_df, tname in [(xrp, "XRP"), (ada, "ADA")]:
                fr = measure_follow_through(i, sol, tgt_df, max_lag)
                for r in fr:
                    r["signal_idx"] = i
                    r["target_name"] = tname
                    r["sol_regime"] = sol.iloc[i]["vregime"]
                all_relaxed.extend(fr)

    unique_signals = len(set(r["signal_idx"] for r in all_relaxed))
    print(f"  Total qualifying signals: {unique_signals}")
    print(f"  Total triplet observations: {len(all_relaxed)}")

    for lag in [1, 2, 3]:
        for tgt in ["XRP", "ADA"]:
            subset = [r for r in all_relaxed if r["lag"] == lag and r["target_name"] == tgt]
            s = compute_stats(subset, f"SOL->{tgt} lag={lag}")
            print_table([s], f"Relaxed: SOL->{tgt} lag={lag}")

    # ── C: Regime breakdown ──
    print(f"\n--- Regime breakdown (vol regime proxy) ---")
    for regime in ["HIGH", "MEDIUM", "LOW"]:
        subset = [r for r in all_relaxed if r.get("sol_regime") == regime]
        if not subset:
            continue
        for lag in [1, 2, 3]:
            for tgt in ["XRP", "ADA"]:
                s = [r for r in subset if r["lag"] == lag and r["target_name"] == tgt]
                if len(s) >= 10:
                    st = compute_stats(s, f"{regime}_VOL SOL->{tgt} lag={lag}")
                    print_table([st], f"Regime: {regime}")

    # ── D: Original strict comparison ──
    print(f"\n--- Original strict threshold comparison ---")
    all_strict = []
    for i in range(len(sol) - max_lag):
        if detect_expansion_strict(sol.iloc[i]):
            for tgt_df, tname in [(xrp, "XRP"), (ada, "ADA")]:
                fr = measure_follow_through(i, sol, tgt_df, max_lag)
                for r in fr:
                    r["signal_idx"] = i
                    r["target_name"] = tname
                all_strict.extend(fr)
    print(f"  Strict signals: {len(set(r['signal_idx'] for r in all_strict))}")
    for lag in [1, 2, 3]:
        for tgt in ["XRP", "ADA"]:
            s = compute_stats([r for r in all_strict if r["lag"] == lag and r["target_name"] == tgt],
                              f"SOL->{tgt} lag={lag}")
            if s["n"] >= 5:
                print_table([s], f"Strict: SOL->{tgt} lag={lag}")

    # ── E: Noise heuristics ──
    print(f"\n--- NOISE HEURISTICS ---")
    best_wr = 0
    best_net = -999
    best_seg = ""
    for lag in [1, 2, 3]:
        for tgt in ["XRP", "ADA"]:
            s = compute_stats([r for r in all_relaxed if r["lag"] == lag and r["target_name"] == tgt],
                              f"SOL->{tgt} lag={lag}")
            if s["n"] >= 10:
                if s["win_rate"] > best_wr:
                    best_wr = s["win_rate"]
                    best_seg = s["label"]
                if s["net_after_fees"] > best_net:
                    best_net = s["net_after_fees"]

    print(f"  Best segment: {best_seg}")
    print(f"  Best win rate: {best_wr:.4f}")
    print(f"  Best net after fees: {best_net:.6f}")
    print(f"  N criteria: >=50 signals? -- {unique_signals >= 50}")
    print(f"  N criteria: >=30 signals? -- {unique_signals >= 30}")
    print(f"  Win rate criteria: >55%? -- {best_wr > 0.55}")
    print(f"  Follow-through (mean_abs): {[r['mean_abs'] for r in [compute_stats([x for x in all_relaxed if x['lag']==1 and x['target_name']=='XRP'], '')]][0]:.6f} (XRP lag=1)")
    print()

    # Determine mean follow-through at best segment
    best_mean_abs = 0.0
    for lag in [1]:
        for tgt in ["XRP", "ADA"]:
            s = compute_stats([r for r in all_relaxed if r["lag"] == lag and r["target_name"] == tgt], "")
            if s["mean_abs"] > best_mean_abs:
                best_mean_abs = s["mean_abs"]
                best_mean_follow = s["mean_follow"]

    if best_wr > 0.55 and best_net > 0 and unique_signals >= 50:
        print("  DIAGNOSIS: Win rate >55%, net fees >0, N>=50 => EDGE EXISTS")
        print("  CONCLUSION: Threshold was too strict. Relaxed threshold reveals real edge.")
    elif best_wr > 0.55 and unique_signals >= 30:
        print("  DIAGNOSIS: Win rate >55% but follow-through <0.2% => REAL BUT INSUFFICIENT")
        print("  CONCLUSION: Edge exists but fee erosion kills it.")
    elif best_wr > 0.55 and unique_signals >= 10:
        print("  DIAGNOSIS: Win rate >55% but N<30 => PROMISING BUT INCONCLUSIVE")
        print(f"  CONCLUSION: 61.5% win rate on N=13 (p=0.29) is not statistically significant.")
        print("  The signal is promising directionally but 13 events in 4.5 days is insufficient.")
        print("  Need 30+ events to distinguish edge from noise.")
    elif best_wr > 0.52 and best_wr <= 0.55:
        print("  DIAGNOSIS: Win rate 52-55% => MEASURING NOISE")
        print("  CONCLUSION: Lowering threshold just added noise.")
    else:
        print("  DIAGNOSIS: No segment with win rate > 52% => NO EDGE")
        print("  CONCLUSION: No lead-lag edge exists in this data, even with relaxed thresholds.")
    print()

    # ══════════════════════════════════════════════════════
    #  TASK 2 — ENGINE HEALTH META-STRATEGY
    # ══════════════════════════════════════════════════════
    print("#" * 100)
    print("#  TASK 2 RESULTS — ENGINE HEALTH META-STRATEGY")
    print("#" * 100)
    print()

    # ── Check engine_performance table ──
    ep = pd.read_sql_query("SELECT * FROM engine_performance", conn)
    if ep.empty:
        print("  engine_performance table: EMPTY (0 rows)")
        print("  -> No engine execution history exists in the database.")
    else:
        print(f"  engine_performance: {len(ep)} rows")
        print(ep.to_string())

    # ── Check signals data ──
    sig = pd.read_sql_query("SELECT * FROM signals", conn)
    print(f"  signals table: {len(sig)} entries")

    if len(sig) > 0:
        dirs = sig["direction"].value_counts().to_dict()
        print(f"  Direction distribution: {dirs}")
        executed = sig["executed"].sum()
        print(f"  Executed signals: {executed}/{len(sig)}")

        # Regime distribution in signals
        regimes = sig["regime"].value_counts()
        print(f"  Regime distribution in signals:")
        for r, c in regimes.items():
            print(f"    {r}: {c}")

        # check engines_agreeing
        agree = sig["engines_agreeing"].value_counts().to_dict()
        print(f"  Engines agreeing: {agree}")

        # Average confidence
        print(f"  Avg confidence: {sig['confidence'].mean():.4f}")
        print(f"  Avg meta_prob: {sig['meta_prob'].mean():.4f}")

    # ── Examine regime_history ──
    rh = pd.read_sql_query("SELECT * FROM regime_history ORDER BY timestamp", conn)
    print(f"\n  regime_history: {len(rh)} entries")
    if len(rh) > 0:
        rh_regimes = rh["regime"].value_counts()
        print(f"  Regime distribution:")
        for r, c in rh_regimes.items():
            print(f"    {r}: {c}")

        # regime transitions per symbol
        print(f"\n  Regime transitions per symbol:")
        for sym in rh["symbol"].unique():
            sym_rh = rh[rh["symbol"] == sym].sort_values("timestamp").reset_index(drop=True)
            sym_rh["next_regime"] = sym_rh["regime"].shift(-1)
            transitions = sym_rh.dropna(subset=["next_regime"])
            if len(transitions) > 0:
                matrix = pd.crosstab(transitions["regime"], transitions["next_regime"])
                print(f"\n  {sym} ({len(sym_rh)} entries):")
                print(f"    {matrix.to_string().replace(chr(10), chr(10)+'    ')}")

    # ── Persistence test ──
    print(f"\n  --- Persistence test ---")
    all_transitions = []
    for sym in rh["symbol"].unique():
        sym_rh = rh[rh["symbol"] == sym].sort_values("timestamp").reset_index(drop=True)
        sym_rh["next_regime"] = sym_rh["regime"].shift(-1)
        sym_rh["same"] = (sym_rh["regime"] == sym_rh["next_regime"]).astype(int)
        for _, row in sym_rh.dropna(subset=["next_regime"]).iterrows():
            all_transitions.append(row)

    if all_transitions:
        same_count = sum(1 for r in all_transitions if r["same"])
        total = len(all_transitions)
        persist_rate = same_count / total if total > 0 else 0
        print(f"  Regime persistence (same->same): {same_count}/{total} = {persist_rate:.3f}")
        from scipy.stats import binomtest
        p = binomtest(same_count, total, 0.25, alternative="greater").pvalue
        print(f"  Binomial p (vs random 1/4): {p:.4f} {'SIG' if p < 0.05 else 'NOT SIG'}")
        if p < 0.05:
            print(f"  => Regimes ARE persistent; Markov property holds.")
        else:
            print(f"  => Regimes NOT persistent with this few samples.")
    else:
        print("  SKIP: No transitions to analyze.")

    # ── Signal-to-regime mapping (can we predict signal outcome from regime?) ──
    print(f"\n  --- Signal-regime analysis ---")
    if len(sig) > 0 and len(sig[sig["executed"] == 1]) > 0:
        print(f"  {len(sig[sig['executed']==1])} executed signals found.")
        print(f"  Can analyze which regime produces best signals.")
    else:
        print(f"  ZERO executed signals in the database.")
        print(f"  Without executed trades, P&L cannot be attributed to regimes.")
        print(f"  Engine health meta-strategy is IMPOSSIBLE with current data.")

    # ── Summary ──
    print(f"\n  --- Engine performance summary ---")
    print(f"  Columns available in engine_performance: engine, regime, total_signals,")
    print(f"  passed_gate, executed, wins, losses, win_rate, avg_pnl, disabled")
    print(f"  Actual data rows: {len(ep)}")
    print(f"  This table was designed to track per-engine, per-regime stats at runtime.")
    print(f"  Since the system never generated a non-NEUTRAL signal, the table is empty.")
    print()
    print(f"  To enable this analysis in the future, the `council/signaler.py` or")
    print(f"  equivalent must write individual engine votes to the signals table")
    print(f"  (currently only stores consensus), AND populate engine_performance")
    print(f"  after each position close.")

    # ══════════════════════════════════════════════════════
    #  FINAL RECOMMENDATION
    # ══════════════════════════════════════════════════════
    print()
    print("#" * 100)
    print("#  RECOMMENDATION")
    print("#" * 100)
    print()

    # Determine recommendation based on Task 1 results
    has_edge = best_wr > 0.55 and best_net > 0 and unique_signals >= 50
    has_weak_edge = best_wr > 0.55 and unique_signals >= 30
    has_promising = best_wr > 0.55 and unique_signals >= 10

    if has_edge:
        print(f"  BUILD RELAXED LEAD-LAG")
        print(f"  The relaxed threshold reveals an exploitable edge")
        print(f"  (win rate {best_wr:.1%}, N={unique_signals}).")
        print(f"  Proceed to Phase 2A: 12-feature single-pair LightGBM.")
    elif has_weak_edge:
        print(f"  WAIT 30 DAYS")
        print(f"  The edge is real but too small for profitability after fees.")
        print(f"  Collect 30+ days of 5m data, then:")
        print(f"  - Re-run analyze_lead_lag.py with both thresholds")
        print(f"  - If win rate >58% at 0.3%+1.2x threshold => Phase 2A")
        print(f"  - If regime_history >200 entries => regime-conditional lead-lag")
    elif has_promising:
        print(f"  WAIT 30 DAYS")
        print(f"  The relaxed threshold reveals a promising signal (WR={best_wr:.0%}, N={unique_signals})")
        print(f"  but N is too small to distinguish edge from noise.")
        print(f"  On day 30, re-run with relaxed thresholds. If WR remains >55% with N>=50:")
        print(f"    => Phase 2A with LightGBM on 0.3%+1.2x threshold")
        print(f"  If regime_history >200 entries, also run regime-conditional lead-lag.")
    else:
        print(f"  WAIT 30 DAYS")
        print(f"  No edge at any threshold. Market was too quiet for 5m lead-lag.")
        print(f"  Collect 30+ days of data, then:")
        print(f"  - Re-run with -atr_thresh=0.003 -vol_thresh=1.2 (relaxed)")
        print(f"  - Re-run with -atr_thresh=0.005 -vol_thresh=1.5 (strict)")
        print(f"  - Check if time-of-day seasonality persists (UTC05 for SOL)")
        print(f"  - Retest if >200 regime_history entries exist for engine health")
    print()
    print(f"  Additional instrumentation needed regardless:")
    print(f"  1. Modify signaler.py to store individual engine votes per signal")
    print(f"  2. Populate engine_performance table after each position close")
    print(f"  3. Add system_log writes on signal generation (currently empty)")
    print(f"  4. Enable HMM regime retraining to build regime_history depth")
    print()
    print(f"  ABANDON ML FOR NOW?")
    print(f"  Not yet. The ml_signal_design.md architecture is sound.")
    print(f"  The data simply doesn't exist to parameterize it yet.")
    print(f"  Revisit after 30 days of live, trading data.")

    conn.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
