"""Tier 1 backtest CLI.

Run the pipeline over a short window (default = all available, typically ~4-5 days):

    python -m backtest.run --days 5                # last 5 days, real signals only
    python -m backtest.run --smoke --days 3         # software-path test only; never performance evidence
    python -m backtest.run --smoke --symbols SOLUSDT --initial 20000

`--smoke` deliberately synthesizes entries and its output must not be used for
strategy selection, performance reporting, or any weight adjustment.

Writes a JSON report to backtest/run.json (or --out).
"""
import argparse
import json
import sys
import time
import logging

sys.path.insert(0, ".")
logging.basicConfig(level=logging.ERROR)

from backtest.engine import run_backtest, run_smoke  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Run the TRIDENT-X backtester over a short window.")
    ap.add_argument("--days", type=int, default=None, help="Limit window to last N days (default: all).")
    ap.add_argument("--smoke", action="store_true", help="Software-path test only: synthesize entries; never strategy evidence.")
    ap.add_argument("--symbols", nargs="*", default=None, help="Subset of pairs, e.g. SOLUSDT XRPUSDT.")
    ap.add_argument("--initial", type=float, default=10000.0)
    ap.add_argument("--apply-weights", action="store_true", help="Disabled: live/adaptive weight changes require offline promotion review.")
    ap.add_argument("--out", default="backtest/run.json", help="Output JSON path.")
    args = ap.parse_args()

    if args.apply_weights:
        ap.error("--apply-weights is disabled. Export research metrics and complete an offline promotion review instead.")

    t0 = time.time()
    fn = run_smoke if args.smoke else run_backtest
    sim, stats, rejections, eng_stats, consensus_reasons = fn(
        symbols=args.symbols, initial=args.initial, days=args.days
    )

    report = {
        "mode": "smoke" if args.smoke else "real",
        "performance_evidence": not args.smoke,
        "disclaimer": "Synthetic forced-entry smoke output is a software-path test, not trading-performance evidence." if args.smoke else "Historical results require independent review of data, costs, and validation design.",
        "days": args.days,
        "elapsed_s": round(time.time() - t0, 1),
        "stats": stats,
        "rejections": dict(sorted(rejections.items(), key=lambda kv: -kv[1])),
        "engine_activity": eng_stats,
        "consensus_neutral_reasons": dict(sorted(consensus_reasons.items(), key=lambda kv: -kv[1])),
        "sample_trades": [{
            "symbol": c["symbol"], "dir": c["direction"], "pnl": round(c["pnl"], 2),
            "exit": c["exit_reason"], "opened": c["opened_at"], "closed": c["closed_at"],
        } for c in sim.closed[:10]],
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)

    s = stats
    print(f"[{report['mode']}] {s['trades']} trades | win_rate {s['win_rate']:.3f} | "
          f"pnl {s['total_pnl']:.2f} | equity {s['final_equity']:.2f} (peak {s['peak']:.2f})")
    print(f"[{report['mode']}] exits: " + ", ".join(f"{k}={v}" for k, v in s["exit_reasons"].items()))
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()