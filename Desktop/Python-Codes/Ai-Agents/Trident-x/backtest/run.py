import sys, time, json, logging

sys.path.insert(0, ".")
logging.basicConfig(level=logging.ERROR)

from backtest.engine import run_backtest

t0 = time.time()
sim, stats, rejections, eng_stats, consensus_reasons = run_backtest()
out = {
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
print("saving run.json")
with open("backtest/run.json", "w") as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out["stats"], indent=2))
print("\nREJECTIONS:")
for k, v in out["rejections"].items():
    print(f"  {k}: {v}")
print("\nENGINE ACTIVITY:")
for k, v in out["engine_activity"].items():
    print(f"  {k}: fired {v['fired']}/{v['eval']}")
print("\nCONSENSUS NEUTRAL REASONS:")
for k, v in out["consensus_neutral_reasons"].items():
    print(f"  {k}: {v}")