"""Tier 2 feedback: load engine metrics produced by a backtest run and feed them
into the consensus per-regime weight table.

Usage:
    python -m backtest.apply_weights        # uses backtest/run.json engine_metrics
    python -m backtest.apply_weights path/to/run.json
"""
import json
import sys

from council.consensus import consensus

DEFAULT = "backtest/run.json"


def apply_metrics(metrics: dict):
    """Feed engine->regime->{wins,losses} into the consensus per-regime weight table."""
    if not metrics:
        print("No engine_metrics to apply.")
        return

    print("BEFORE weights (per regime):")
    before = consensus.get_weights()
    for regime, w in before.items():
        print(f"  {regime}: " + ", ".join(f"{k}={v:.3f}" for k, v in w.items()))

    consensus.apply_backtest_metrics(metrics)

    print("\nAFTER weights (per regime):")
    for regime, w in consensus.get_weights().items():
        print(f"  {regime}: " + ", ".join(f"{k}={v:.3f}" for k, v in w.items()))

    print("\nApplied metrics:")
    for eng, regs in metrics.items():
        for reg, m in regs.items():
            print(f"  {eng}/{reg}: wins={m['wins']} losses={m['losses']}")


def main(path: str = DEFAULT):
    with open(path) as fh:
        data = json.load(fh)
    apply_metrics(data.get("stats", {}).get("engine_metrics", {}))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)