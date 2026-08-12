# TRIDENT-X Upgrade Guide

## What this upgrade changes

This working copy extends the original **paper-trading** system with a research-grade decision path. It preserves the existing multi-engine market monitor, risk controls, SQLite store, and FastAPI service, while adding a canonical price-action record, deterministic decision review, audit storage, a calibrated-ML safety boundary, and a redesigned operator dashboard.

| Area | New behavior | Why it matters |
| --- | --- | --- |
| Price action | Each closed 5-minute bar produces a `SetupSnapshot` using confirmed pivots, structure, sweep/reclaim, displacement, participation, location, a trade plan, and timestamped evidence. | Candidate evidence is explicit and can be audited or labeled later. |
| Decision review | A deterministic reviewer produces `TRADE`, `WATCH`, or `NO_TRADE`, with structured supporting and rejecting evidence. | The explanation layer cannot invent data or override risk controls. |
| ML qualifier | A versioned calibrated tabular artifact is the only model allowed to provide a probability to the review layer. | Existing raw model confidence is not mistaken for a calibrated probability. |
| Execution safety | Until an approved calibrated artifact exists, valid setups are `WATCH` records and cannot become paper-execution eligible through the new path. | Prevents automatic promotion of unvalidated ML scores. |
| Validation | Purged walk-forward split, fixed trade-plan labeling, cost scenarios, latency, and conservative same-bar stop/target treatment are provided as research primitives. | Reduces common leakage and optimistic backtest errors. |
| Dashboard | The operator console now displays current decision, price-action plan, binding no-trade reasons, review notes, research health, audit history, risk, watchlist, and positions. | Operators can distinguish a setup, a model state, and a risk outcome. |

## Important operating boundary

> **The upgraded repository is a research and paper-trading implementation. It does not establish that any setup is profitable or accurate, and it must not be configured for live execution based on the included code or tests.**

The new pipeline has two separate gates. First, the existing deterministic risk gate must pass. Second, the new review must have a valid setup, compatible multi-timeframe context, adequate participation, an acceptable plan, and a registered **calibrated** model probability. If the model artifact is missing, stale, incompatible, or fails, the decision is deliberately `WATCH` or `NO_TRADE`.

## Key files

| File | Purpose |
| --- | --- |
| `research/models.py` | Stable schemas for setup evidence, trade plan, reviewer notes, and final decision. |
| `research/price_action.py` | Closed-candle pivot confirmation, setup-family identification, score, and trade plan creation. |
| `research/review.py` | Deterministic evidence review and hard decision boundary. |
| `research/validation.py` | Purged walk-forward split, target-before-stop labels, cost and latency scenarios. |
| `ml/calibrated_qualifier.py` | Versioned calibrated tabular model interface and chronological baseline training function. |
| `ml/deep_sequence_experiment.py` | Optional, isolated LSTM experiment; not used in the runtime decision path. |
| `core/state_manager.py` | Adds `setup_snapshots` and `decision_records` SQLite audit tables. |
| `api/server.py` | Adds read-only decision and research-health endpoints and dashboard state. |
| `api/dashboard.html` | Decision-first dark operator dashboard. |
| `tests/run_research_checks.py` | Dependency-light verification runner. |

## Installation and checks

Install the normal runtime requirements in an isolated environment. The `requirements-research.txt` file is optional: it contains test and TensorFlow packages used only for offline research.

```bash
pip install -r requirements.txt
pip install -r requirements-research.txt  # only in a research environment
python tests/run_research_checks.py
python tests/check_dashboard_api.py
```

The focused checks confirm that setup records serialize, missing model artifacts cannot yield a model-backed signal, deterministic review blocks model-unavailable candidates from trade eligibility, calibrated artifacts can clear an otherwise valid paper candidate, decision records persist, same-bar stop/target outcomes are treated conservatively, and walk-forward splits purge outcome overlap.

## Producing a trainable dataset

Do not train the qualifier from generic candle direction. The label should be created only for a historically emitted `SetupSnapshot` using its exact stored entry, stop, target, and expiry. The recommended result is `1` only when the target is reached before the stop and time barrier after declared fees, spread, slippage, and latency.

The data pipeline should persist enough historical candles for every configured pair and timeframe before candidate generation. It should then replay closed-candle snapshots chronologically, store the resulting decisions, resolve their outcomes using `label_target_before_stop`, and lock a final unobserved holdout interval. The same test interval must not be reused to select parameters, model versions, or probability thresholds.

`train_chronological_baseline` requires at least 250 ordered labeled setups and holds the final 25% of supplied samples for calibration. This is only a minimum software constraint, not a promotion standard. A promotion report should include calibrated probability plots, Brier score, precision at the trading threshold, net expectancy after multiple cost scenarios, drawdown, turnover, and results split by symbol, regime, direction, and setup family.

## Deep-learning experiment policy

The optional sequence experiment is intentionally disconnected from `main.py`. It requires at least 5,000 ordered labeled rows, TensorFlow in a separate research environment, chronological train/validation/locked-test segments, and a comparison against the calibrated tabular baseline. It should not be considered for promotion if it does not improve *locked* net performance and calibration after trading costs, or if its advantage is limited to a subset that cannot be explained and monitored.

## Dashboard endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /api/state` | Current system, risk, pair state, latest structured decision, and research-health counters. |
| `GET /api/decisions?limit=30` | Immutable decision trail containing no-trade reasons and review evidence. |
| `GET /api/research/health` | Counts of total, watch, no-trade, and model/risk-cleared decisions. |
| `GET /api/signals` | Backwards-compatible legacy signal history. |

The dashboard’s pause/resume/flatten controls remain protected by the existing authentication setting. They are operational controls, not proof of strategy quality.

## Before considering any broader deployment

The system still needs user-specific market and execution information: exact pairs, exchange, market type, maker/taker fee schedule, funding handling, expected spread and slippage, latency assumptions, and the intended holding horizon. It also needs real historical data and a written acceptance threshold. No result from a synthetic or forced-entry test should be accepted as financial-performance evidence.
