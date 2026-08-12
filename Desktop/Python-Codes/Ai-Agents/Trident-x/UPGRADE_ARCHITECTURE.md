# TRIDENT-X Research-Grade Upgrade Architecture

## Purpose and operating boundary

TRIDENT-X should be treated as a **paper-trading research platform** until it produces sufficient out-of-sample evidence under realistic assumptions. The upgraded system will emit structured candidate decisions for selected crypto pairs on closed 5-minute candles. It will not represent calibrated model scores as certain predictions, and it will keep automatic order eligibility behind deterministic risk controls.

The target flow is:

```text
closed 5m candle
  -> data validation and multi-timeframe snapshot
  -> price-action setup ledger
  -> rule-engine evidence and regime filter
  -> calibrated ML qualifier (when an approved model is available)
  -> deterministic review panel
  -> hard risk/execution gate
  -> paper-trade eligibility or recorded no-trade
  -> immutable decision record + dashboard event
```

## 1. Canonical price-action setup ledger

The system should build one `SetupSnapshot` per symbol and closed 5-minute candle. It should make price-action observations explicit rather than hiding them inside oscillator-weighted confidence values.

| Evidence family | Required fields | Interpretation |
| --- | --- | --- |
| Market regime | regime label, confidence, realized volatility, trend strength, uncertain flag | Determines which setup families are allowed and which are blocked. |
| Higher-timeframe context | 15m and 1h direction, last confirmed swing, distance to supply/demand or liquidity | Establishes directional context rather than a raw-RSI-only proxy. |
| Structure | confirmed pivot highs/lows, break of structure, change of character, structure direction | Separates continuation from reversal candidates. |
| Liquidity | prior-session / rolling swing pools, sweep side, reclaim confirmation | Confirms whether a stop-run/rejection event occurred. |
| Displacement and imbalance | true-range z-score, body-to-range ratio, close location, fair-value-gap bounds | Measures whether the move has enough directional intent to support an entry. |
| Location | VWAP, session range, ATR-normalized distance to invalidation, retest state | Avoids entering directly into nearby opposing liquidity. |
| Participation | relative volume, volume percentile, CVD proxy and data-quality flag | Requires adequate participation while explicitly reporting proxy limitations. |
| Trade plan | direction, entry reference, invalidation, first target, R-multiple, maximum holding horizon | Generates a falsifiable setup, not merely a directional score. |

The ledger must include both a numeric score and human-readable evidence. Price action should only be computed from data available on or before the candle close. Pivots must use a confirmation lag, so later data cannot create look-ahead bias.

## 2. Setup families

The initial system should use three mutually documented candidate families instead of adding a deep neural model prematurely.

| Setup family | Entry prerequisites | Disqualification |
| --- | --- | --- |
| Trend pullback continuation | aligned 15m/1h context, confirmed 5m structure in direction of trend, pullback into location zone, renewed displacement and volume confirmation | opposing structure break, poor reward-to-risk, stale/weak participation, nearby opposing liquidity. |
| Sweep and reclaim reversal | a defined liquidity sweep, close/reclaim back inside range, confirmed change of character, acceptable higher-timeframe location | no reclaim, continued displacement through invalidation, low relative volume, high-regime uncertainty. |
| Compression breakout and retest | range compression, breakout displacement, participation expansion, optional retest that holds, room to the next liquidity pool | failed breakout, no volume confirmation, adverse higher-timeframe context, entry too far from invalidation. |

Only one setup family may produce an eligible candidate per symbol/candle. Competing candidates are saved as rejected alternatives. This prevents duplicated evidence from being mistaken for independent engine consensus.

## 3. Model hierarchy

The modeling layer is a **qualifier**, not a price oracle or an override for risk gates.

1. A baseline rules-only strategy establishes whether the setup ledger has signal value.
2. A tabular model then estimates the probability that a setup achieves a pre-defined net target before its stop or time barrier. It receives snapshot features only; its training schema is versioned.
3. The model is calibrated on a later validation segment using isotonic or sigmoid calibration, and it outputs a probability with a calibration status.
4. A deep-learning experiment is optional and research-only. It must compete against the tabular baseline on identical, time-ordered folds after data volume supports it. It cannot be introduced as a production dependency merely because TensorFlow/PyTorch is installed.
5. If an approved model, compatible feature schema, calibration curve, or data-freshness requirement is unavailable, the output is `MODEL_UNAVAILABLE`; the candidate is logged but must not receive model-based confidence uplift.

The decision target must be specific: over a chosen maximum holding window, did the entry reach the target net of conservative fees and slippage before it reached the stop? Model evaluation must include precision at the selected threshold, recall, Brier score, calibration plot, net expectancy, drawdown, turnover, and performance segmented by symbol, regime, direction, and setup family.

## 4. Validation and research controls

The new backtester will not reuse a smoke-mode synthetic entry as evidence. It will simulate a portfolio with mark-to-market equity and an explicit fill policy.

| Control | Requirement |
| --- | --- |
| Splits | Chronological train / validation / holdout periods; walk-forward retraining only. |
| Leakage | Purge/embargo observations whose outcome windows overlap the training split. All features, labels, and pivots are timestamped. |
| Costs | Configurable maker/taker fees, spread, slippage, latency, funding where applicable, and partial/no-fill behavior. |
| Bar ambiguity | If high and low could hit stop and target in the same bar, resolve with a declared conservative policy unless intrabar data is supplied. |
| Selection | Maintain a parameter registry and test-count log; do not tune parameters on the final holdout period. |
| Reporting | Produce all candidates, all rejections, trade ledger, equity curve, drawdown, setup-family metrics, and cost sensitivity. |
| Release gate | Research results may progress only if sufficient sample size, positive net expectancy after conservative costs, calibrated probabilities, and stability across holdouts are demonstrated. |

## 5. Deterministic signal review panel

The “agent discussion” experience should be an auditable deterministic panel built from computed evidence. It should not fabricate live market facts, silently change the trade decision, or submit orders.

The review panel will produce:

| Reviewer | Responsibility | Example output |
| --- | --- | --- |
| Structure reviewer | Checks higher-timeframe alignment, pivot status, and structure shift. | “15m context is bullish; 5m pullback held above confirmed pivot.” |
| Liquidity reviewer | Checks sweep/reclaim and distance to opposing pools. | “Buy-side liquidity was swept and reclaimed; next opposing pool is 1.8R away.” |
| Participation reviewer | Checks relative volume and displacement. | “Displacement is 1.7 ATR with volume at the 82nd percentile.” |
| Model reviewer | Reports approved calibrated probability and model/data status. | “Calibrated target-before-stop probability: 0.58; calibration window current.” |
| Risk reviewer | Applies hard exclusions: data freshness, risk band, circuit breaker, size, reward-to-risk, correlation and exposure. | “No trade: risk band ORANGE permits no new candidate.” |
| Verdict | Summarizes trade / watch / no-trade and lists binding reasons. | “NO TRADE — structure evidence valid, but 1h context conflicts.” |

The final verdict is computed from transparent rules. An optional natural-language layer can only rephrase the evidence and must remain isolated from the execution path.

## 6. Operator dashboard

The existing single-page FastAPI dashboard should become a responsive dark operator console. The visual hierarchy should keep the most safety-critical information above the fold and show a clear `PAPER / RESEARCH` status.

| Panel | Contents | Data source |
| --- | --- | --- |
| System ribbon | Paper/research mode, data freshness, data-quality state, model version, calibrated/unavailable status, circuit breaker and pause state. | Health, monitor, model registry. |
| Candidate card | Current setup family, direction, trade/no-trade status, setup score, calibrated probability, entry, stop, target, R-multiple, expiry. | `DecisionRecord`. |
| Evidence ledger | Higher-timeframe context, structure, liquidity, displacement, volume, location, model, and risk rows with pass/warn/fail states. | `SetupSnapshot` and reviewer outputs. |
| Review dialogue | Compact sequential reviewer arguments, counterarguments, and binding verdict rationale. | Deterministic review output. |
| Market watchlist | Pair, last price, regime, setup state, model status, current signal state, 24h change and freshness. | Existing pair state plus decision summary. |
| Positions/risk | Open position plan, live R, drawdown, daily loss, exposure, correlated exposure, circuit breaker. | State manager and risk modules. |
| Research monitor | Candidate count, trade eligibility rate, calibrated-score distribution, paper outcomes by setup family/regime, model version and feature drift flags. | Research tables. |
| Audit timeline | Every decision, no-trade reason, execution simulation event, and model/retrain change. | Append-only event records. |

The dashboard must not put a live-order button beside a persuasive model score. Existing pause and emergency flatten controls should remain clearly separated and protected by existing dashboard authentication.

## 7. Persistence and APIs

Two structured tables are required: `setup_snapshots` and `decision_records`. The first stores timestamped evidence; the second stores the verdict, candidate plan, reviewer arguments, model metadata, and binding gates. The signal table remains for backwards compatibility, but should refer to the decision record when available.

New read-only endpoints should expose the latest decision, decision history, and research health. Existing state and websocket updates should include a compact decision summary so that the dashboard remains usable without repeated expensive queries.

## 8. Implementation order

1. Correct higher-timeframe calculation and add stable domain schemas.
2. Add price-action snapshots, setup detection, deterministic review, and no-trade logging.
3. Add persistence and dashboard cards / API endpoints.
4. Replace the current smoke-style research harness with a conservative simulator and walk-forward evaluation utilities.
5. Train and register a calibrated tabular baseline only once the label dataset exists.
6. Add deep-learning experiments only after the baseline and data-volume gates are satisfied.
7. Keep `PAPER_TRADING=True`; live routing remains out of scope for this upgrade.

## References

The validation approach is grounded in research on finance-specific leakage/overfitting controls and in standard probability-calibration documentation.[1][2] Cost and slippage modeling are included because they can materially change trading-strategy backtest results.[3]

[1]: https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110
[2]: https://scikit-learn.org/stable/modules/calibration.html
[3]: https://www.quantstart.com/articles/Successful-Backtesting-of-Algorithmic-Trading-Strategies-Part-II/
