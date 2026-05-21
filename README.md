# Nifty-Predictor
Helps to predict the daily movement of Nifty 50 Stocks

## Institutional ownership radar

The dashboard includes an internal FII/DII ownership tracker under
`Ownership Radar`. It reads normalized quarterly ownership data from:

```text
storage/shareholding/quarterly_ownership.csv
```

To import a new quarterly shareholding export:

```bash
python scripts/update_shareholding.py storage/shareholding/incoming/q1_2026.csv --quarter 2026Q1
```

Expected columns are flexible, but the importer needs the concepts
`quarter`, `symbol`, `fii_pct`, and `dii_pct`. It computes quarter-on-quarter
percentage-point deltas, relative ownership increase, and sector concentration.

## Swing ideas

`Swing Ideas` scans the tracked universe for recurring technical setups:

- Momentum breakout
- Trend pullback
- Mean-reversion bounce

Each idea includes entry, stop, targets, reward/risk, rationale, and
invalidation so it can be used as an action board rather than a black-box call.
