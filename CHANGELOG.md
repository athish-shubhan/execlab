# Changelog

## v0.1.0 (2026-10-31)

- Event-driven execution simulator (numba) with pessimistic, base and optimistic fill models and a reconciled shortfall decomposition.
- Market, TWAP, VWAP, POV, Almgren-Chriss and signal-aware strategies; no-lookahead test across all of them.
- Cost curve from reconstructed taker sweeps; price-response regression with hour-block bootstrap.
- Fill-probability and opportunity-cost models (logistic baseline, XGBoost tuned with Optuna on Skeptic's purged walk-forward splits).
- Live L2 recorder (snapshot plus diff, gap detection, automatic resync, Parquet output) and data quality reports.
- Study on three days of BTCUSDT and ETHUSDT trades; research note.
