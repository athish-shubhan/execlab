## Appendix A. Median shortfall by strategy and size (bps, base fill model, 95% interval)

**Retail fees, 10/10 bps**

*BTCUSDT*

| Strategy | $250k | $1M | $5M |
|---|---|---|---|
| Market | 10.19 (10.16 to 10.19) | 10.90 (10.64 to 10.90) | 14.64 (13.12 to 15.88) |
| TWAP | 10.28 (10.26 to 10.28) | 11.06 (11.01 to 11.07) | 15.02 (14.89 to 15.14) |
| VWAP | 10.28 (10.26 to 10.32) | 11.04 (11.00 to 11.13) | 15.19 (14.95 to 15.40) |
| POV | 10.35 (10.35 to 10.41) | 11.30 (11.21 to 11.37) | 14.64 (14.44 to 14.91) |
| Almgren-Chriss (k=0.05) | 10.39 (10.38 to 10.41) | 11.43 (11.43 to 11.45) | 17.10 (17.04 to 17.15) |
| Passive TWAP | 10.40 (10.35 to 10.45) | 11.20 (11.14 to 11.25) | 15.21 (15.10 to 15.30) |
| Signal-aware | 10.27 (10.25 to 10.27) | 11.05 (11.00 to 11.06) | 15.01 (14.89 to 15.13) |

*ETHUSDT*

| Strategy | $100k | $500k | $2M |
|---|---|---|---|
| Market | 10.29 (10.24 to 10.29) | 11.59 (11.54 to 11.59) | 19.14 (17.89 to 19.26) |
| TWAP | 10.18 (10.17 to 10.18) | 10.60 (10.59 to 10.72) | 12.49 (12.17 to 13.25) |
| VWAP | 10.21 (10.18 to 10.27) | 10.84 (10.59 to 11.02) | 13.15 (12.45 to 13.69) |
| POV | 10.29 (10.22 to 10.37) | 11.06 (10.86 to 11.06) | 14.04 (13.53 to 14.39) |
| Almgren-Chriss (k=0.05) | 10.27 (10.24 to 10.32) | 11.13 (10.95 to 11.21) | 13.88 (13.61 to 14.23) |
| Passive TWAP | 10.61 (10.57 to 10.73) | 11.18 (11.07 to 11.34) | 13.10 (12.66 to 13.80) |
| Signal-aware | 10.16 (10.16 to 10.17) | 10.61 (10.57 to 10.70) | 12.46 (12.16 to 13.23) |

**Low-fee schedule, 0/2 bps**

*BTCUSDT*

| Strategy | $250k | $1M | $5M |
|---|---|---|---|
| Market | 2.19 (2.16 to 2.19) | 2.90 (2.64 to 2.90) | 6.64 (5.12 to 7.88) |
| TWAP | 2.28 (2.26 to 2.28) | 3.06 (3.01 to 3.07) | 7.02 (6.89 to 7.14) |
| VWAP | 2.28 (2.26 to 2.32) | 3.04 (3.00 to 3.13) | 7.18 (6.94 to 7.40) |
| POV | 2.35 (2.35 to 2.41) | 3.30 (3.21 to 3.37) | 6.64 (6.44 to 6.91) |
| Almgren-Chriss (k=0.05) | 2.39 (2.38 to 2.41) | 3.43 (3.43 to 3.45) | 9.10 (9.03 to 9.15) |
| Passive TWAP | 0.90 (0.81 to 1.10) | 1.88 (1.75 to 2.10) | 6.33 (6.16 to 6.50) |
| Signal-aware | 0.62 (0.54 to 0.87) | 1.61 (1.44 to 1.86) | 5.72 (5.40 to 5.85) |

*ETHUSDT*

| Strategy | $100k | $500k | $2M |
|---|---|---|---|
| Market | 2.29 (2.24 to 2.29) | 3.59 (3.54 to 3.59) | 11.14 (9.88 to 11.27) |
| TWAP | 2.18 (2.17 to 2.18) | 2.60 (2.59 to 2.72) | 4.49 (4.17 to 5.25) |
| VWAP | 2.21 (2.18 to 2.27) | 2.84 (2.59 to 3.02) | 5.15 (4.45 to 5.69) |
| POV | 2.29 (2.22 to 2.37) | 3.06 (2.86 to 3.06) | 6.05 (5.53 to 6.39) |
| Almgren-Chriss (k=0.05) | 2.27 (2.24 to 2.32) | 3.13 (2.95 to 3.21) | 5.88 (5.61 to 6.22) |
| Passive TWAP | 1.41 (1.30 to 1.63) | 2.09 (2.01 to 2.22) | 3.99 (3.75 to 4.82) |
| Signal-aware | 1.32 (1.22 to 1.48) | 2.08 (1.75 to 2.15) | 3.77 (3.55 to 4.73) |

## Appendix B. Where the cost goes (mean bps, low-fee schedule, base fill model, largest order size)

| Symbol | Strategy | Spread | Impact | Delay | Fees | Total | Passive share |
|---|---|---|---|---|---|---|---|
| BTCUSDT | Market | 0.03 | 4.50 | 0.00 | 2.00 | 6.53 | 0% |
| BTCUSDT | TWAP | 0.02 | 5.02 | 0.00 | 2.00 | 7.04 | 0% |
| BTCUSDT | Passive TWAP | -0.01 | 4.96 | 0.46 | 0.92 | 6.34 | 54% |
| BTCUSDT | Signal-aware | 0.02 | 4.71 | 0.92 | 0.31 | 5.95 | 85% |
| ETHUSDT | Market | 0.08 | 8.55 | 0.00 | 2.00 | 10.63 | 0% |
| ETHUSDT | TWAP | 0.07 | 2.83 | 0.00 | 2.00 | 4.90 | 0% |
| ETHUSDT | Passive TWAP | -0.03 | 2.78 | 0.76 | 0.79 | 4.29 | 61% |
| ETHUSDT | Signal-aware | -0.02 | 2.75 | 0.87 | 0.51 | 4.11 | 75% |

Delay is positive for the passive strategies because the unfilled part of a resting order tends to be the part the market moved away from: adverse selection shows up as timing cost.

## Appendix C. Reconstructed taker orders by size (training days)

*BTCUSDT*

| Size bucket (USD) | Orders | Mean walk (bps) | 95th percentile (bps) |
|---|---|---|---|
| 1,000 to 3,000 | 86,965 | 0.015 | 0.092 |
| 3,000 to 10,000 | 60,219 | 0.037 | 0.221 |
| 10,000 to 30,000 | 36,347 | 0.032 | 0.261 |
| 30,000 to 100,000 | 8,665 | 0.047 | 0.316 |
| 100,000 to 300,000 | 1,540 | 0.110 | 0.685 |
| 300,000 to 1,000,000 | 346 | 0.363 | 1.533 |
| 1,000,000 to 3,000,000 | 42 | 1.368 | 3.096 |
| 3,000,000 to 10,000,000 | 7 | 3.408 | 6.391 |

*ETHUSDT*

| Size bucket (USD) | Orders | Mean walk (bps) | 95th percentile (bps) |
|---|---|---|---|
| 1,000 to 3,000 | 80,428 | 0.037 | 0.210 |
| 3,000 to 10,000 | 41,652 | 0.026 | 0.138 |
| 10,000 to 30,000 | 11,763 | 0.043 | 0.265 |
| 30,000 to 100,000 | 2,860 | 0.142 | 0.874 |
| 100,000 to 300,000 | 505 | 0.353 | 1.676 |
| 300,000 to 1,000,000 | 105 | 1.367 | 3.033 |
| 1,000,000 to 3,000,000 | 12 | 3.517 | 6.328 |

