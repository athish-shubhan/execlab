"""Replay throughput: numba-compiled simulator versus the same loop in pure Python.

A passive TWAP is used because it is the most expensive plan to replay: every step scans
the trades printed while its order rests.

    python studies/benchmark.py [--sample]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from execlab import strategies as S
from execlab.data import quote_grid, read_agg_trades, read_sample
from execlab.sim import Market, Params, simulate_one

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    a = ap.parse_args()
    if a.sample:
        tr = read_sample(ROOT / "sample" / "btcusdt_trades.csv.gz")
    else:
        tr = read_agg_trades([ROOT.parent / "data" / "binance" / "BTCUSDT-aggTrades-2026-10-29.zip"])
    g = quote_grid(tr, 5_000_000)
    mk = Market.build(tr.ts, tr.price, tr.qty, tr.side, g.t, g.bid, g.ask)
    n = 180
    starts = np.arange(60, g.t.size - n - 2, 37)
    plan = S.twap(10.0, n, S.PASSIVE)
    z = np.zeros(g.t.size)
    acts = np.zeros(n, dtype=np.int64)
    p = Params(np.log([1e3, 1e7]), np.log([0.02, 3.0]), 3.5, 0.03, 0.0, 2.0)

    def call(fn, k0):
        return fn(mk.ts, mk.price, mk.qty, mk.side, mk.trade_idx, mk.grid_t, mk.bid, mk.ask, int(k0), n, 1, 10.0,
                  plan.target, plan.mode, z, z, p.walk_x, p.walk_y, p.lam_per_musd, p.queue, p.maker_bps,
                  p.taker_bps, p.fill_model, p.catchup, p.decay, acts)

    call(simulate_one, starts[0])  # compile
    t = time.perf_counter()
    for k0 in starts:
        call(simulate_one, k0)
    jit_s = time.perf_counter() - t
    py_starts = starts[:20]
    t = time.perf_counter()
    for k0 in py_starts:
        call(simulate_one.py_func, k0)
    py_s = time.perf_counter() - t
    # trades replayed per parent order: everything inside its 15-minute window
    span = [np.searchsorted(mk.ts, mk.grid_t[k0 + n]) - mk.trade_idx[k0] for k0 in starts]
    res = {"parent_orders": int(starts.size), "decision_steps_per_order": n,
           "trades_in_window_per_order": float(np.mean(span)),
           "numba_orders_per_s": starts.size / jit_s, "python_orders_per_s": py_starts.size / py_s,
           "numba_decision_steps_per_s": starts.size * n / jit_s,
           "python_decision_steps_per_s": py_starts.size * n / py_s,
           "numba_window_trades_per_s": float(np.sum(span) / jit_s)}
    res["speedup"] = res["numba_orders_per_s"] / res["python_orders_per_s"]
    out = ROOT / "reports" / ("sample" if a.sample else "")
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
