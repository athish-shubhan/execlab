"""End-to-end experiment: calibrate on training days, simulate parent orders on test days.

Walk-forward by day: for test day d, everything (cost curve, impact slope, queue size,
volume profile, ML models) is fitted on days before d only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import strategies as S
from .calibrate import WalkFit, fit_walk, price_response, queue_estimate, sigma_bps, volume_profile
from .data import QuoteGrid, Trades, quote_grid
from .features import FEATURES, build, targets
from .models import Fitted, fit_baseline, tune_xgb
from .sim import FILL_MODELS, OUT_NAMES, Market, Params, run

DAY_US = 86_400_000_000


def concat(trs: list[Trades]) -> Trades:
    return Trades(trs[0].symbol, *(np.concatenate([getattr(t, k) for t in trs])
                                   for k in ("ts", "price", "qty", "side", "agg_id")))


def split_days(tr: Trades) -> dict[str, Trades]:
    day = tr.ts // DAY_US
    out = {}
    for d in np.unique(day):
        m = day == d
        out[str(pd.to_datetime(d * DAY_US, unit="us").date())] = Trades(
            tr.symbol, tr.ts[m], tr.price[m], tr.qty[m], tr.side[m], tr.agg_id[m])
    return out


@dataclass
class Calibration:
    walk: WalkFit
    lam: float
    response_exponent: float
    response_ci: tuple[float, float]
    queue: float
    profile: pd.Series
    sigma_step_bps: float
    sigma_1m_bps: float
    adv_usd: float
    response_table: pd.DataFrame


def calibrate(train: Trades, dt_s: int) -> Calibration:
    w = fit_walk(train)
    pr = price_response(train)
    n_days = max(1, round((train.ts[-1] - train.ts[0]) / DAY_US))
    return Calibration(w, pr.lam_per_musd, pr.exponent, pr.exponent_ci, queue_estimate(train),
                       volume_profile(train, dt_s), sigma_bps(train, dt_s), sigma_bps(train, 60),
                       float((train.qty * train.price).sum() / n_days), pr.table)


@dataclass
class ModelSet:
    pfill: dict[int, Fitted]   # side -> classifier
    opp: dict[int, Fitted]     # side -> regressor
    info: dict[str, Any] = field(default_factory=dict)


def train_models(train: Trades, cal: Calibration, fill_h: int, n_trials: int, ledger: Any = None,
                 study: str = "", use_xgb: bool = True) -> tuple[ModelSet, pd.DataFrame, pd.DataFrame]:
    q1 = quote_grid(train, 1_000_000)
    x = build(train, q1)
    y = targets(train, q1, fill_h=fill_h, queue=cal.queue)
    ok = y.notna().all(axis=1).to_numpy() & x.notna().all(axis=1).to_numpy()
    x, y = x[ok], y[ok]
    pf: dict[int, Fitted] = {}
    op: dict[int, Fitted] = {}
    info: dict[str, Any] = {}
    for side, name in ((1, "buy"), (-1, "sell")):
        for task, tgt, store in (("binary", f"fill_{name}", pf), ("regression", f"opp_{name}", op)):
            yy = y[tgt].to_numpy()
            if use_xgb:
                m, best, trials = tune_xgb(x, yy, task, FEATURES, fill_h, n_trials, ledger,
                                           f"{study}_{tgt}")
                info[tgt] = {"best_params": best, "trials": trials.to_dict(orient="records")}
            else:
                m = fit_baseline(x, yy, task, FEATURES)
            store[side] = m
    return ModelSet(pf, op, info), x, y


def decision_grid(day: Trades, dt_s: int) -> QuoteGrid:
    start = (int(day.ts[0]) // DAY_US) * DAY_US + dt_s * 1_000_000
    return quote_grid(day, dt_s * 1_000_000, t0=start)


def predictions_on_grid(models: ModelSet | None, day: Trades, grid: QuoteGrid,
                        side: int) -> tuple[np.ndarray, np.ndarray]:
    if models is None:
        z = np.zeros(grid.t.size)
        return z, z
    q1 = quote_grid(day, 1_000_000, t0=int(grid.t[0]) - 300_000_000)
    x = build(day, q1).reindex(grid.t).ffill().fillna(0.0)
    return models.pfill[side].predict(x), models.opp[side].predict(x)


@dataclass(frozen=True)
class OrderSpec:
    k0: int
    side: int
    notional: float


def parent_orders(grid: QuoteGrid, n_steps: int, notionals: list[float], every_s: int,
                  dt_s: int) -> list[OrderSpec]:
    starts = np.arange(300 // dt_s, grid.t.size - n_steps - 2, every_s // dt_s)
    return [OrderSpec(int(k), s, n) for k in starts for s in (1, -1) for n in notionals]


def simulate_day(day: Trades, cal: Calibration, models: ModelSet | None, dt_s: int, horizon_s: int,
                 notionals: list[float], every_s: int, fees: dict[str, tuple[float, float]],
                 ac_sweep: list[float], pov_mult: float = 1.5, lam_mult: float = 1.0,
                 half_life_s: float = 300.0, fill_models: tuple[str, ...] = tuple(FILL_MODELS),
                 ) -> tuple[pd.DataFrame, dict[str, float]]:
    grid = decision_grid(day, dt_s)
    mk = Market.build(day.ts, day.price, day.qty, day.side, grid.t, grid.bid, grid.ask)
    n = horizon_s // dt_s
    decay = 0.5 ** (dt_s / half_life_s) if np.isfinite(half_life_s) else 1.0
    # market volume per decision step, observed after the step ends
    k = np.searchsorted(grid.t, day.ts, side="right") - 1
    m = k >= 0
    step_vol = np.bincount(k[m], day.qty[m], minlength=grid.t.size)
    sod_step = ((grid.t // 1_000_000) % 86400) // dt_s
    preds = {s: predictions_on_grid(models, day, grid, s) for s in (1, -1)}
    rows = []
    scanned = 0
    t_start = time.perf_counter()
    for o in parent_orders(grid, n, notionals, every_s, dt_s):
        mid0 = 0.5 * (grid.bid[o.k0] + grid.ask[o.k0])
        q = o.notional / mid0
        prof = cal.profile.reindex(sod_step[o.k0:o.k0 + n] % len(cal.profile)).to_numpy()
        exp_vol = prof.sum() * cal.adv_usd / mid0
        rate = min(0.5, pov_mult * q / max(exp_vol, 1e-9))
        plans = [S.market(q, n), S.twap(q, n), S.vwap(q, n, prof),
                 S.pov(q, n, step_vol[o.k0:o.k0 + n], rate)]
        plans += [S.almgren_chriss(q, n, kp) for kp in ac_sweep]
        plans += [S.twap(q, n, S.PASSIVE, "passive_twap")]
        if models is not None:
            plans += [S.twap(q, n, S.MODEL, "signal_aware")]
        for fee_name, (maker, taker) in fees.items():
            for fm_name in fill_models:
                fm = FILL_MODELS[fm_name]
                p = Params(cal.walk.knots_x, cal.walk.knots_y, cal.lam * lam_mult, cal.queue, maker, taker, fm,
                           decay=decay)
                for plan in plans:
                    if fm_name != "base" and plan.mode[0] == S.AGGRESSIVE:
                        continue  # aggressive plans never rest, so the fill model is irrelevant
                    pf, op = preds[o.side]
                    res, _ = run(mk, p, o.k0, n, o.side, q, plan.target, plan.mode, pf, op)
                    scanned += n
                    rows.append({"day": str(pd.to_datetime(grid.t[o.k0], unit="us").date()),
                                 "t0": int(grid.t[o.k0]), "side": o.side, "notional": o.notional,
                                 "strategy": plan.name, "fees": fee_name, "fill_model": fm_name,
                                 **dict(zip(OUT_NAMES, res, strict=True))})
    elapsed = time.perf_counter() - t_start
    df = pd.DataFrame(rows)
    # aggressive plans: copy base results to the other fill models so every table is complete
    aggr = df[(df["fill_model"] == "base") & ~df["strategy"].isin(["passive_twap", "signal_aware"])]
    extra = [aggr.assign(fill_model=f) for f in fill_models if f != "base"]
    df = pd.concat([df, *extra], ignore_index=True)
    trades_in_window = len(day)
    perf = {"sims": len(rows), "decision_steps": scanned, "seconds": elapsed,
            "steps_per_second": scanned / elapsed if elapsed else float("nan"),
            "trades_in_day": trades_in_window}
    return df, perf
