"""Calibration on training days: walk cost, price response, queue size, volume profile.

Everything here is estimated from public trades. Public data shows how prices
respond to *market* order flow; it cannot show the impact of *our* metaorder.
The square-root law (Toth et al., 2011; Bouchaud et al., 2018) is used only as a
reference curve to compare against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Trades, quote_grid, taker_orders


@dataclass(frozen=True)
class WalkFit:
    knots_x: np.ndarray  # log notional (bucket median)
    knots_y: np.ndarray  # log mean walk cost, bps (made non-decreasing)
    a: float             # power-law reference fit on buckets >= $30k: bps = a * notional^b
    b: float
    bins: pd.DataFrame   # per size bucket: n, mean, median, p95 walk bps

    def __call__(self, notional: np.ndarray | float) -> np.ndarray:
        lx = np.log(np.maximum(np.asarray(notional, float), 1e-9))
        x, y = self.knots_x, self.knots_y
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        inner = np.interp(lx, x, y)
        return np.exp(np.where(lx > x[-1], y[-1] + slope * (lx - x[-1]), inner))


def fit_walk(tr: Trades, edges: tuple[float, ...] = (1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7)
             ) -> WalkFit:
    """Walk-cost curve from bucket means of reconstructed taker orders.

    The simulator interpolates between bucket means rather than using a single power law,
    because the observed curve steepens with size: a power law fitted across all sizes is
    dominated by the many small orders and badly understates the cost of large ones.
    """
    to = taker_orders(tr)
    to = to[to["notional"] >= edges[0]]
    cut = pd.cut(to["notional"], list(edges), right=False)
    g = to.groupby(cut, observed=True)["walk_bps"]
    bins = pd.DataFrame({"n": g.size(), "mean": g.mean(), "median": g.median(),
                         "p95": g.quantile(0.95),
                         "notional_mid": to.groupby(cut, observed=True)["notional"].median()})
    ok = (bins["n"] >= 20) & (bins["mean"] > 0)
    x = np.log(bins.loc[ok, "notional_mid"].to_numpy(float))
    y = np.maximum.accumulate(np.log(bins.loc[ok, "mean"].to_numpy(float)))
    big = bins.loc[ok, "notional_mid"].to_numpy(float) >= 3e4
    b, loga = np.polyfit(x[big], y[big], 1) if big.sum() >= 2 else np.polyfit(x, y, 1)
    return WalkFit(x, y, float(np.exp(loga)), float(b), bins.reset_index(names="bucket"))


@dataclass(frozen=True)
class ResponseFit:
    lam_per_musd: float       # linear slope, bps per $1M signed notional
    exponent: float           # power-law exponent of |response| vs |signed notional|
    exponent_ci: tuple[float, float]
    table: pd.DataFrame


def price_response(tr: Trades, interval_s: int = 60, n_bins: int = 12, n_boot: int = 300,
                   seed: int = 0) -> ResponseFit:
    """Regress mid changes over fixed intervals on signed taker notional in the same interval.

    Also fits |dm| ~ c * |Q|^delta on binned data; delta < 1 means concave.
    The bootstrap resamples whole hours to respect dependence within the day.
    """
    step = interval_s * 1_000_000
    q = quote_grid(tr, step)
    t = q.t
    ret = np.diff(q.mid) / q.mid[:-1] * 1e4
    k = np.searchsorted(t, tr.ts, side="right") - 1  # interval i covers [t_i, t_i+1)
    m = (k >= 0) & (k < t.size - 1)
    signed = np.bincount(k[m], (tr.qty * tr.price * tr.side)[m], minlength=t.size - 1) / 1e6
    df = pd.DataFrame({"t": t[:-1], "ret_bps": ret, "signed_musd": signed})
    lam = float(np.polyfit(df["signed_musd"], df["ret_bps"], 1)[0])

    def exponent(d: pd.DataFrame) -> tuple[float, pd.DataFrame]:
        x = d["signed_musd"].abs()
        y = d["ret_bps"] * np.sign(d["signed_musd"])
        b = pd.qcut(x, n_bins, duplicates="drop")
        tab = pd.DataFrame({"abs_musd": x.groupby(b, observed=True).mean(),
                            "resp_bps": y.groupby(b, observed=True).mean(),
                            "n": x.groupby(b, observed=True).size()}).reset_index(drop=True)
        tab = tab[tab["resp_bps"] > 0]
        e = float(np.polyfit(np.log(tab["abs_musd"]), np.log(tab["resp_bps"]), 1)[0])
        return e, tab

    e, tab = exponent(df)
    hours = (df["t"] // 3_600_000_000).to_numpy()
    uh = np.unique(hours)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uh, size=uh.size, replace=True)
        d = pd.concat([df[hours == h] for h in pick])
        try:
            boots.append(exponent(d)[0])
        except (ValueError, TypeError, np.linalg.LinAlgError):
            continue
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return ResponseFit(lam, e, (float(lo), float(hi)), tab)


def queue_estimate(tr: Trades) -> float:
    """Mean seller volume printed at a bid level before that level is traded through.

    Each run of seller-aggressor prints at one price, ended by a seller print at a
    lower price, approximates the resting volume at that level when it was depleted.
    Runs that end because the price moved up are censored and excluded. Used as
    the queue ahead of a newly joined passive order in the base fill model. Using
    the mean rather than the median makes the base model a little more conservative.
    """
    m = tr.side < 0
    px, qy = tr.price[m], tr.qty[m]
    new = np.r_[True, px[1:] != px[:-1]]
    run = np.cumsum(new) - 1
    vol = np.bincount(run, qy)
    run_px = px[new]
    depleted = np.r_[run_px[1:] < run_px[:-1], False]
    return float(vol[depleted].mean()) if depleted.any() else float(vol.mean())


def volume_profile(tr: Trades, step_s: int) -> pd.Series:
    """Share of daily volume by time-of-day bucket of ``step_s`` seconds (training data only)."""
    sod = (tr.ts // 1_000_000) % 86400
    b = sod // step_s
    v = np.bincount(b, tr.qty, minlength=86400 // step_s)
    return pd.Series(v / v.sum())


def sigma_bps(tr: Trades, step_s: int) -> float:
    """Std of mid returns over ``step_s`` seconds, in bps."""
    q = quote_grid(tr, step_s * 1_000_000)
    r = np.diff(np.log(q.mid)) * 1e4
    return float(np.std(r))
