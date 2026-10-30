"""Order-flow features and execution targets on a 1-second grid.

Every feature at grid time t uses only trades printed strictly before t.
Targets look forward from t and are only ever used as labels.

Trade-based analogues are used for the book quantities in the spec because the
historical files have no book: signed taker volume replaces order flow imbalance
(Cont, Kukanov and Stoikov, 2014), and volume printed at the current touch stands
in for queue imbalance. With recorded L2 data, ``book_features`` adds true queue
imbalance and the micro-price (Stoikov, 2018).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from .data import QuoteGrid, Trades, per_second

FEATURE_GROUPS = {
    "flow": ["flow_1", "flow_5", "flow_30", "flow_120"],
    "activity": ["lvol_1", "lvol_5", "lvol_30", "lcount_5", "since_move"],
    "returns": ["ret_1", "ret_5", "ret_30", "ret_120"],
    "volatility": ["rv_30", "rv_300", "spread_ticks"],
    "touch": ["touch_sell_5", "touch_buy_5"],
    "clock": ["hour_sin", "hour_cos"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def _roll(x: np.ndarray, w: int) -> np.ndarray:
    """Sum of x over the w bins *before* each index (bins i-w .. i-1)."""
    c = np.r_[0.0, np.cumsum(x)]
    i = np.arange(x.size)
    return c[i] - c[np.maximum(i - w, 0)]


def build(tr: Trades, q: QuoteGrid) -> pd.DataFrame:
    """Features on the 1-second quote grid ``q`` (grid step must be 1s)."""
    t0 = int(q.t[0])
    n = q.t.size
    ps = per_second(tr, t0 - 1_000_000, n + 1)  # bin i covers [t_i - 1s, t_i)
    vol, sgn, cnt = ps["volume"][:n], ps["signed"][:n], ps["count"][:n]
    # shift so that index i aggregates bins strictly before grid time t_i: bin i is [t_i-1s, t_i)
    vol_c, sgn_c, cnt_c = np.r_[vol[1:], 0], np.r_[sgn[1:], 0], np.r_[cnt[1:], 0]
    mid = q.mid
    lm = np.log(mid) * 1e4
    f = pd.DataFrame(index=pd.Index(q.t, name="t"))
    for w in (1, 5, 30, 120):
        v, s = _roll(vol_c, w), _roll(sgn_c, w)
        f[f"flow_{w}"] = np.where(v > 0, s / np.where(v > 0, v, 1), 0.0)
    for w in (1, 5, 30):
        f[f"lvol_{w}"] = np.log1p(_roll(vol_c, w))
    f["lcount_5"] = np.log1p(_roll(cnt_c, 5))
    for w in (1, 5, 30, 120):
        f[f"ret_{w}"] = lm - np.r_[np.full(w, lm[0]), lm[:-w]]
    r1 = np.r_[0.0, np.diff(lm)]
    f["rv_30"] = np.sqrt(_roll(r1**2, 30) / 30)
    f["rv_300"] = np.sqrt(_roll(r1**2, 300) / 300)
    f["spread_ticks"] = (q.ask - q.bid) / q.tick
    moved = np.r_[True, np.diff(mid) != 0]
    last = np.maximum.accumulate(np.where(moved, np.arange(n), 0))
    f["since_move"] = np.log1p(np.arange(n) - last)
    sell_at_bid, buy_at_ask = _touch_volume(tr.ts, tr.price, tr.qty, tr.side.astype(np.int64),
                                            q.t, q.bid, q.ask, 5_000_000)
    f["touch_sell_5"] = np.log1p(sell_at_bid)
    f["touch_buy_5"] = np.log1p(buy_at_ask)
    hour = (q.t // 3_600_000_000) % 24
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return f


@njit(cache=True)
def _touch_volume(ts, price, qty, side, grid, bid, ask, window):
    """Volume sold at the current bid and bought at the current ask over the last ``window`` us."""
    n = grid.size
    sb = np.zeros(n)
    ba = np.zeros(n)
    lo = 0
    hi = 0
    for k in range(n):
        t = grid[k]
        while hi < ts.size and ts[hi] < t:
            hi += 1
        while lo < hi and ts[lo] < t - window:
            lo += 1
        for i in range(lo, hi):
            if side[i] < 0 and price[i] == bid[k]:
                sb[k] += qty[i]
            elif side[i] > 0 and price[i] == ask[k]:
                ba[k] += qty[i]
    return sb, ba


@njit(cache=True)
def _fill_labels(ts, price, qty, side, grid, bid, ask, horizon, queue0, fill_model, order_side):
    """For a small resting order posted at each grid time at our touch, was it filled
    within ``horizon`` us, and when? Same queue logic as the simulator."""
    n = grid.size
    filled = np.zeros(n, dtype=np.int8)
    i0 = 0
    for k in range(n):
        t = grid[k]
        while i0 < ts.size and ts[i0] < t:
            i0 += 1
        lim = bid[k] if order_side > 0 else ask[k]
        qa = queue0 if fill_model == 1 else 0.0
        i = i0
        while i < ts.size and ts[i] < t + horizon:
            if side[i] == -order_side:
                p = price[i]
                if (order_side > 0 and p < lim) or (order_side < 0 and p > lim):
                    filled[k] = 1
                    break
                if p == lim and fill_model != 0:
                    x = qty[i] - qa
                    qa = max(qa - qty[i], 0.0)
                    if x > 0:
                        filled[k] = 1
                        break
            i += 1
    return filled


def targets(tr: Trades, q: QuoteGrid, horizons: tuple[int, ...] = (1, 5, 30), fill_h: int = 5,
            queue: float = 0.0, fill_model: int = 1) -> pd.DataFrame:
    """Forward-looking labels on the 1s grid.

    dir_h:      -1/0/+1 sign of the mid move over h seconds beyond half the spread
    dm_h:       mid move over h seconds, bps
    fill_buy:   a resting bid posted now fills within fill_h seconds (base fill model)
    fill_sell:  same for a resting ask
    opp_buy:    dm_{fill_h} * (1 - fill_buy): the drift a buyer pays when the bid is not filled
    opp_sell:   -dm_{fill_h} * (1 - fill_sell)
    as_buy:     dm_{fill_h} * fill_buy: drift after a passive buy fill (adverse selection if < 0)
    """
    lm = np.log(q.mid) * 1e4
    hs = 0.5 * (q.ask - q.bid) / q.mid * 1e4
    y = pd.DataFrame(index=pd.Index(q.t, name="t"))
    for h in horizons:
        dm = np.r_[lm[h:] - lm[:-h], np.full(h, np.nan)]
        y[f"dm_{h}"] = dm
        y[f"dir_{h}"] = np.where(np.isnan(dm), np.nan, np.where(dm > hs, 1, np.where(dm < -hs, -1, 0)))
    side = tr.side.astype(np.int64)
    for name, s in (("buy", 1), ("sell", -1)):
        y[f"fill_{name}"] = _fill_labels(tr.ts, tr.price, tr.qty, side, q.t, q.bid, q.ask,
                                         fill_h * 1_000_000, queue, fill_model, s).astype(float)
    dm = y[f"dm_{fill_h}"]
    y["opp_buy"] = dm * (1 - y["fill_buy"])
    y["opp_sell"] = -dm * (1 - y["fill_sell"])
    y["as_buy"] = dm * y["fill_buy"]
    y["as_sell"] = -dm * y["fill_sell"]
    y.iloc[-max(*horizons, fill_h) - 1:] = np.nan
    return y
