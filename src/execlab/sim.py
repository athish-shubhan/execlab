"""Event-driven execution simulator: replay recorded trades, overlay our own orders.

A parent order of quantity Q (side +1 buy, -1 sell) must complete within K
decision steps of ``dt`` seconds. At each step a strategy supplies a cumulative
target and a mode:

- mode 0, aggressive: cross the spread for whatever is needed to reach target.
- mode 1, passive: rest a limit order at the touch for what is needed, re-pegging
  each step; cross anything more than ``catchup`` slices behind schedule.
- mode 2, model: like mode 1, but at each step compare the predicted cost of
  resting against the cost of crossing now, and pick the cheaper action.

On the last step every remaining unit crosses, so every parent order completes.

Pricing of our fills:
- Crossing pays the touch plus a size-dependent walk cost, interpolated in log-log space
  between the mean costs of reconstructed taker orders by size bucket in the training data
  (and extended beyond the largest bucket along the last segment's slope).
- Our own trading moves later prices: each $1M we execute adds ``lam`` bps of impact,
  which decays by a factor ``decay`` per step (a simple transient-impact kernel).
  ``lam`` defaults to the slope from regressing mid changes on signed market flow.
  Public data cannot show the impact of *our* metaorder, so this is an assumption,
  stated as such and varied in the sensitivity runs.
- Passive orders fill only from replayed opposite-side aggressor trades, under one
  of three fill models (see :data:`FILL_MODELS`).

The hot loop is compiled with numba.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

PESSIMISTIC, BASE, OPTIMISTIC = 0, 1, 2
FILL_MODELS = {"pessimistic": PESSIMISTIC, "base": BASE, "optimistic": OPTIMISTIC}
"""
pessimistic: a resting order fills only when a trade prints strictly through its price
             (the whole level was consumed, so we must have been filled).
base:        we join the back of a queue of ``queue`` base units at our price; prints at
             our price consume the queue ahead of us first; prints through fill us fully.
optimistic:  we are at the front of the queue: any print at or through our price fills us.
"""

N_OUT = 9
OUT_NAMES = ["total_bps", "delay_bps", "spread_bps", "impact_bps", "fee_bps", "opportunity_bps",
             "filled_frac", "passive_frac", "cash_check_bps"]


@dataclass(frozen=True)
class Market:
    """Trades plus the decision grid for one symbol-day."""

    ts: np.ndarray
    price: np.ndarray
    qty: np.ndarray
    side: np.ndarray
    grid_t: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    trade_idx: np.ndarray  # index of first trade at or after each grid time

    @classmethod
    def build(cls, ts: np.ndarray, price: np.ndarray, qty: np.ndarray, side: np.ndarray,
              grid_t: np.ndarray, bid: np.ndarray, ask: np.ndarray) -> Market:
        idx = np.searchsorted(ts, grid_t, side="left").astype(np.int64)
        return cls(ts, price, qty, side.astype(np.int64), grid_t, bid, ask, idx)


@dataclass(frozen=True)
class Params:
    walk_x: np.ndarray     # log notional knots of the walk-cost curve
    walk_y: np.ndarray     # log walk cost (bps) at each knot
    lam_per_musd: float    # permanent impact, bps per $1M signed notional executed by us
    queue: float           # queue ahead (base units) for the base fill model
    maker_bps: float
    taker_bps: float
    fill_model: int = BASE
    catchup: float = 2.0   # slices behind schedule before mode 1 crosses the excess
    decay: float = 1.0     # per-step retention of our own impact (1.0 = permanent)


@njit(cache=True)
def _walk(notional: float, wx: np.ndarray, wy: np.ndarray) -> float:
    """Walk cost in bps: log-log interpolation between knots, flat below the first knot,
    and extended along the last segment's slope above the last knot."""
    if notional <= 0.0:
        return 0.0
    lx = np.log(notional)
    if lx <= wx[0]:
        return np.exp(wy[0])
    n = wx.shape[0]
    if lx >= wx[n - 1]:
        slope = (wy[n - 1] - wy[n - 2]) / (wx[n - 1] - wx[n - 2])
        return np.exp(wy[n - 1] + slope * (lx - wx[n - 1]))
    return np.exp(np.interp(lx, wx, wy))


@njit(cache=True)
def simulate_one(ts, price, qty, tside, trade_idx, grid_t, bid, ask,
                 k0, n_steps, side, total_qty, target, mode, pfill, opp,
                 walk_x, walk_y, lam, queue0, maker, taker, fill_model, catchup,
                 decay, actions):
    """Simulate one parent order. Returns an array of N_OUT results (see OUT_NAMES).

    ``target[j]`` is the cumulative quantity the strategy wants done by the end of
    step j; ``mode[j]`` is 0, 1 or 2; ``pfill`` and ``opp`` are the model's
    predictions at grid index k0 + j (only used in mode 2). ``actions[j]`` is set to
    0 (crossed), 1 (rested), or -1 (nothing to do), for the no-lookahead test.
    """
    out = np.zeros(N_OUT)
    m0 = 0.5 * (bid[k0] + ask[k0])
    filled = 0.0
    cash = 0.0            # signed: buys spend (positive), sells receive (negative)
    delay = 0.0
    spread = 0.0
    impact = 0.0
    fees = 0.0
    passive_qty = 0.0
    imp = 0.0             # our own accumulated impact, bps
    slice_qty = total_qty / n_steps
    rest_px = -1.0
    rest_queue = 0.0
    for j in range(n_steps):
        k = k0 + j
        mid = 0.5 * (bid[k] + ask[k])
        hs_bps = (ask[k] - bid[k]) / mid * 1e4 * 0.5
        if j > 0:
            imp *= decay
        perm = imp
        need = target[j] - filled
        last = j == n_steps - 1
        if last:
            need = total_qty - filled
        if need <= 1e-12:
            actions[j] = -1
            rest_px = -1.0
            continue
        # decide
        act = 0
        if not last and mode[j] != 0:
            act = 1
            if mode[j] == 1:
                behind = need - catchup * slice_qty
                if behind > 1e-12:  # cross the part we are too far behind on
                    touch = ask[k] if side > 0 else bid[k]
                    px = touch * (1.0 + side * (_walk(behind * touch, walk_x, walk_y) + perm) / 1e4)
                    cash += side * behind * px
                    fees += behind * px * taker / 1e4
                    delay += behind * side * (mid - m0)
                    spread += behind * side * (touch - mid)
                    impact += behind * side * (px - touch)
                    imp += lam * behind * px / 1e6
                    filled += behind
                    need -= behind
            else:
                w = _walk(need * mid, walk_x, walk_y)
                d = pfill[k] * (-2.0 * hs_bps - w + maker - taker) + opp[k]
                act = 1 if d < 0.0 else 0
        actions[j] = act
        if act == 0:
            touch = ask[k] if side > 0 else bid[k]
            px = touch * (1.0 + side * (_walk(need * touch, walk_x, walk_y) + perm) / 1e4)
            cash += side * need * px
            fees += need * px * taker / 1e4
            delay += need * side * (mid - m0)
            spread += need * side * (touch - mid)
            impact += need * side * (px - touch)
            imp += lam * need * px / 1e6
            filled += need
            rest_px = -1.0
            continue
        # rest passively at the touch on our side for the window [t_k, t_{k+1})
        lim = bid[k] if side > 0 else ask[k]
        if lim != rest_px:
            rest_queue = queue0 if fill_model == 1 else 0.0
            rest_px = lim
        t_end = grid_t[k + 1]
        i = trade_idx[k]
        rem = need
        got = 0.0
        while i < ts.shape[0] and ts[i] < t_end and rem > 1e-12:
            if tside[i] == -side:  # an aggressor on the other side hits our side
                p = price[i]
                through = (side > 0 and p < lim) or (side < 0 and p > lim)
                at = p == lim
                if through:
                    got += rem
                    rem = 0.0
                elif at and fill_model != 0:
                    x = qty[i]
                    if rest_queue > 0.0:
                        use = min(x, rest_queue)
                        rest_queue -= use
                        x -= use
                    f = min(x, rem)
                    got += f
                    rem -= f
            i += 1
        if got > 0.0:
            px = lim * (1.0 + side * perm / 1e4)
            cash += side * got * px
            fees += got * px * maker / 1e4
            delay += got * side * (mid - m0)
            spread += got * side * (lim - mid)
            impact += got * side * (px - lim)
            imp += lam * got * px / 1e6
            filled += got
            passive_qty += got
    m_end = 0.5 * (bid[k0 + n_steps] + ask[k0 + n_steps])
    unfilled = total_qty - filled
    opportunity = unfilled * side * (m_end - m0)
    denom = total_qty * m0
    out[0] = (delay + spread + impact + fees + opportunity) / denom * 1e4
    out[1] = delay / denom * 1e4
    out[2] = spread / denom * 1e4
    out[3] = impact / denom * 1e4
    out[4] = fees / denom * 1e4
    out[5] = opportunity / denom * 1e4
    out[6] = filled / total_qty
    out[7] = passive_qty / total_qty
    # cash reconciliation: shortfall computed from cash flows alone
    out[8] = ((cash - side * filled * m0) + fees + opportunity) / denom * 1e4
    return out


def run(market: Market, p: Params, k0: int, n_steps: int, side: int, total_qty: float,
        target: np.ndarray, mode: np.ndarray, pfill: np.ndarray | None = None,
        opp: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Python wrapper around :func:`simulate_one`. Returns (results, actions)."""
    if k0 + n_steps >= market.grid_t.size:
        raise ValueError("parent order runs past the end of the data")
    z = np.zeros(market.grid_t.size)
    actions = np.zeros(n_steps, dtype=np.int64)
    res = simulate_one(market.ts, market.price, market.qty, market.side, market.trade_idx,
                       market.grid_t, market.bid, market.ask, k0, n_steps, side, total_qty,
                       np.asarray(target, float), np.asarray(mode, np.int64),
                       z if pfill is None else pfill, z if opp is None else opp,
                       np.asarray(p.walk_x, float), np.asarray(p.walk_y, float), p.lam_per_musd, p.queue, p.maker_bps, p.taker_bps,
                       p.fill_model, p.catchup, p.decay, actions)
    return res, actions
