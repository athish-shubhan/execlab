import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from execlab import strategies as S
from execlab.data import Trades, quote_grid
from execlab.sim import FILL_MODELS, Market, Params, run
from tests.conftest import synthetic_trades

DT = 5_000_000


def _market(tr: Trades) -> Market:
    g = quote_grid(tr, DT)
    return Market.build(tr.ts, tr.price, tr.qty, tr.side, g.t, g.bid, g.ask)


def _params(fill: str = "base", **kw) -> Params:
    # two knots on a power law: log-log interpolation reproduces 0.001 * sqrt(notional) exactly
    kx = np.log([1e2, 1e8])
    base = dict(walk_x=kx, walk_y=np.log(0.001 * np.sqrt(np.exp(kx))), lam_per_musd=2.0, queue=0.05, maker_bps=1.0, taker_bps=3.0,
                fill_model=FILL_MODELS[fill], decay=0.98)
    base.update(kw)
    return Params(**base)


def _plans(q: float, n: int, mk: Market, k0: int) -> list[S.Plan]:
    vol = np.ones(n)
    return [S.market(q, n), S.twap(q, n), S.vwap(q, n, np.linspace(1, 2, n)), S.pov(q, n, vol, 0.2),
            S.almgren_chriss(q, n, 0.05), S.twap(q, n, S.PASSIVE, "passive"), S.twap(q, n, S.MODEL, "model")]


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 2), st.sampled_from([1, -1]), st.floats(0.01, 20.0), st.integers(4, 60),
       st.sampled_from(list(FILL_MODELS)))
def test_completes_reconciles_and_never_overfills(seed, side, q, n, fill):
    tr = synthetic_trades(n_sec=600, seed=seed)
    mk = _market(tr)
    rng = np.random.default_rng(seed)
    pf, op = rng.random(mk.grid_t.size), rng.normal(0, 1, mk.grid_t.size)
    for plan in _plans(q, n, mk, 10):
        res, _ = run(mk, _params(fill), 10, n, side, q, plan.target, plan.mode, pf, op)
        total, filled, cash_check = res[0], res[6], res[8]
        assert filled == pytest.approx(1.0, abs=1e-9)  # completion constraint
        assert filled <= 1.0 + 1e-12                    # never fill more than the parent
        assert total == pytest.approx(cash_check, abs=1e-6)  # cash and inventory reconcile
        assert sum(res[1:6]) == pytest.approx(total, abs=1e-9)  # decomposition adds up


def test_no_lookahead_actions_identical_on_truncated_data(trades):
    """Run every strategy on data truncated at time t and on the full data:
    every action taken before t must be identical."""
    full = _market(trades)
    k0, n = 20, 120
    cut_step = 70
    t_cut = int(full.grid_t[k0 + cut_step])
    trunc_tr = trades.slice_time(int(trades.ts[0]), t_cut)
    # quotes and model inputs after the cut are garbage on purpose in the truncated market
    g = quote_grid(trunc_tr, DT, t0=int(full.grid_t[0]), t1=int(full.grid_t[-1]) + 1)
    pad = full.grid_t.size - g.t.size
    bid = np.r_[g.bid, np.full(pad, g.bid[-1])]
    ask = np.r_[g.ask, np.full(pad, g.ask[-1])]
    trunc = Market.build(trunc_tr.ts, trunc_tr.price, trunc_tr.qty, trunc_tr.side, full.grid_t, bid, ask)
    rng = np.random.default_rng(1)
    pf, op = rng.random(full.grid_t.size), rng.normal(0, 1, full.grid_t.size)
    pf_t, op_t = pf.copy(), op.copy()
    pf_t[k0 + cut_step:] = rng.random(pf_t[k0 + cut_step:].size)
    op_t[k0 + cut_step:] = 99.0
    for plan in _plans(2.0, n, full, k0):
        for fill in FILL_MODELS:
            _, a_full = run(full, _params(fill), k0, n, 1, 2.0, plan.target, plan.mode, pf, op)
            _, a_cut = run(trunc, _params(fill), k0, n, 1, 2.0, plan.target, plan.mode, pf_t, op_t)
            # step j is decided at grid time k0+j and rests until k0+j+1; steps that finish
            # before the cut only ever saw pre-cut data
            assert np.array_equal(a_full[:cut_step], a_cut[:cut_step]), (plan.name, fill)


def test_quote_grid_never_crosses(trades):
    g = quote_grid(trades, 1_000_000)
    assert np.all(g.ask > g.bid)


def test_fill_model_ordering(trades):
    """More optimistic fill assumptions can only fill more passively."""
    mk = _market(trades)
    q, n = 5.0, 100
    plan = S.twap(q, n, S.PASSIVE)
    frac = {f: run(mk, _params(f, catchup=1e9), 10, n, 1, q, plan.target, plan.mode)[0][7] for f in FILL_MODELS}
    assert frac["pessimistic"] <= frac["base"] + 1e-12 <= frac["optimistic"] + 2e-12


def test_market_order_costs_spread_walk_and_fee(trades):
    mk = _market(trades)
    p = _params(lam_per_musd=0.0)
    res, _ = run(mk, p, 10, 10, 1, 1.0, S.market(1.0, 10).target, S.market(1.0, 10).mode)
    k = 10
    mid = 0.5 * (mk.bid[k] + mk.ask[k])
    hs = (mk.ask[k] - mid) / mid * 1e4
    assert res[2] == pytest.approx(hs, rel=1e-6)
    assert res[3] == pytest.approx(0.001 * np.sqrt(mk.ask[k]) * mk.ask[k] / mid, rel=1e-3)
    assert res[4] == pytest.approx(3.0, rel=1e-3)


def test_almgren_chriss_limits():
    tw = S.twap(10, 20)
    assert np.allclose(S.almgren_chriss(10, 20, 0.0).target, tw.target)
    front = S.almgren_chriss(10, 20, 1.0).target
    assert front[0] > tw.target[0] and front[-1] == pytest.approx(10)
    assert np.all(np.diff(front) >= -1e-12)
