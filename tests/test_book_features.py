import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from execlab.book import GapError, LocalBook, microprice, queue_imbalance, walk_cost_bps
from execlab.calibrate import queue_estimate
from execlab.data import quote_grid, taker_orders
from execlab.features import FEATURES, build, targets
from execlab.quality import trades_report


def snap(uid=100):
    return {"lastUpdateId": uid, "bids": [["99.0", "1"], ["98.0", "2"]], "asks": [["101.0", "1"], ["102.0", "3"]]}


def ev(first, last, b=(), a=()):
    return {"U": first, "u": last, "b": [list(x) for x in b], "a": [list(x) for x in a]}


def test_snapshot_then_diffs():
    bk = LocalBook()
    bk.apply_snapshot(snap(100))
    assert not bk.apply_diff(ev(90, 100))                 # stale, ignored
    assert bk.apply_diff(ev(95, 105, b=[("99.5", "4")]))  # first event straddles 101
    assert bk.apply_diff(ev(106, 110, a=[("101.0", "0")]))  # zero removes level
    bids, asks = bk.top(3)
    assert bids[0].tolist() == [99.5, 4.0] and asks[0].tolist() == [102.0, 3.0]
    assert np.isnan(asks[2, 0])


def test_gap_detection():
    bk = LocalBook()
    bk.apply_snapshot(snap(100))
    with pytest.raises(GapError):
        bk.apply_diff(ev(103, 110))  # does not cover 101
    bk.apply_snapshot(snap(100))
    bk.apply_diff(ev(101, 105))
    with pytest.raises(GapError):
        bk.apply_diff(ev(107, 110))  # skipped 106


@settings(max_examples=40, deadline=None)
@given(st.lists(st.tuples(st.booleans(), st.integers(1, 40), st.integers(0, 5)), min_size=1, max_size=60))
def test_book_never_crosses_with_consistent_updates(updates):
    """Updates that respect the touch (bids below best ask, asks above best bid) never cross."""
    bk = LocalBook()
    bk.apply_snapshot({"lastUpdateId": 0, "bids": [["100", "1"]], "asks": [["101", "1"]]})
    uid = 0
    for is_bid, off, q in updates:
        b, a = bk.best()
        if is_bid:
            px = (a if np.isfinite(a) else 101) - off * 0.5
            upd = ev(uid + 1, uid + 1, b=[(str(px), str(q))])
        else:
            px = (b if np.isfinite(b) else 100) + off * 0.5
            upd = ev(uid + 1, uid + 1, a=[(str(px), str(q))])
        uid += 1
        bk.apply_diff(upd)
        assert not bk.crossed()


def test_walk_cost_and_book_helpers():
    asks = np.array([[100.0, 1.0], [101.0, 1.0], [np.nan, np.nan]])
    assert walk_cost_bps(asks, 50.0, 99.5) == pytest.approx((100 - 99.5) / 99.5 * 1e4)
    v = walk_cost_bps(asks, 150.0, 99.5)
    assert v > walk_cost_bps(asks, 50.0, 99.5)
    assert np.isnan(walk_cost_bps(asks, 1e6, 99.5))
    assert queue_imbalance(3, 1) == 0.5
    assert microprice(99, 101, 3, 1) == pytest.approx(100.5)


def test_features_use_only_past_trades(trades):
    """Features computed on truncated data equal those on full data up to the cut."""
    q = quote_grid(trades, 1_000_000)
    f_full = build(trades, q)
    cut = int(q.t[900])
    tr_cut = trades.slice_time(int(trades.ts[0]), cut)
    q_cut = quote_grid(tr_cut, 1_000_000, t0=int(q.t[0]), t1=cut + 1)
    f_cut = build(tr_cut, q_cut)
    common = f_cut.index.intersection(f_full.index)
    pd.testing.assert_frame_equal(f_full.loc[common, FEATURES], f_cut.loc[common, FEATURES])


def test_targets_and_calibration_run(trades):
    q = quote_grid(trades, 1_000_000)
    y = targets(trades, q, queue=0.01)
    assert set(y["dir_5"].dropna().unique()) <= {-1.0, 0.0, 1.0}
    assert 0 < y["fill_buy"].mean() < 1
    assert queue_estimate(trades) > 0
    assert len(taker_orders(trades)) > 0
    rep = trades_report(trades, q)
    assert rep["agg_id_gaps"] == 0 and rep["proxy_crossed"] == 0
