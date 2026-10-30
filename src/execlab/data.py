"""Loading Binance public trade data and deriving a quote proxy from it.

Binance aggTrades carry the aggressor side directly (``is_buyer_maker``), so no
trade-classification heuristic is needed. Since there is no order book in the
historical files, the best bid and ask are inferred from trades: the latest
buyer-initiated trade price is a lower bound on the ask at that moment and the
latest seller-initiated trade price an upper bound on the bid. For BTCUSDT and
ETHUSDT, which trade at a one-tick spread almost all the time, this proxy is
close to the true touch; when the recorder runs, the real book replaces it.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

AGG_COLS = ["agg_id", "price", "qty", "first_id", "last_id", "ts", "is_buyer_maker", "best_match"]


@dataclass(frozen=True)
class Trades:
    """Column arrays for one symbol, sorted by time. ``ts`` is integer microseconds UTC."""

    symbol: str
    ts: np.ndarray       # int64 us
    price: np.ndarray    # float64
    qty: np.ndarray      # float64, base units
    side: np.ndarray     # int8, +1 buyer aggressor, -1 seller aggressor
    agg_id: np.ndarray   # int64

    def __len__(self) -> int:
        return int(self.ts.size)

    def slice_time(self, t0: int, t1: int) -> Trades:
        a, b = np.searchsorted(self.ts, [t0, t1], side="left")
        return Trades(self.symbol, self.ts[a:b], self.price[a:b], self.qty[a:b],
                      self.side[a:b], self.agg_id[a:b])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"ts": self.ts, "price": self.price, "qty": self.qty,
                             "side": self.side, "agg_id": self.agg_id})


def _to_us(ts: np.ndarray) -> np.ndarray:
    ts = ts.astype(np.int64)
    return ts if ts.size and ts[0] > 10**14 else ts * 1000  # older files are in ms


def read_agg_trades(paths: list[str | Path], symbol: str | None = None) -> Trades:
    frames = []
    for p in sorted(map(Path, paths)):
        with zipfile.ZipFile(p) as z, z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None, names=AGG_COLS,
                             dtype={"price": float, "qty": float})
        if df.iloc[0, 0] == "agg_trade_id" or not str(df.iloc[0, 0]).isdigit():
            df = df.iloc[1:]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    is_bm = df["is_buyer_maker"].astype(str).str.lower().eq("true").to_numpy()
    out = Trades(
        symbol or Path(paths[0]).name.split("-")[0],
        _to_us(df["ts"].to_numpy()),
        df["price"].to_numpy(float),
        df["qty"].to_numpy(float),
        np.where(is_bm, -1, 1).astype(np.int8),
        df["agg_id"].astype(np.int64).to_numpy(),
    )
    order = np.argsort(out.ts, kind="stable")
    return Trades(out.symbol, out.ts[order], out.price[order], out.qty[order],
                  out.side[order], out.agg_id[order])


def read_sample(path: str | Path) -> Trades:
    df = pd.read_csv(path)
    return Trades(Path(path).name.split("_")[0].upper(), df["ts"].to_numpy(np.int64),
                  df["price"].to_numpy(float), df["qty"].to_numpy(float),
                  df["side"].to_numpy(np.int8), df["agg_id"].to_numpy(np.int64))


@dataclass(frozen=True)
class QuoteGrid:
    """Quote proxy sampled on a regular grid of decision times (start of each step)."""

    t: np.ndarray       # int64 us, grid times
    bid: np.ndarray
    ask: np.ndarray
    tick: float

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread_bps(self) -> np.ndarray:
        return (self.ask - self.bid) / self.mid * 1e4


def infer_tick(price: np.ndarray) -> float:
    d = np.diff(np.unique(np.round(price[:200_000], 8)))
    d = d[d > 0]
    return float(np.round(np.min(d), 8)) if d.size else 0.01


def quote_grid(tr: Trades, step_us: int, t0: int | None = None, t1: int | None = None,
               stale_us: int = 1_000_000) -> QuoteGrid:
    """Bid/ask proxy at each grid time, using only trades strictly before that time.

    ask = last buyer-aggressor price, bid = last seller-aggressor price. If the two
    cross, or if one side is wide and its last print is over ``stale_us`` older than the
    other side's, the stale side is placed one tick from the fresher one. The proxy
    therefore never crosses, and it only reports a spread wider than one tick when
    both sides have printed recently at those prices.
    """
    tick = infer_tick(tr.price)
    start = t0 if t0 is not None else (tr.ts[0] // step_us + 1) * step_us
    end = t1 if t1 is not None else tr.ts[-1]
    grid = np.arange(start, end, step_us, dtype=np.int64)
    buy = tr.side > 0
    out = {}
    for name, m in (("ask", buy), ("bid", ~buy)):
        ts, px = tr.ts[m], tr.price[m]
        i = np.searchsorted(ts, grid, side="left") - 1
        val = np.where(i >= 0, px[np.clip(i, 0, None)], np.nan)
        last_t = np.where(i >= 0, ts[np.clip(i, 0, None)], -1)
        out[name], out[name + "_t"] = val, last_t
    bid, ask = out["bid"].copy(), out["ask"].copy()
    ask_fresher = out["ask_t"] >= out["bid_t"]
    age_gap = np.abs(out["ask_t"] - out["bid_t"])
    # a side is stale if it crosses the fresher side, or if it is wider than one tick
    # and its last print is more than ``stale_us`` older than the other side's
    crossed = ~(ask > bid)
    stale = crossed | ((ask - bid > tick * 1.5) & (age_gap > stale_us))
    bid = np.where(stale & ask_fresher, ask - tick, bid)
    ask = np.where(stale & ~ask_fresher, bid + tick, ask)
    ok = np.isfinite(bid) & np.isfinite(ask)
    return QuoteGrid(grid[ok], bid[ok], ask[ok], tick)


def taker_orders(tr: Trades) -> pd.DataFrame:
    """Reconstruct taker orders: consecutive aggTrades with the same timestamp and side.

    A market order that sweeps several price levels prints as several aggTrades
    with one timestamp. Returns notional, VWAP and the walk cost relative to the
    first (best) fill price, in bps. This is the cost observed for orders other
    traders chose to send, so it understates the cost of blindly sweeping a given size.
    """
    new = np.r_[True, (np.diff(tr.ts) != 0) | (np.diff(tr.side) != 0)]
    gid = np.cumsum(new) - 1
    notional = np.bincount(gid, tr.price * tr.qty)
    qty = np.bincount(gid, tr.qty)
    first_px = tr.price[new]
    side = tr.side[new]
    vwap = notional / qty
    walk_bps = side * (vwap - first_px) / first_px * 1e4
    n_levels = np.bincount(gid, np.r_[True, np.diff(tr.price) != 0] | new).astype(int)
    return pd.DataFrame({"ts": tr.ts[new], "side": side, "qty": qty, "notional": notional,
                         "first_px": first_px, "vwap": vwap, "walk_bps": walk_bps,
                         "levels": n_levels})


def per_second(tr: Trades, t0: int, n_sec: int) -> dict[str, np.ndarray]:
    """Per-second totals over [t0, t0 + n_sec): volume, signed volume, notional, trade count."""
    s = ((tr.ts - t0) // 1_000_000).astype(np.int64)
    m = (s >= 0) & (s < n_sec)
    s = s[m]
    return {
        "volume": np.bincount(s, tr.qty[m], minlength=n_sec),
        "signed": np.bincount(s, tr.qty[m] * tr.side[m], minlength=n_sec),
        "notional": np.bincount(s, tr.qty[m] * tr.price[m], minlength=n_sec),
        "count": np.bincount(s, minlength=n_sec).astype(float),
    }
