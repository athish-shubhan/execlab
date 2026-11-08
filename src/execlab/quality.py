"""Data quality reports for historical trades and for the recorder's L2 output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import QuoteGrid, Trades


def trades_report(tr: Trades, q: QuoteGrid | None = None) -> dict[str, Any]:
    dt = np.diff(tr.ts) / 1e6
    ids = np.diff(tr.agg_id)
    rep: dict[str, Any] = {
        "symbol": tr.symbol,
        "start_utc": str(pd.to_datetime(tr.ts[0], unit="us", utc=True)),
        "end_utc": str(pd.to_datetime(tr.ts[-1], unit="us", utc=True)),
        "trades": len(tr),
        "agg_id_gaps": int((ids != 1).sum()),
        "out_of_order_timestamps": int((dt < 0).sum()),
        "max_gap_seconds": float(np.max(dt)) if dt.size else 0.0,
        "gaps_over_10s": int((dt > 10).sum()),
        "zero_or_negative_qty": int((tr.qty <= 0).sum()),
        "trades_per_second_mean": float(len(tr) / max((tr.ts[-1] - tr.ts[0]) / 1e6, 1)),
        "buyer_aggressor_share": float((tr.side > 0).mean()),
        "max_abs_1trade_move_bps": float(np.max(np.abs(np.diff(np.log(tr.price)))) * 1e4),
    }
    if q is not None:
        st = (q.ask - q.bid) / q.tick
        rep |= {"proxy_crossed": int((q.ask <= q.bid).sum()),
                "proxy_spread_1tick_share": float(np.mean(np.isclose(st, 1.0))),
                "proxy_spread_p99_ticks": float(np.percentile(st, 99))}
    return rep


def l2_report(root: str | Path, symbol: str) -> dict[str, Any]:
    """Quality of recorder output: event rate, gaps and resyncs, crossed books, clock offset."""
    import pyarrow.dataset as ds

    root = Path(root)
    book = ds.dataset(root / "book" / f"symbol={symbol}", format="parquet").to_table(
        columns=["exch_ts_ms", "recv_ts_ns", "U", "u", "crossed"]).to_pandas()
    ev_path = root / "events" / f"symbol={symbol}"
    events = ds.dataset(ev_path, format="parquet").to_table().to_pandas() if ev_path.exists() else pd.DataFrame(columns=["event"])
    book = book.sort_values("exch_ts_ms")
    offset_ms = book["recv_ts_ns"] / 1e6 - book["exch_ts_ms"]
    span_s = (book["exch_ts_ms"].iloc[-1] - book["exch_ts_ms"].iloc[0]) / 1000
    gaps = np.diff(book["exch_ts_ms"].to_numpy()) / 1000
    return {
        "symbol": symbol,
        "book_updates": len(book),
        "hours_covered": span_s / 3600,
        "updates_per_second": len(book) / max(span_s, 1),
        "gaps_detected": int((events["event"] == "gap").sum()),
        "resyncs": int((events["event"] == "resync").sum()),
        "crossed_books": int(book["crossed"].sum()),
        "silent_periods_over_5s": int((gaps > 5).sum()),
        "clock_offset_ms_median": float(offset_ms.median()),
        "clock_offset_ms_p99": float(offset_ms.quantile(0.99)),
    }


def to_markdown(reports: list[dict[str, Any]]) -> str:
    df = pd.DataFrame(reports).set_index("symbol").T
    return df.to_markdown()
