"""Live L2 recorder for Binance spot: depth diffs + trades, written to Parquet.

    python -m execlab.recorder --symbols BTCUSDT ETHUSDT --levels 20 --out data/l2

Run it on your own machine for 2 to 4 weeks (e.g. under tmux or systemd). It:

- subscribes to ``<symbol>@depth@100ms`` and ``<symbol>@trade`` on one combined stream,
- rebuilds each local book with the snapshot-plus-diff procedure in :mod:`execlab.book`,
- resynchronises automatically on any update-ID gap or disconnect, and logs each resync,
- stores the top ``levels`` of the book after every applied update, plus every trade,
  with both the exchange event time and the local receive time (for clock-offset checks),
- writes zstd-compressed Parquet partitioned as ``symbol=<S>/date=<YYYY-MM-DD>/``.

No API key is needed: these are public endpoints.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .book import GapError, LocalBook

WS_URL = "wss://stream.binance.com:9443/stream?streams="
REST_DEPTH = "https://api.binance.com/api/v3/depth"
log = logging.getLogger("execlab.recorder")


class ParquetSink:
    """Buffers rows per (kind, symbol, date) and flushes them as Parquet parts."""

    def __init__(self, root: Path, flush_rows: int = 50_000) -> None:
        self.root, self.flush_rows = root, flush_rows
        self.buf: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        self.part = 0

    def add(self, kind: str, symbol: str, row: dict[str, Any]) -> None:
        date = datetime.fromtimestamp(row["exch_ts_ms"] / 1000, UTC).strftime("%Y-%m-%d")
        key = (kind, symbol, date)
        self.buf[key].append(row)
        if len(self.buf[key]) >= self.flush_rows:
            self.flush(key)

    def flush(self, key: tuple[str, str, str] | None = None) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for k in [key] if key else list(self.buf):
            rows = self.buf.pop(k, [])
            if not rows:
                continue
            kind, sym, date = k
            d = self.root / kind / f"symbol={sym}" / f"date={date}"
            d.mkdir(parents=True, exist_ok=True)
            self.part += 1
            pq.write_table(pa.Table.from_pylist(rows), d / f"part-{int(time.time())}-{self.part:06d}.parquet",
                           compression="zstd")


class SymbolState:
    def __init__(self, symbol: str, levels: int) -> None:
        self.symbol, self.levels = symbol, levels
        self.book = LocalBook()
        self.buffer: list[dict[str, Any]] = []
        self.ready = False
        self.resyncs = 0


async def fetch_snapshot(session: Any, symbol: str, limit: int = 1000) -> dict[str, Any]:
    async with session.get(REST_DEPTH, params={"symbol": symbol, "limit": limit}) as r:
        r.raise_for_status()
        return await r.json()  # type: ignore[no-any-return]


async def sync(session: Any, st: SymbolState, sink: ParquetSink) -> None:
    """Snapshot + buffered diffs -> synced book (Binance procedure, steps 1-4)."""
    st.ready = False
    while not st.buffer:
        await asyncio.sleep(0.05)
    first_u = int(st.buffer[0]["U"])
    while True:
        snap = await fetch_snapshot(session, st.symbol)
        if int(snap["lastUpdateId"]) >= first_u:
            break
        await asyncio.sleep(0.25)
    st.book.apply_snapshot(snap)
    pending, st.buffer = st.buffer, []
    try:
        for ev in pending:
            st.book.apply_diff(ev)
    except GapError as e:  # buffer does not bridge the snapshot: the next event retries
        log.warning("%s sync failed (%s); retrying", st.symbol, e)
        return
    st.ready = True
    sink.add("events", st.symbol, {"exch_ts_ms": int(time.time() * 1000), "recv_ts_ns": time.time_ns(),
                                   "event": "resync", "update_id": st.book.last_update_id})
    log.info("%s synced at update %d", st.symbol, st.book.last_update_id)


def book_row(st: SymbolState, ev: dict[str, Any], recv_ns: int) -> dict[str, Any]:
    bids, asks = st.book.top(st.levels)
    return {"exch_ts_ms": int(ev["E"]), "recv_ts_ns": recv_ns, "U": int(ev["U"]), "u": int(ev["u"]),
            "bid_px": bids[:, 0].tolist(), "bid_qty": bids[:, 1].tolist(),
            "ask_px": asks[:, 0].tolist(), "ask_qty": asks[:, 1].tolist(),
            "crossed": st.book.crossed()}


async def record(symbols: list[str], levels: int, out: Path, minutes: float | None) -> None:
    import aiohttp
    import websockets

    sink = ParquetSink(out)
    states = {s.upper(): SymbolState(s.upper(), levels) for s in symbols}
    streams = "/".join(f"{s.lower()}@depth@100ms/{s.lower()}@trade" for s in symbols)
    deadline = time.time() + minutes * 60 if minutes else None
    async with aiohttp.ClientSession() as session:
        while deadline is None or time.time() < deadline:
            try:
                async with websockets.connect(WS_URL + streams, ping_interval=20, max_size=2**23) as ws:
                    for st in states.values():
                        st.buffer, st.ready = [], False
                    syncers: dict[str, asyncio.Task[None]] = {}
                    async for raw in ws:
                        recv_ns = time.time_ns()
                        msg = json.loads(raw)
                        data, stream = msg["data"], msg["stream"]
                        sym = data["s"]
                        st = states[sym]
                        if stream.endswith("@trade"):
                            sink.add("trades", sym, {"exch_ts_ms": int(data["T"]), "recv_ts_ns": recv_ns,
                                                     "trade_id": int(data["t"]), "price": float(data["p"]),
                                                     "qty": float(data["q"]),
                                                     "side": -1 if data["m"] else 1})
                        elif not st.ready:
                            st.buffer.append(data)
                            if sym not in syncers or syncers[sym].done():
                                syncers[sym] = asyncio.create_task(sync(session, st, sink))
                        else:
                            try:
                                if st.book.apply_diff(data):
                                    sink.add("book", sym, book_row(st, data, recv_ns))
                            except GapError as e:
                                st.resyncs += 1
                                log.warning("%s gap (%s); resyncing", sym, e)
                                sink.add("events", sym, {"exch_ts_ms": int(data["E"]), "recv_ts_ns": recv_ns,
                                                         "event": "gap", "update_id": int(data["U"])})
                                st.buffer, st.ready = [data], False
                                syncers[sym] = asyncio.create_task(sync(session, st, sink))
                        if deadline and time.time() > deadline:
                            break
            except (TimeoutError, OSError, websockets.ConnectionClosed) as e:
                log.warning("connection lost (%s); reconnecting in 2s", e)
                await asyncio.sleep(2)
    sink.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--levels", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("data/l2"))
    ap.add_argument("--minutes", type=float, default=None, help="stop after this long (default: run forever)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    np.seterr(all="ignore")
    asyncio.run(record(a.symbols, a.levels, a.out, a.minutes))


if __name__ == "__main__":
    main()
