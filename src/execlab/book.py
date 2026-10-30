"""Local order book maintained from a snapshot plus diff updates (Binance spot procedure),
and the book-walk cost of a market order.

The update-ID rules follow Binance's documented procedure for managing a local
order book ("How to manage a local order book correctly", Binance Spot API docs):

1. Buffer diff events from the depth stream; note ``U`` of the first event.
2. Fetch a REST snapshot. If its ``lastUpdateId`` is below that first ``U``, fetch again.
3. Drop buffered events with ``u <= lastUpdateId``.
4. The first applied event must satisfy ``U <= lastUpdateId + 1 <= u``.
5. Afterwards an event with ``U`` greater than the local update ID + 1 means updates
   were missed; the book must be resynchronised from a new snapshot.
6. Quantities are absolute; a quantity of zero removes the level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class GapError(RuntimeError):
    """The diff stream skipped an update; the book must be resynchronised."""


@dataclass
class LocalBook:
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    last_update_id: int = -1
    synced: bool = False

    def apply_snapshot(self, snap: dict) -> None:
        self.bids = {float(p): float(q) for p, q in snap["bids"] if float(q) > 0}
        self.asks = {float(p): float(q) for p, q in snap["asks"] if float(q) > 0}
        self.last_update_id = int(snap["lastUpdateId"])
        self.synced = False

    def apply_diff(self, ev: dict) -> bool:
        """Apply one depth event. Returns False if it was stale and ignored.

        Raises :class:`GapError` if the event does not continue the sequence.
        """
        first, last = int(ev["U"]), int(ev["u"])
        if last <= self.last_update_id:
            return False
        if not self.synced:
            if not first <= self.last_update_id + 1 <= last:
                raise GapError(f"first event [{first}, {last}] does not cover {self.last_update_id + 1}")
            self.synced = True
        elif first > self.last_update_id + 1:
            raise GapError(f"expected U<={self.last_update_id + 1}, got {first}")
        for side, levels in ((self.bids, ev["b"]), (self.asks, ev["a"])):
            for p, q in levels:
                pf, qf = float(p), float(q)
                if qf == 0.0:
                    side.pop(pf, None)
                else:
                    side[pf] = qf
        self.last_update_id = last
        return True

    def top(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Arrays of shape (n, 2): [[price, qty], ...] for bids (desc) and asks (asc), NaN-padded."""
        out = []
        for side, rev in ((self.bids, True), (self.asks, False)):
            keys = sorted(side, reverse=rev)[:n]
            a = np.full((n, 2), np.nan)
            if keys:
                a[: len(keys), 0] = keys
                a[: len(keys), 1] = [side[k] for k in keys]
            out.append(a)
        return out[0], out[1]

    def best(self) -> tuple[float, float]:
        return (max(self.bids) if self.bids else np.nan, min(self.asks) if self.asks else np.nan)

    def crossed(self) -> bool:
        b, a = self.best()
        return bool(np.isfinite(b) and np.isfinite(a) and b >= a)


def walk_cost_bps(levels: np.ndarray, notional: float, mid: float) -> float:
    """Cost in bps versus mid of a market order of ``notional`` walking ``levels``.

    ``levels`` is (n, 2) [[price, qty], ...] ordered from the touch outwards (asks for
    a buy, bids for a sell). Returns NaN if the visible book is too thin.
    """
    remaining = notional
    spent_qty = 0.0
    for px, qty in levels:
        if not np.isfinite(px):
            break
        take = min(remaining, px * qty)
        spent_qty += take / px
        remaining -= take
        if remaining <= 1e-9:
            vwap = notional / spent_qty
            return float(abs(vwap - mid) / mid * 1e4)
    return float("nan")


def queue_imbalance(bid_qty: float, ask_qty: float) -> float:
    """(Qb - Qa) / (Qb + Qa) at the touch."""
    s = bid_qty + ask_qty
    return (bid_qty - ask_qty) / s if s > 0 else 0.0


def microprice(bid: float, ask: float, bid_qty: float, ask_qty: float) -> float:
    """Imbalance-weighted mid, the first-order version of Stoikov's (2018) micro-price."""
    s = bid_qty + ask_qty
    return (bid * ask_qty + ask * bid_qty) / s if s > 0 else 0.5 * (bid + ask)
