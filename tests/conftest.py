import numpy as np
import pytest

from execlab.data import Trades


def synthetic_trades(n_sec: int = 1800, rate: float = 8.0, seed: int = 0, t0: int = 1_793_059_200_000_000) -> Trades:
    """A random-walk market with a one-tick spread: buyers lift the ask, sellers hit the bid."""
    rng = np.random.default_rng(seed)
    n = rng.poisson(rate * n_sec)
    ts = np.sort(rng.integers(t0, t0 + n_sec * 1_000_000, n)).astype(np.int64)
    side = np.where(rng.random(n) < 0.5, 1, -1).astype(np.int8)
    steps = rng.choice([-1, 0, 0, 0, 1], size=n)
    bid = 50_000 + 0.01 * np.cumsum(steps)
    price = np.round(np.where(side > 0, bid + 0.01, bid), 2)
    qty = np.round(rng.exponential(0.05, n) + 0.0001, 5)
    return Trades("SYN", ts, price, qty, side, np.arange(n, dtype=np.int64))


@pytest.fixture(scope="session")
def trades() -> Trades:
    return synthetic_trades()
