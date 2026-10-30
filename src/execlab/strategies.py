"""Execution schedules. Each returns (cumulative target per step, mode per step).

All schedules use only information available before each step starts: the
volume profile comes from training days, and percent-of-volume reads volume
already printed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AGGRESSIVE, PASSIVE, MODEL = 0, 1, 2


@dataclass(frozen=True)
class Plan:
    name: str
    target: np.ndarray
    mode: np.ndarray


def market(q: float, n: int) -> Plan:
    """Everything at arrival: the immediate-execution benchmark."""
    return Plan("market", np.full(n, q), np.zeros(n, np.int64))


def twap(q: float, n: int, mode: int = AGGRESSIVE, name: str = "twap") -> Plan:
    return Plan(name, q * np.arange(1, n + 1) / n, np.full(n, mode, np.int64))


def vwap(q: float, n: int, profile: np.ndarray, mode: int = AGGRESSIVE) -> Plan:
    """``profile`` holds expected volume weights for each of the n steps (training days only)."""
    w = np.asarray(profile, float)
    w = w / w.sum() if w.sum() > 0 else np.full(n, 1 / n)
    return Plan("vwap", q * np.cumsum(w), np.full(n, mode, np.int64))


def pov(q: float, n: int, past_volume: np.ndarray, rate: float) -> Plan:
    """Percent of volume: by the end of step j, target ``rate`` x the market volume that
    printed during steps 0..j-1 (already observed when step j starts). The last step
    finishes whatever is left."""
    seen = np.r_[0.0, np.cumsum(past_volume[: n - 1])]
    tgt = np.minimum(q, rate * seen + q / n)  # q/n keeps the first step from being idle
    tgt[-1] = q
    return Plan("pov", np.maximum.accumulate(tgt), np.zeros(n, np.int64))


def almgren_chriss(q: float, n: int, kappa_per_step: float) -> Plan:
    """Almgren and Chriss (2000) optimal trajectory, holdings x_j = Q sinh(k(N-j)) / sinh(kN).

    ``kappa_per_step`` = sqrt(risk_aversion * sigma^2 / eta) in units of 1/step.
    kappa -> 0 recovers TWAP; large kappa front-loads towards the market order.
    """
    j = np.arange(1, n + 1)
    if kappa_per_step < 1e-9:
        return Plan("ac", q * j / n, np.zeros(n, np.int64))
    hold = q * np.sinh(kappa_per_step * (n - j)) / np.sinh(kappa_per_step * n)
    return Plan(f"ac_k{kappa_per_step:.3g}", q - hold, np.zeros(n, np.int64))


def ac_kappa(risk_aversion: float, sigma_bps_step: float, eta_bps_per_unit: float) -> float:
    """kappa from risk aversion (per bps^2 of variance), per-step vol and linear temporary impact."""
    return float(np.sqrt(max(risk_aversion, 0.0) * sigma_bps_step**2 / max(eta_bps_per_unit, 1e-12)))
