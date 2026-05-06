"""Evaluation metrics for rainfall reconstruction."""

from __future__ import annotations

import numpy as np


def _valid_pair(observed, simulated) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observed, dtype=float)
    sim = np.asarray(simulated, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    return obs[mask], sim[mask]


def rmse(observed, simulated) -> float:
    """Root mean square error."""
    obs, sim = _valid_pair(observed, simulated)
    return float(np.sqrt(np.mean((sim - obs) ** 2))) if len(obs) else float("nan")


def mae(observed, simulated) -> float:
    """Mean absolute error."""
    obs, sim = _valid_pair(observed, simulated)
    return float(np.mean(np.abs(sim - obs))) if len(obs) else float("nan")


def nse(observed, simulated) -> float:
    """Nash-Sutcliffe efficiency."""
    obs, sim = _valid_pair(observed, simulated)
    if len(obs) == 0:
        return float("nan")
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((sim - obs) ** 2) / denom)


def pbias(observed, simulated) -> float:
    """Percent bias."""
    obs, sim = _valid_pair(observed, simulated)
    denom = np.sum(obs)
    if len(obs) == 0 or denom == 0:
        return float("nan")
    return float(100.0 * np.sum(sim - obs) / denom)
