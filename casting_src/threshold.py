"""Validation-only bootstrap cost selection with a one-standard-error-style plateau heuristic."""
from __future__ import annotations

import numpy as np


def expected_cost(y_true, p_defect, thresholds, cost_fn=10.0, cost_fp=1.0):
    """Vectorised cost curve: cost_fn * FN + cost_fp * FP at each threshold."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(p_defect)
    pred = p[None, :] >= np.asarray(thresholds)[:, None]      # (T, N)
    fn = (~pred & y[None, :]).sum(axis=1)                      # true defect, called ok
    fp = (pred & ~y[None, :]).sum(axis=1)                      # true ok, called defect
    return cost_fn * fn + cost_fp * fp


def _bootstrap_se(y_true, p_defect, thresholds, cost_fn, cost_fp, n_boot, rng):
    """Standard error of the minimum achievable cost, by resampling the val set."""
    y = np.asarray(y_true)
    p = np.asarray(p_defect)
    n = len(y)
    mins = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        mins[b] = expected_cost(y[idx], p[idx], thresholds, cost_fn, cost_fp).min()
    return float(mins.std(ddof=1))


def select_threshold(y_true, p_defect, cost_fn=10.0, cost_fp=1.0,
                     grid=None, n_boot=200, seed=0):
    """Choose an operating threshold from validation data."""
    if grid is None:
        grid = np.linspace(0.001, 0.999, 999)
    grid = np.asarray(grid)
    rng = np.random.default_rng(seed)

    costs = expected_cost(y_true, p_defect, grid, cost_fn, cost_fp)
    min_cost = float(costs.min())
    argmin_t = float(grid[int(np.argmin(costs))])

    se = _bootstrap_se(y_true, p_defect, grid, cost_fn, cost_fp, n_boot, rng)
    tolerance = min_cost + se

    acceptable = costs <= tolerance
    if not acceptable.any():
        return {"threshold": argmin_t, "argmin_threshold": argmin_t,
                "plateau": (argmin_t, argmin_t), "min_cost": min_cost,
                "se": se, "tolerance": tolerance, "plateau_width": 0.0}

    best_lo = best_hi = 0
    best_width = -1
    lo = None
    for i, ok in enumerate(acceptable):
        if ok and lo is None:
            lo = i
        if (not ok or i == len(acceptable) - 1) and lo is not None:
            hi = i if ok else i - 1
            if hi - lo > best_width:
                best_lo, best_hi = lo, hi
                best_width = hi - lo
            lo = None

    t_lo, t_hi = float(grid[best_lo]), float(grid[best_hi])
    return {"threshold": 0.5 * (t_lo + t_hi),
            "argmin_threshold": argmin_t,
            "plateau": (t_lo, t_hi),
            "min_cost": min_cost,
            "se": se,
            "tolerance": tolerance,
            "plateau_width": t_hi - t_lo}
