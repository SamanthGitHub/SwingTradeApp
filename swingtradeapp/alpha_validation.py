"""Backtest-integrity tools: Probability of Backtest Overfitting (PBO) + purged/embargoed CV.

The PSR/DSR in ``alpha_engine`` ask "is this Sharpe real given the sample and the number of tries?".
**PBO** asks the complementary, brutal question: *if I pick the config that looks best in-sample, how
often is it actually below-median out-of-sample?* A high PBO means your selection process is fitting
noise — the most important number a backtester can compute, and the one most often skipped.

Pure logic, no Streamlit/network. Method: Combinatorially-Symmetric Cross-Validation (CSCV),
Bailey, Borwein, López de Prado & Zhu (2015).
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _sharpe_cols(returns: np.ndarray) -> np.ndarray:
    """Per-column Sharpe (mean/std) of a (rows × configs) return matrix; 0 where std is 0."""
    mu = returns.mean(axis=0)
    sd = returns.std(axis=0)
    out = np.zeros_like(mu)
    nz = sd > 0
    out[nz] = mu[nz] / sd[nz]
    return out


def pbo_cscv(returns: pd.DataFrame, n_blocks: int = 10) -> Dict[str, float]:
    """Probability of Backtest Overfitting via CSCV.

    ``returns`` is a (T observations × N configurations) matrix of per-period returns — one column per
    candidate strategy config. Splits time into ``n_blocks`` contiguous blocks, and over every way of
    choosing half the blocks as in-sample (the rest out-of-sample): pick the best-Sharpe config IS,
    then record its OOS rank. PBO = fraction of splits where that IS-winner lands below the OOS median
    (logit ≤ 0). Lower is better: <~0.2 robust, ~0.5 coin-flip (pure overfit).

    Returns ``{"PBO", "n_configs", "n_splits"}`` (NaN PBO if too few configs/observations).
    """
    M = returns.dropna(how="any")
    T, N = M.shape
    if N < 2 or T < 2 * n_blocks:
        return {"PBO": float("nan"), "n_configs": N, "n_splits": 0}
    rows = (T // n_blocks) * n_blocks
    arr = M.values[:rows]
    blocks = np.array_split(np.arange(rows), n_blocks)

    logits: List[float] = []
    for combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        is_rows = np.concatenate([blocks[i] for i in combo])
        oos_rows = np.concatenate([blocks[i] for i in range(n_blocks) if i not in combo])
        perf_is = _sharpe_cols(arr[is_rows])
        perf_oos = _sharpe_cols(arr[oos_rows])
        n_star = int(np.argmax(perf_is))                       # config chosen in-sample
        # relative OOS rank of the IS winner, in (0, 1)
        ranks = perf_oos.argsort().argsort()                   # 0 = worst
        w = (ranks[n_star] + 1) / (N + 1)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(w / (1 - w))))

    logits_arr = np.array(logits)
    return {"PBO": float((logits_arr <= 0).mean()), "n_configs": N, "n_splits": len(logits_arr)}


def purged_kfold_indices(n: int, k: int = 5, embargo: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Purged K-fold splits: each contiguous test fold, with an ``embargo`` band of training rows
    around it removed to stop label/feature windows leaking across the train/test boundary.

    A utility for leak-free ML CV (the meta-label in ``alpha_ml`` already embargoes via its horizon
    cutoff; this generalises it). Returns ``[(train_idx, test_idx), ...]``.
    """
    idx = np.arange(n)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold in np.array_split(idx, k):
        if len(fold) == 0:
            continue
        lo, hi = max(0, fold[0] - embargo), min(n, fold[-1] + embargo + 1)
        train = np.concatenate([idx[:lo], idx[hi:]])
        out.append((train, fold))
    return out
