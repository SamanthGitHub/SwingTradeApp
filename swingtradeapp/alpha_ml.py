"""Meta-labeling for the Alpha Engine: a walk-forward P(profit) overlay on the factor ranking.

Idea (López de Prado's *meta-labeling*): the factor model decides *direction/ranking*; a secondary
classifier decides *how much to trust each pick* by predicting the probability the name beats its
peers over the next holding period. The engine then drops/keeps names by that probability.

Leak-free by construction
-------------------------
* **Features** are the cross-sectional factor z-scores at the rebalance date — lookahead-free.
* **Labels** are a *forward* return (they peek ahead) — which is fine for *training* as long as we
  only ever train on samples whose forward window has **already closed** by the prediction date. We
  enforce that positionally: to predict at bar ``i`` we train only on samples at bars ``≤ i-horizon``.
* The model is refit walk-forward on an expanding window; the prediction for date ``d`` is therefore
  genuine out-of-sample.

Degrades gracefully: if scikit-learn is unavailable the overlay is skipped (returns ``None``).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _forward_label(px: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """1.0 if a name's forward ``horizon``-day return beats the cross-sectional median, else 0.0.

    Relative (beat-the-peers) labels match what the ranking actually does and are stationary across
    bull/bear regimes. NaN where the forward window runs off the end of the data.
    """
    fwd = px.shift(-horizon) / px - 1.0
    rel = fwd.sub(fwd.median(axis=1), axis=0)
    return (rel > 0).astype(float).where(fwd.notna())


def _feature_frame(factor_panels: Dict[str, pd.DataFrame], dates: List[pd.Timestamp],
                   feature_names: List[str]) -> pd.DataFrame:
    """Stack the per-factor z-score panels into a (date, symbol) × factor feature matrix."""
    frames = [factor_panels[name].loc[dates].stack().rename(name) for name in feature_names]
    return pd.concat(frames, axis=1)


def walkforward_proba(
    px: pd.DataFrame,
    factor_panels: Dict[str, pd.DataFrame],
    rebal_dates: List[pd.Timestamp],
    *,
    horizon: int = 5,
    min_train: int = 250,
    retrain_every: int = 4,
    model: str = "logit",
) -> Optional[pd.DataFrame]:
    """Walk-forward out-of-sample P(profit) panel aligned to ``px`` (values only at rebalance dates).

    Returns ``None`` if scikit-learn is missing or there's never enough realised history to train.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None

    feats = [n for n in factor_panels if n != "composite"]
    rebal_dates = [pd.Timestamp(d) for d in rebal_dates if d in px.index]
    label = _forward_label(px, horizon)

    X_all = _feature_frame(factor_panels, rebal_dates, feats)
    y_all = label.loc[rebal_dates].stack().rename("y")
    data = X_all.join(y_all, how="inner").dropna()
    if data.empty:
        return None

    pos = {d: i for i, d in enumerate(px.index)}            # bar position for the horizon cutoff
    sample_dates = data.index.get_level_values(0)
    out = pd.DataFrame(np.nan, index=px.index, columns=px.columns)

    def _new_model():
        if model == "hgb":
            return HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                                  learning_rate=0.05, l2_regularization=1.0)
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5))

    clf, trained = None, False
    for i, d in enumerate(rebal_dates):
        cutoff = pos[d] - horizon                            # only labels realised by d
        train = data[sample_dates <= px.index[cutoff]] if cutoff >= 0 else data.iloc[:0]
        if len(train) < min_train or train["y"].nunique() < 2:
            continue
        if clf is None or i % retrain_every == 0:
            clf = _new_model().fit(train[feats].values, train["y"].values.astype(int))
            trained = True
        today = data[sample_dates == d]
        if today.empty:
            continue
        p = clf.predict_proba(today[feats].values)[:, 1]
        for (dd, sym), pv in zip(today.index, p):
            out.loc[dd, sym] = float(pv)
    return out if trained else None
