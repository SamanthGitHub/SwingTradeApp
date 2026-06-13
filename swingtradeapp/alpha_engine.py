"""Alpha Engine: cross-sectional ranking, portfolio construction, and a lookahead-free backtest.

Pure logic (no Streamlit, no network). Consumes a *price panel* (dates × symbols, adjusted close —
see ``alpha_factors``) and produces:

* **today's book** — the names to hold right now with target weights, and
* an **out-of-sample equity curve + metrics** earned by following the same rules historically.

No-lookahead guarantee
----------------------
Factor scores at date ``t`` use only prices ≤ ``t`` (see ``alpha_factors``). We decide target weights
on a rebalance date ``t`` and apply them to returns from ``t+1`` onward (``holdings.shift(1)``), so the
backtest can never trade on information it wouldn't have had. Factor weights are fixed a priori — there
is **no in-sample optimisation** of the combination, the classic retail-backtest trap.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .alpha_factors import TRADING_DAYS, composite_score, realized_vol


# ── Portfolio construction (one cross-section) ───────────────────────────────────

def _cap_normalize(w: pd.Series, max_weight: float) -> pd.Series:
    """Normalise non-negative weights to sum 1 with **every** name ≤ ``max_weight``.

    Water-filling: cap the over-limit names, redistribute their excess proportionally to the rest,
    repeat to convergence. (A single clip-then-divide would re-breach the cap — the usual bug.) If the
    cap is too tight to be fully invested (``n * max_weight < 1``) all names sit at the cap and the book
    is intentionally under-invested rather than violating the limit.
    """
    w = w.clip(lower=0.0)
    s = w.sum()
    if s <= 0:
        return w
    w = w / s
    for _ in range(100):
        over = w > max_weight + 1e-12
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        room = w[under].sum()
        if room <= 0:
            break
        w[under] += excess * (w[under] / room)
    return w

def target_weights(
    scores: pd.Series,
    vol: pd.Series,
    *,
    top_quantile: float = 0.2,
    long_short: bool = False,
    sizing: str = "inv_vol",
    max_weight: float = 0.10,
    gross: float = 1.0,
    min_names: int = 5,
    prob: Optional[pd.Series] = None,
    prob_floor: float = 0.5,
) -> pd.Series:
    """Target portfolio weights for a single date from composite ``scores`` and per-name ``vol``.

    Longs the top ``top_quantile`` of ranked names (and shorts the bottom if ``long_short``). Sizing is
    inverse-vol (risk parity-ish) or equal; each name is capped at ``max_weight`` and the book is scaled
    to ``gross`` exposure (which the caller may shrink for risk-off regimes).

    **Meta-label overlay** (``prob`` = P(name beats peers next period): when given, low-conviction names
    (``prob < prob_floor``) are dropped *and* weights are tilted toward higher-probability names — but a
    ``min_names`` diversification floor is always respected (we never collapse the book onto 1-2 names,
    which a naive hard filter would do). For shorts the conviction is ``1 - prob``.

    Returns 0.0 for names not held; empty Series if fewer than ``min_names`` rankable names.
    """
    valid = scores.dropna()
    if len(valid) < min_names:
        return pd.Series(dtype=float)
    n_pick = max(min_names, int(round(len(valid) * top_quantile)))

    def _size(names: pd.Index) -> pd.Series:
        if sizing == "inv_vol":
            iv = (1.0 / vol.reindex(names)).replace([np.inf, -np.inf], np.nan).dropna()
            if not iv.empty:
                return (iv / iv.sum()).reindex(names).fillna(0.0)
        return pd.Series(1.0 / len(names), index=names)

    def _meta_select(cand: pd.Index, conviction: Optional[pd.Series]) -> pd.Index:
        """Drop below-floor names but keep at least ``min_names`` (best conviction) for diversification."""
        if conviction is None:
            return cand
        c = conviction.reindex(cand).fillna(0.5)
        keep = c[c >= prob_floor].index
        return keep if len(keep) >= min_names else c.nlargest(min(min_names, len(c))).index

    def _book(names: pd.Index, conviction: Optional[pd.Series]) -> pd.Series:
        names = _meta_select(names, conviction)
        w = _size(names)
        if conviction is not None:                          # tilt size toward higher conviction
            t = conviction.reindex(names).fillna(0.5).clip(lower=0.0)
            if t.sum() > 0:
                w = w * t
        return _cap_normalize(w, max_weight)                # true per-name cap (water-filled)

    w = pd.Series(0.0, index=scores.index)
    longs = valid.nlargest(n_pick).index
    if long_short:
        shorts = valid.nsmallest(n_pick).index
        w.loc[:] = w.add(0.5 * gross * _book(longs, prob), fill_value=0.0)
        short_conv = (1.0 - prob) if prob is not None else None
        w.loc[:] = w.add(-0.5 * gross * _book(shorts, short_conv), fill_value=0.0)
    else:
        w.loc[:] = w.add(gross * _book(longs, prob), fill_value=0.0)
    return w.reindex(scores.index).fillna(0.0)


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "W-FRI") -> List[pd.Timestamp]:
    """Trading dates on/closest-before each period end in ``freq`` (e.g. weekly Fri, monthly)."""
    s = pd.Series(index, index=index)
    # last available trading day within each period bucket
    picks = s.groupby(s.dt.to_period(_freq_to_period(freq))).max()
    return list(picks.values)


def _freq_to_period(freq: str) -> str:
    f = freq.upper()
    if f.startswith("W"):
        return "W"
    if f.startswith("M"):
        return "M"
    if f.startswith("Q"):
        return "Q"
    return "W"


# ── Performance metrics ──────────────────────────────────────────────────────────

def performance_metrics(returns: pd.Series, benchmark: Optional[pd.Series] = None,
                        rf: float = 0.0) -> Dict[str, float]:
    """Annualised performance stats for a daily return series (+ alpha/beta vs a benchmark)."""
    r = returns.dropna()
    out = {k: float("nan") for k in
           ("CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar", "HitRate", "ProfitFactor")}
    if len(r) < 2:
        return out
    equity = (1.0 + r).cumprod()
    years = len(r) / TRADING_DAYS
    total = float(equity.iloc[-1])
    out["CAGR"] = total ** (1.0 / years) - 1.0 if years > 0 and total > 0 else float("nan")
    out["Vol"] = float(r.std() * np.sqrt(TRADING_DAYS))
    out["Sharpe"] = float((r.mean() * TRADING_DAYS - rf) / out["Vol"]) if out["Vol"] > 0 else float("nan")
    downside = r[r < 0].std()
    out["Sortino"] = (float(r.mean() * TRADING_DAYS / (downside * np.sqrt(TRADING_DAYS)))
                      if downside and downside > 0 else float("nan"))
    dd = equity / equity.cummax() - 1.0
    out["MaxDD"] = float(dd.min())
    out["Calmar"] = float(out["CAGR"] / abs(out["MaxDD"])) if out["MaxDD"] < 0 else float("nan")
    out["HitRate"] = float((r > 0).mean())
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    out["ProfitFactor"] = float(gains / losses) if losses > 0 else float("nan")
    if benchmark is not None:
        joined = pd.concat([r, benchmark], axis=1).dropna()
        if len(joined) > 2:
            rr, b = joined.iloc[:, 0], joined.iloc[:, 1]
            if b.var() > 0:
                beta = float(np.cov(rr, b)[0, 1] / b.var())
                out["Beta"] = beta
                out["Alpha"] = float((rr.mean() - beta * b.mean()) * TRADING_DAYS)
    return out


# ── Anti-overfitting statistics (scipy-free) ─────────────────────────────────────

import math


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation) — avoids a scipy dependency."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > ``sr_benchmark``), correcting for sample length and non-normal returns.

    Sharpes here are *per-period* (not annualised). A fat left tail / negative skew lowers the
    probability — exactly the honesty a raw Sharpe hides. (Bailey & López de Prado, 2012.)
    """
    r = returns.dropna()
    n = len(r)
    sd = r.std()
    if n < 4 or sd == 0:
        return float("nan")
    sr = r.mean() / sd
    skew = float(((r - r.mean()) ** 3).mean() / sd ** 3)
    kurt = float(((r - r.mean()) ** 4).mean() / sd ** 4)          # raw kurtosis (normal = 3)
    denom = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2, 1e-12))
    return _norm_cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom)


def deflated_sharpe_ratio(main_returns: pd.Series, trial_returns: List[pd.Series]) -> Dict[str, float]:
    """Deflated Sharpe: PSR against the Sharpe you'd expect to hit *by luck* across N trials.

    Selecting the best of many configs inflates Sharpe; DSR deflates the benchmark to the expected
    maximum under the null, so a config that only looks good because you tried many will score low.
    Returns ``{"DSR", "SR0_ann", "n_trials"}``. SR0 is the annualised luck-benchmark Sharpe.
    """
    srs = []
    for r in trial_returns:
        r = r.dropna()
        if len(r) > 4 and r.std() > 0:
            srs.append(float(r.mean() / r.std()))
    n = len(srs)
    if n < 2:
        return {"DSR": probabilistic_sharpe_ratio(main_returns), "SR0_ann": 0.0, "n_trials": n}
    var = float(np.var(srs, ddof=1))
    gamma = 0.5772156649015329
    z1, z2 = _norm_ppf(1 - 1.0 / n), _norm_ppf(1 - 1.0 / (n * math.e))
    sr0 = math.sqrt(var) * ((1 - gamma) * z1 + gamma * z2)        # expected max Sharpe by luck
    return {"DSR": probabilistic_sharpe_ratio(main_returns, sr_benchmark=sr0),
            "SR0_ann": sr0 * math.sqrt(TRADING_DAYS), "n_trials": n}


# ── Scenario stress testing (geopolitical "what-ifs") ────────────────────────────

def factor_betas(holding_returns: pd.DataFrame, factor_returns: pd.DataFrame) -> pd.DataFrame:
    """OLS betas of each holding's daily returns on a set of factor return series (with intercept)."""
    fac = factor_returns.dropna()
    rows = {}
    for sym in holding_returns.columns:
        joined = pd.concat([holding_returns[sym], fac], axis=1).dropna()
        if len(joined) < 30:
            continue
        y = joined.iloc[:, 0].values
        X = np.column_stack([np.ones(len(joined)), joined.iloc[:, 1:].values])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rows[sym] = beta[1:]                                       # drop intercept
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(fac.columns))


def scenario_pnl(weights: pd.Series, betas: pd.DataFrame,
                 scenarios: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Estimated portfolio P&L for each scenario = Σ weightᵢ · βᵢ,f · shock_f (idiosyncratic ignored)."""
    w = weights.reindex(betas.index).fillna(0.0)
    port_beta = betas.mul(w, axis=0).sum(axis=0)                   # portfolio beta per factor
    rows = []
    for name, shocks in scenarios.items():
        pnl = float(sum(port_beta.get(f, 0.0) * s for f, s in shocks.items()))
        rows.append({"Scenario": name,
                     "Est. P&L %": pnl * 100.0,
                     "Shock": ", ".join(f"{f} {s:+.0%}" for f, s in shocks.items())})
    return pd.DataFrame(rows)


# ── The backtest ─────────────────────────────────────────────────────────────────

def run_backtest(
    px: pd.DataFrame,
    *,
    freq: str = "W-FRI",
    top_quantile: float = 0.2,
    long_short: bool = False,
    sizing: str = "inv_vol",
    max_weight: float = 0.10,
    cost_bps: float = 10.0,
    gross: float = 1.0,
    regime: Optional[pd.Series] = None,
    min_names: int = 5,
    vol_window: int = 21,
    factors: Optional[Dict] = None,
    composite: Optional[pd.DataFrame] = None,
    benchmark: Optional[pd.Series] = None,
    prob: Optional[pd.DataFrame] = None,
    prob_floor: float = 0.5,
    cost_model: str = "flat",
    volume: Optional[pd.DataFrame] = None,
    aum: float = 1_000_000.0,
    half_spread_bps: float = 2.0,
    impact_coef: float = 0.5,
    adv_window: int = 21,
) -> Dict:
    """Walk-forward, lookahead-free cross-sectional backtest over a price panel.

    Ranks on the composite factor score, longs the top quantile (+ shorts the bottom if requested),
    sizes inverse-vol with per-name caps, scales gross by ``regime`` (risk-off shrink), rebalances at
    ``freq``. If ``prob`` (an ML P(profit) panel) is given it tilts/filters via the meta-label overlay.

    **Costs.** ``cost_model="flat"`` charges ``cost_bps`` on turnover. ``cost_model="impact"`` (needs a
    ``volume`` panel) uses a square-root market-impact model *per name*: cost ≈ half-spread + ``impact_coef``
    · dailyσ · √(trade$ / ADV$), where trade$ = |Δweight|·``aum``. So a big book in thin names pays more —
    the honest, size-dependent cost institutions model (and the reason naive backtests overstate alpha).

    Returns a dict: ``equity``/``returns`` (net), ``gross_returns``, ``turnover``, ``metrics``,
    ``benchmark_metrics``, ``holdings`` (daily), ``latest_book`` (today's picks), ``rebalances``.
    """
    px = px.sort_index()
    rets = px.pct_change()
    vol = realized_vol(px, window=vol_window)
    if composite is None:
        composite = composite_score(px, factors)["composite"]

    holdings = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    book_by_date: Dict[pd.Timestamp, pd.Series] = {}
    for d in rebalance_dates(px.index, freq):
        d = pd.Timestamp(d)
        prob_row = prob.loc[d] if prob is not None else None
        g = gross * float(regime.get(d, 1.0)) if regime is not None else gross
        w = target_weights(composite.loc[d], vol.loc[d], top_quantile=top_quantile,
                            long_short=long_short, sizing=sizing, max_weight=max_weight, gross=g,
                            min_names=min_names, prob=prob_row, prob_floor=prob_floor)
        if w.empty:
            continue
        w = w.reindex(px.columns).fillna(0.0)
        book_by_date[d] = w
        holdings.loc[d] = w
        last_w = w

    # Carry weights forward between rebalances; trade next day (shift) → no lookahead.
    # (Only rebalance rows were assigned; everything else stayed NaN.)
    holdings = holdings.ffill().fillna(0.0)
    daily_w = holdings.shift(1).fillna(0.0)
    gross_ret = (daily_w * rets).sum(axis=1)

    # Costs: turnover is decided at the rebalance date but the trade (and slippage) executes the
    # next day, alongside the new weights — so charge it on the shifted axis to match.
    dW = (holdings - holdings.shift(1).fillna(0.0)).abs()
    turnover = dW.sum(axis=1)
    if cost_model == "impact" and volume is not None:
        adv_dollar = (px * volume).rolling(adv_window).mean().reindex_like(holdings)
        part = (dW * aum) / adv_dollar.where(adv_dollar > 0)        # participation rate per name
        dvol = (vol / np.sqrt(TRADING_DAYS)).reindex_like(holdings)  # daily σ for the sqrt-impact law
        impact_frac = (half_spread_bps / 1e4) + impact_coef * dvol * np.sqrt(part.clip(lower=0))
        cost_series = (dW * impact_frac.fillna(half_spread_bps / 1e4)).sum(axis=1)
    else:
        cost_series = turnover * (cost_bps / 1e4)
    cost = cost_series.shift(1).fillna(0.0)
    net_ret = (gross_ret - cost).loc[px.index]

    # Restrict to the live period (first day we actually held something).
    first = daily_w.abs().sum(axis=1)
    start = first[first > 0].index.min()
    if start is not None:
        net_ret = net_ret.loc[start:]
        gross_ret = gross_ret.loc[start:]

    bench = benchmark.reindex(net_ret.index) if benchmark is not None else rets.mean(axis=1).reindex(net_ret.index)
    result = {
        "returns": net_ret,
        "gross_returns": gross_ret,
        "equity": (1.0 + net_ret).cumprod(),
        "turnover": turnover.loc[net_ret.index] if start is not None else turnover,
        "metrics": performance_metrics(net_ret, benchmark=bench),
        "benchmark_metrics": performance_metrics(bench),
        "holdings": holdings,
        "rebalances": sorted(book_by_date.keys()),
    }
    result["latest_book"] = latest_book(px, composite, vol, top_quantile=top_quantile,
                                        long_short=long_short, sizing=sizing, max_weight=max_weight,
                                        gross=(gross * float(regime.iloc[-1]) if regime is not None
                                               and len(regime) else gross),
                                        min_names=min_names, prob=prob, prob_floor=prob_floor)
    return result


def latest_book(px: pd.DataFrame, composite: pd.DataFrame, vol: pd.DataFrame, **kw) -> pd.DataFrame:
    """Today's recommended holdings (the most recent cross-section) as a tidy table."""
    prob = kw.pop("prob", None)
    prob_floor = kw.pop("prob_floor", 0.5)
    d = px.index[-1]
    prob_row = prob.loc[d] if prob is not None else None
    w = target_weights(composite.loc[d], vol.loc[d], prob=prob_row, prob_floor=prob_floor, **kw)
    if w.empty:
        return pd.DataFrame(columns=["Symbol", "Side", "Weight", "Score", "Vol", "Price"])
    held = w[w != 0.0]
    rows = []
    for sym, weight in held.items():
        rows.append({"Symbol": sym, "Side": "LONG" if weight > 0 else "SHORT",
                     "Weight": float(weight), "Score": float(composite.loc[d, sym]),
                     "Vol": float(vol.loc[d, sym]) if pd.notna(vol.loc[d, sym]) else float("nan"),
                     "Price": float(px.loc[d, sym])})
    book = pd.DataFrame(rows).sort_values("Weight", key=lambda s: s.abs(), ascending=False)
    return book.reset_index(drop=True)
