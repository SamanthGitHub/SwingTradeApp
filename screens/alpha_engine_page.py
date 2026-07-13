"""Alpha Engine page (extracted verbatim from the pre-split app.py monolith)."""

from services import *  # noqa: F401,F403 — the shared data/scan/chart layer


def render(ctx) -> None:
    config = ctx.config
    account_size = ctx.account_size
    watchlist_mgr = ctx.watchlist_mgr
    st.header("🧠 Alpha Engine")
    mode = st.radio("View", ["🟢 Simple — what to buy", "🔬 Advanced — quant"],
                    horizontal=True, label_visibility="collapsed")
    advanced = mode.startswith("🔬")

    if advanced:
        st.caption("A quant long(/short) book: ranks a liquid universe on **momentum · trend · "
                   "52-week-high · low-volatility · reversal**, sizes inverse-vol with caps, tilts by "
                   "an **ML meta-label** and gates by a **VIX regime**. Today's book + a "
                   "**lookahead-free** backtest. Educational, not advice; survivorship bias remains.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            uni_n = st.slider("Universe size", 20, len(ALPHA_UNIVERSE), 60, 10)
        with c2:
            years = st.slider("Backtest years", 2, 10, 6, 1)
        with c3:
            freq_label = st.selectbox("Rebalance", ["Weekly", "Monthly"], index=0)
        with c4:
            topq = st.slider("Long top %", 5, 50, 25, 5)
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            ls = st.checkbox("Long / Short", value=False, help="Also short the bottom quantile")
        with a2:
            use_ml = st.checkbox("ML meta-label", value=True, help="P(beat peers) tilt + gentle filter")
        with a3:
            use_regime = st.checkbox("VIX regime gating", value=True, help="Cut gross when VIX is high")
        with a4:
            sector_neutral = st.checkbox("Sector-neutral", value=False,
                                         help="Rank vs sector peers — strips out sector bets")
        with st.expander("⚙️ Institutional realism — impact costs · overfitting · stress"):
            r1, r2, r3 = st.columns(3)
            with r1:
                cost_mode = st.radio("Cost model", ["ADV market-impact", "Flat bps"], index=0,
                                     help="ADV impact scales cost with the % of daily volume traded")
            with r2:
                if cost_mode.startswith("ADV"):
                    aum_m = st.number_input("Book size ($M)", 0.1, 5000.0, 5.0, 0.5)
                    cost = 10.0
                else:
                    cost = float(st.slider("Cost (bps/side)", 0, 50, 10, 5))
                    aum_m = 5.0
            with r3:
                show_overfit = st.checkbox("Anti-overfitting (PSR/DSR/PBO)", value=True)
                show_stress = st.checkbox("Geopolitical stress tests", value=True)
            f1, f2 = st.columns(2)
            with f1:
                factor_set_label = st.selectbox(
                    "Factor set", ["Core (5 factors)", "Enhanced (+ residual momentum · low idio-vol)"],
                    index=0, help="Enhanced adds market-neutral factors computed from the same data")
            with f2:
                snap_choice = st.selectbox("Data source (reproducibility)",
                                           ["Live (latest)"] + datalake.list_snapshots("alpha"),
                                           help="Pin a dated snapshot to re-run on identical data")
    else:
        st.caption("Tell me your budget and I'll show **which stocks to buy, and how much**, this "
                   "month — from a rigorously back-tested model. Educational, not financial advice.")
        account = st.number_input("How much do you want to invest? ($)", 500, 100_000_000,
                                  10_000, 500)
        # Sensible defaults — the quant knobs are hidden in Simple mode.
        uni_n, years, freq_label, topq = 60, 6, "Weekly", 25
        ls, use_ml, use_regime, sector_neutral = False, True, True, False
        cost_mode, aum_m, cost = "ADV market-impact", max(account / 1e6, 0.1), 10.0
        show_overfit, show_stress = False, False
        factor_set_label, snap_choice = "Core (5 factors)", "Live (latest)"

    cost_model = "impact" if cost_mode.startswith("ADV") else "flat"
    factors = alpha_factors.ENHANCED_FACTORS if factor_set_label.startswith("Enhanced") else None

    if not scan_gate("alpha", RECOMMENDED_TIMES["Alpha Engine"],
                     clear=lambda: (fetch_price_panel.clear(), fetch_close_series.clear(),
                                    fetch_volume_panel.clear())):
        st.stop()

    universe = tuple(ALPHA_UNIVERSE[:uni_n])
    if snap_choice != "Live (latest)":
        prices = datalake.load_snapshot(snap_choice)
        if prices is None or prices.empty:
            st.warning("Snapshot unavailable — using live data.")
            prices = fetch_price_panel(universe, years)
        elif advanced:
            st.caption(f"🗄️ Pinned snapshot **{snap_choice}** · {prices.shape[1]} names · "
                       f"{prices.index.min().date()}–{prices.index.max().date()} (reproducible).")
    else:
        with st.spinner("Loading market data…"):
            prices = fetch_price_panel(universe, years)
        if _PANEL_SKIPPED_CHUNKS["n"]:
            st.caption(f"⚠ {_PANEL_SKIPPED_CHUNKS['n']} download chunk(s) failed after a retry "
                       "and were skipped — the panel may be missing some names.")
        if advanced and not prices.empty and prices.shape[1] >= 10:
            age = datalake.panel_age_hours(datalake.panel_key("prices", universe, years))
            ds1, ds2 = st.columns([3, 1])
            with ds1:
                st.caption(f"🗄️ Data lake · {prices.shape[1]} names · "
                           f"{prices.index.min().date()}–{prices.index.max().date()}"
                           + (f" · cached {age:.0f}h ago" if age and age > 0.1 else " · fresh"))
            with ds2:
                if st.button("📸 Pin snapshot", use_container_width=True,
                             help="Save this exact data for a reproducible re-run"):
                    nm = datalake.save_snapshot(prices, f"alpha_{uni_n}x{years}y")
                    st.success(f"Pinned {nm}" if nm else "Snapshot failed")
    if prices.empty or prices.shape[1] < 10:
        st.warning("Couldn't load enough market data right now — please try again in a moment.")
        st.stop()

    freq = "W-FRI" if freq_label == "Weekly" else "M"
    horizon = 5 if freq_label == "Weekly" else 21

    with st.spinner("Computing factors & cross-sectional ranking…"):
        panels = alpha_factors.composite_score(prices, factors=factors)
    composite_used = panels["composite"]
    if sector_neutral:
        composite_used = alpha_factors.sector_neutralize(composite_used, ALPHA_SECTORS)

    prob = None
    if use_ml:
        with st.spinner("Training walk-forward meta-label (out-of-sample)…"):
            prob = alpha_ml.walkforward_proba(
                prices, panels, alpha_engine.rebalance_dates(prices.index, freq), horizon=horizon)
        if prob is None:
            st.caption("ℹ️ ML meta-label unavailable (scikit-learn or insufficient history) — "
                       "running factor-only.")

    bench = fetch_close_series("SPY", years).reindex(prices.index).pct_change()
    regime = None
    if use_regime:
        vix = fetch_close_series("^VIX", years).reindex(prices.index).ffill()
        if not vix.dropna().empty:
            regime = _vix_regime(vix).reindex(prices.index).ffill().fillna(1.0)

    volume = None
    if cost_model == "impact":
        volume = fetch_volume_panel(tuple(prices.columns), years).reindex(index=prices.index,
                                                                         columns=prices.columns)
        if volume.empty or volume.isna().all().all():
            cost_model, volume = "flat", None
            st.caption("ℹ️ Volume feed unavailable — falling back to flat costs.")

    bt_kw = dict(freq=freq, top_quantile=topq / 100.0, long_short=ls, regime=regime,
                 cost_model=cost_model, cost_bps=float(cost), volume=volume, aum=aum_m * 1e6)
    with st.spinner("Running lookahead-free walk-forward backtest…"):
        res = alpha_engine.run_backtest(prices, composite=composite_used, prob=prob,
                                        prob_floor=0.5, benchmark=bench, **bt_kw)

    m, bm = res["metrics"], res["benchmark_metrics"]

    def _f(d, k):
        v = d.get(k)
        return v if v is not None and v == v else float("nan")

    # Robustness stats — needed for the Simple confidence line and the Advanced "edge real?" panel.
    psr, dsr, pbo = float("nan"), None, None
    if (not advanced) or show_overfit:
        with st.spinner("Checking robustness across configurations…"):
            trials, cols = [], {}
            for cfg in [{"freq": fr2, "top_quantile": tq2}
                        for tq2 in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35) for fr2 in ("W-FRI", "M")]:
                rr = alpha_engine.run_backtest(prices, composite=composite_used,
                                               **{**bt_kw, **cfg})["returns"]
                trials.append(rr)
                cols[f"{cfg['freq']}_{int(cfg['top_quantile']*100)}"] = rr
            psr = alpha_engine.probabilistic_sharpe_ratio(res["returns"])
            dsr = alpha_engine.deflated_sharpe_ratio(res["returns"], trials)
            pbo = alpha_validation.pbo_cscv(pd.DataFrame(cols), n_blocks=10)

    _book_syms = list(res["latest_book"]["Symbol"]) if not res["latest_book"].empty else []
    reasons = alpha_factor_reasons(panels, _book_syms)

    if not advanced:
        render_alpha_simple_plan(res, account, m, bm, dsr, pbo, reasons)
    else:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("CAGR", f"{_f(m,'CAGR'):.2%}", f"{_f(m,'CAGR')-_f(bm,'CAGR'):+.2%} vs SPY")
        k2.metric("Sharpe", f"{_f(m,'Sharpe'):.2f}", f"SPY {_f(bm,'Sharpe'):.2f}")
        k3.metric("Max DD", f"{_f(m,'MaxDD'):.2%}", f"SPY {_f(bm,'MaxDD'):.2%}")
        k4.metric("Alpha (ann)", f"{_f(m,'Alpha'):.2%}")
        k5.metric("Beta", f"{_f(m,'Beta'):.2f}")

        eq = res["equity"]
        beq = (1.0 + bench.reindex(eq.index).fillna(0.0)).cumprod()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Alpha Engine",
                                 line=dict(color="#00c851", width=2)))
        fig.add_trace(go.Scatter(x=beq.index, y=beq.values, name="SPY (buy & hold)",
                                 line=dict(color="#888888", width=1, dash="dash")))
        fig.update_layout(
            height=400, margin=dict(t=48, b=70, l=10, r=10), hovermode="x unified",
            title=dict(text="Out-of-sample growth of $1 (net of costs)", x=0.5, xanchor="center",
                       y=0.96, yanchor="top"),
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True, key="alpha_equity")

        with st.expander("All metrics — strategy vs SPY"):
            order = ["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar", "HitRate",
                     "ProfitFactor", "Alpha", "Beta"]
            mt = pd.DataFrame({"Strategy": {k: _f(m, k) for k in order},
                               "SPY": {k: _f(bm, k) for k in order}})
            st.dataframe(mt.style.format("{:.3f}", na_rep="—"), use_container_width=True)
            yrs = max(len(res["returns"]) / 252.0, 1e-9)
            cost_txt = (f"ADV market-impact @ ${aum_m:.1f}M book" if cost_model == "impact"
                        else f"flat {cost:.0f} bps/side")
            st.caption(f"Rebalances: {len(res['rebalances'])} · turnover "
                       f"{res['turnover'].sum()/yrs:.1f}x/yr · live ~{yrs:.1f}y · costs: {cost_txt}"
                       + (" · sector-neutral" if sector_neutral else ""))

        if show_overfit and dsr is not None:
            st.subheader("🧪 Is the edge real?")
            o1, o2, o3 = st.columns(3)
            o1.metric("Probabilistic Sharpe", f"{psr:.1%}",
                      help="P(true Sharpe > 0), adjusted for sample length and non-normal returns")
            o2.metric("Deflated Sharpe", f"{dsr['DSR']:.1%}",
                      help=f"PSR vs the Sharpe expected by luck across {dsr['n_trials']} configs tried "
                           f"(luck-benchmark ≈ {dsr['SR0_ann']:.2f} ann.)")
            o3.metric("Overfit prob. (PBO)", f"{pbo['PBO']:.1%}",
                      help=f"P(best in-sample config is below-median OOS), CSCV over "
                           f"{pbo['n_configs']} configs × {pbo['n_splits']} splits. Lower is better.")
            robust = dsr["DSR"] >= 0.95 and pbo["PBO"] <= 0.35
            shaky = dsr["DSR"] < 0.80 or (pbo["PBO"] > 0.50 and dsr["DSR"] < 0.95)
            verdict = ("🟢 Edge looks robust" if robust else
                       "🔴 Likely overfit / not significant" if shaky else
                       "🟡 Real edge, but config-sensitive")
            st.caption(
                f"{verdict}. **Deflated Sharpe** = is the Sharpe real after correcting for how many "
                "configs were tried (higher better). **PBO** = chance the best-looking config is "
                "below-median out-of-sample (lower better). High DSR **and** high PBO means the "
                "*edge is real but the exact quantile/rebalance is mostly noise*.")

        if show_stress:
            book0 = res["latest_book"]
            held = ([s for s in book0["Symbol"].tolist() if s in prices.columns]
                    if not book0.empty else [])
            if held:
                with st.spinner("Stress-testing today's book against macro shocks…"):
                    w = book0.set_index("Symbol")["Weight"]
                    fac_ret = pd.DataFrame({
                        k: fetch_close_series(tk, years).reindex(prices.index).pct_change()
                        for k, tk in ALPHA_FACTOR_PROXIES.items()})
                    betas = alpha_engine.factor_betas(prices[held].pct_change(), fac_ret)
                    stress = alpha_engine.scenario_pnl(w, betas, ALPHA_SCENARIOS)
                st.subheader("🌍 Geopolitical stress tests (today's book)")

                def _pnl_c(v):
                    try:
                        return "color:#00c851" if float(v) > 0 else ("color:#ff4444" if float(v) < 0 else "")
                    except (TypeError, ValueError):
                        return ""

                st.dataframe(
                    stress.style.format({"Est. P&L %": "{:+.2f}%"}).map(_pnl_c, subset=["Est. P&L %"]),
                    use_container_width=True, hide_index=True)
                st.caption("Estimated one-day P&L if each shock hit, from the book's betas to SPY · "
                           "oil (CL=F) · long bonds (TLT) · USD (UUP). First-order — a planning aid, "
                           "**not** a prediction.")

        st.subheader("📋 Today's book")
        book = res["latest_book"]
        if book.empty:
            st.caption("No rankable book for the latest session.")
        else:
            gross_txt = f"gross {book['Weight'].abs().sum():.0%}"
            net_txt = f" · net {book['Weight'].sum():+.0%}" if ls else ""
            st.caption(f"{len(book)} positions · {gross_txt}{net_txt} · sized inverse-vol, capped, "
                       f"{'meta-label tilted' if prob is not None else 'factor-only'}"
                       f"{', regime-scaled' if regime is not None else ''}.")
            book = book.copy()
            book["Why"] = book["Symbol"].map(reasons)
            st.dataframe(
                book.style.format({"Weight": "{:.2%}", "Score": "{:.2f}", "Vol": "{:.2%}",
                                   "Price": "${:.2f}"}, na_rep="—")
                    .map(lambda v: "color:#00c851;font-weight:bold" if v == "LONG"
                         else "color:#ff4444;font-weight:bold", subset=["Side"]),
                use_container_width=True, hide_index=True, height=min(460, 80 + 34 * len(book)))
            st.download_button("Download book CSV", book.to_csv(index=False),
                               file_name=f"alpha_book_{datetime.now():%Y%m%d}.csv", mime="text/csv")

        with st.expander("ℹ️ How this works / how to read it"):
            st.markdown(
                "- **Factors** (all 'higher = more long-worthy'): 12-1 momentum, price-vs-200d trend, "
                "proximity to the 52-week high, *low* realized volatility, and short-term reversal. "
                "Each is z-scored across names every day, then blended into one **composite**.\n"
                "- **Book**: longs the top *N%* of the composite (and shorts the bottom if Long/Short), "
                "sizes **inverse-volatility**, caps any single name, scales exposure by the **VIX "
                "regime**.\n"
                "- **ML meta-label**: a walk-forward classifier estimates P(a pick beats peers next "
                "period) and tilts/filters the book.\n"
                "- **Backtest**: rebalanced weekly/monthly, **no lookahead**, net of costs. Alpha/Beta "
                "vs SPY.\n"
                "- **Reality check**: currently-listed universe → the backtest still flatters "
                "(survivorship). A rigorous *relative* tool, not a promise.")
