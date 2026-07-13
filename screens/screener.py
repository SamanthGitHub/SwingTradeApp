"""Screener page (extracted verbatim from the pre-split app.py monolith)."""

from services import *  # noqa: F401,F403 — the shared data/scan/chart layer


def render(ctx) -> None:
    config = ctx.config
    account_size = ctx.account_size
    watchlist_mgr = ctx.watchlist_mgr
    st.header("Advanced Stock Screener")

    # ── Look up any ticker (independent of the scan) ─────────────────────
    sc1, sc2 = st.columns([3, 1])
    with sc1:
        search_ticker = st.text_input("🔍 Look up any ticker", placeholder="e.g. AAPL, NVDA, TSLA",
                                      key="screener_search").strip().upper()
    with sc2:
        st.write("")
        do_search = st.button("Analyze", use_container_width=True)
    if search_ticker and (do_search or st.session_state.get("last_search") == search_ticker):
        st.session_state["last_search"] = search_ticker
        with st.spinner(f"Analyzing {search_ticker}…"):
            render_ticker_analysis(search_ticker, config, account_size)
        _render_legend()
        st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sample_size = st.number_input("Universe size", 10, 500, 100, 10)
    with c2:
        min_score = st.slider("Min signal score", 0.3, 1.0, 0.45, 0.05)
    with c3:
        allocation_scale = st.slider("Allocation scale", 0.25, 4.0, 1.0, 0.25)
    with c4:
        include_bt = st.checkbox("Include backtest metrics", value=False,
                                 help="Slower — runs per-symbol backtest to show win rate/Sharpe")

    with st.expander("Advanced Filters"):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            mc_input = st.number_input("Min market cap ($B)", 0.0, 5000.0, 0.0, 1.0)
            min_market_cap = mc_input * 1e9 if mc_input > 0 else None
        with fc2:
            max_pe = st.number_input("Max P/E ratio", 0.0, 500.0, 0.0, 5.0)
        with fc3:
            min_div = st.number_input("Min dividend yield (%)", 0.0, 20.0, 0.0, 0.5) / 100
        with fc4:
            sectors = st.multiselect("Sectors", [
                "Technology", "Healthcare", "Financials", "Energy",
                "Consumer Cyclical", "Industrials", "Materials", "Utilities",
            ])
        filters = {
            "min_market_cap": min_market_cap,
            "max_pe": max_pe if max_pe > 0 else None,
            "min_dividend_yield": min_div if min_div > 0 else None,
            "sectors": sectors if sectors else None,
        }

    if not scan_gate("screener", RECOMMENDED_TIMES["Screener"],
                     clear=lambda: (get_screen_universe.clear(), calibrate_kelly_priors.clear())):
        st.stop()

    tickers = get_screen_universe()
    scan_universe = tickers[:sample_size]
    prefetch_histories(list(dict.fromkeys(scan_universe + tickers[:30])), 300, "screener data")

    # Calibrate Kelly priors once from out-of-sample walk-forward edge (cached 1h).
    with st.spinner("Calibrating position sizer (out-of-sample)…"):
        priors = calibrate_kelly_priors(config)
    if priors:
        get_sizer(config).update_from_backtest(
            win_rate=priors["win_rate"],
            avg_win_pct=priors["avg_win"],
            avg_loss_pct=priors["avg_loss"],
            n_trades=priors["trades"],
        )
        st.caption(f"Sizer calibrated on {priors['trades']} out-of-sample trades · "
                   f"win rate {priors['win_rate']:.2%} · "
                   f"avg win {priors['avg_win']:.2%} / avg loss {priors['avg_loss']:.2%}")
    else:
        st.caption("Sizer using default priors (insufficient out-of-sample trades to calibrate).")

    if _ai_on("ai_ml_signal") and not ML_MODEL_PATH.exists():
        prefetch_histories(tickers[:40], 400, "ML training data")
    ml_model = get_ml_signal_model() if _ai_on("ai_ml_signal") else None
    if _ai_on("ai_ml_signal") and ml_model is None:
        st.caption("ℹ️ ML signal model unavailable (training needs more data, or scikit-learn "
                   "is missing) — using the rule-based score only.")

    _t0 = time.perf_counter()
    with st.spinner(f"Scanning {len(scan_universe)} symbols…"):
        df = build_signal_table(scan_universe, config, min_score, allocation_scale,
                                account_size, filters, include_backtest=include_bt,
                                ml_model=ml_model)
    st.caption(f"⏱ Scanned {len(scan_universe)} symbols in {time.perf_counter() - _t0:.1f}s")

    if df.empty:
        st.warning("No signals passed filters. Lower the min score or expand the universe.")
    else:
        # Summary bar
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Signals", len(df))
        m2.metric("Strong (≥0.7)", int((df["score"] >= 0.7).sum()))
        m3.metric("Avg Score", f"{df['score'].mean():.2f}")
        m4.metric("Avg R:R", f"{df['risk_reward'].mean():.1f}x")
        m5.metric("Avg Sentiment", f"{df['sentiment_pct'].mean():.2f}%")

        # Company info on hover — a strip of ticker pills above the table. Reuses the cached
        # fetch_symbol_info the scan already populated (cache hit, no extra network calls).
        _top = df.sort_values("score", ascending=False).head(40)
        _profiles = []
        for _, _r in _top.iterrows():
            _info = fetch_symbol_info(_r["symbol"])
            _profiles.append({
                "symbol": _r["symbol"],
                "name": _info.get("name"),
                "sector": _r.get("sector") or _info.get("sector"),
                "industry": _info.get("industry"),
                "market_cap": _r.get("market_cap") or _info.get("market_cap"),
                "change_pct": _r.get("change_pct"),
                "summary": _info.get("summary"),
            })
        st.caption("💡 Hover a ticker for company info")
        ui.render_ticker_hovercards(_profiles)

        # Display table
        has_ml = "ml_prob" in df.columns and df["ml_prob"].notna().any()
        disp_cols = ["symbol", "price", "change_pct", "recommendation", "score"]
        if has_ml:
            disp_cols.append("ml_prob")
        disp_cols += ["rsi", "macd_hist", "vol_surge", "entry", "stop", "target",
                      "risk_reward", "allocation_pct", "allocation_usd", "sentiment_pct",
                      "sector"]
        if include_bt:
            disp_cols += ["bt_win_rate", "bt_profit_factor", "bt_sharpe"]

        disp = df[disp_cols].sort_values("score", ascending=False).copy()
        if has_ml:
            disp = disp.rename(columns={"ml_prob": "ML P(up)"})

        fmt = {
            "price": "${:.2f}", "change_pct": "{:.2f}%", "score": "{:.2f}",
            "rsi": "{:.1f}", "macd_hist": "{:.4f}", "vol_surge": "{:.1f}x",
            "entry": "${:.2f}", "stop": "${:.2f}", "target": "${:.2f}",
            "risk_reward": "{:.2f}x", "allocation_pct": "{:.2f}%",
            "allocation_usd": "${:,.0f}", "sentiment_pct": "{:.2f}%",
        }
        if has_ml:
            fmt["ML P(up)"] = "{:.0%}"
        if include_bt:
            fmt.update({"bt_win_rate": "{:.2%}", "bt_profit_factor": "{:.2f}", "bt_sharpe": "{:.2f}"})

        styler = disp.style.format(fmt, na_rep="—").map(_reco_color, subset=["recommendation"])
        if has_ml:
            styler = styler.map(_ml_prob_color, subset=["ML P(up)"])

        st.dataframe(styler, use_container_width=True, height=420)
        if has_ml:
            _m = (get_ml_signal_model() or MLSignalModel()).metrics
            if _m:
                st.caption(f"🤖 **ML P(up)** = calibrated, walk-forward-trained probability of an "
                           f"up move over ~{(get_ml_signal_model().horizon)} sessions "
                           f"(OOS AUC {_m.get('auc', float('nan')):.2f}, "
                           f"Brier {_m.get('brier', float('nan')):.2f}). It also drives Kelly sizing.")

        # Export
        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            file_name=f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        _render_legend()

        # Ticker drill-down
        st.markdown("---")
        selected = st.selectbox("Inspect ticker", df["symbol"].tolist())
        if selected:
            row = df[df["symbol"] == selected].iloc[0]
            st.plotly_chart(create_price_chart(selected, signal_row=row),
                            use_container_width=True, key=f"drill_price_{selected}")
            reco = row.get("recommendation", "—")
            st.markdown(f"**Recommendation:** <span style='{_reco_color(reco)}'>{reco}</span> "
                        f"· score {row['score']:.2f}", unsafe_allow_html=True)
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Entry", f"${row['entry']:.2f}")
            dc2.metric("Stop", f"${row['stop']:.2f}", delta=f"{(row['stop']-row['entry'])/row['entry']*100:.2f}%")
            dc3.metric("Target", f"${row['target']:.2f}", delta=f"+{(row['target']-row['entry'])/row['entry']*100:.2f}%")
            _ln = _limit_note(row.get("entry"), row.get("price"))
            if _ln:
                st.caption(_ln)
            st.write("**Signal reasons:**", row.get("reasons", ""))

            st.markdown(f"#### 📰 News for {selected}")
            render_ticker_news(selected, config)
            render_forecast_panel(selected, config, entry=row["entry"], stop=row["stop"],
                                  key_prefix="drill")

            # One-click log to journal
            st.markdown("**Log this trade:**")
            j1, j2, j3 = st.columns(3)
            with j1:
                log_qty = st.number_input("Qty", min_value=1, value=max(1, int(row["allocation_usd"] / row["entry"])), key="log_qty")
            with j2:
                if st.button("Add to P&L Journal", use_container_width=True):
                    journal = load_journal()
                    add_trade(journal, selected, "long",
                              float(row["entry"]), float(row["stop"]),
                              float(row["target"]), log_qty, float(row["score"]))
                    save_journal(journal)
                    st.success(f"Logged {selected} to journal")
            with j3:
                st.write(f"Alloc: ${row['allocation_usd']:,.0f}  R:R {row['risk_reward']:.1f}x")

        st.session_state["signals_df"] = df

    # ── Overall market news (ticker-independent) ─────────────────────────
    st.markdown("---")
    st.subheader("📰 Overall market news")
    with st.spinner("Loading market news…"):
        render_market_news(config)
