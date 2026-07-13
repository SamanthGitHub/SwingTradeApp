"""SwingTrade Pro — Streamlit entry point: theme, nav and page routing.

The heavy lifting lives in ``services.py`` (data/scan/chart layer) and ``screens/``
(page renderers, migrating out of the ``elif`` chain below one page at a time).
"""

from services import *  # noqa: F401,F403 — the shared data/logic layer

import screens


def run_dashboard() -> None:
    st.set_page_config(
        page_title="SwingTrade Pro Dashboard",
        page_icon="📈",
        layout="wide",
        # "auto" = expanded on desktop, collapsed on mobile (so the nav doesn't overlay the page).
        # Safe now that the » reopen control is always visible (see ui.py).
        initial_sidebar_state="auto",
    )

    ui.inject_theme()
    ui.render_header()

    # On hosted deploys (e.g. Streamlit Community Cloud) secrets live in st.secrets, but the config
    # reads os.environ — copy them across so ALPACA_*/cost overrides take effect. No-op locally /
    # when no secrets are defined, so the app stays in offline mode.
    try:
        for _k, _v in st.secrets.items():
            os.environ.setdefault(_k, str(_v))
    except Exception:
        pass

    config = get_config()
    # Apply live cost-model overrides set on the Settings page (survive cache clears).
    if "slippage_bps" in st.session_state:
        config.slippage_bps = st.session_state["slippage_bps"]
        config.commission_bps = st.session_state["commission_bps"]
    watchlist_mgr = get_watchlist_manager()

    page = ui.render_nav()

    # Reset scan gates whenever the page changes, so each screen waits for an explicit
    # ▶ Scan now on every visit instead of auto-running.
    if st.session_state.get("_active_page") != page:
        for _k in [k for k in list(st.session_state) if k.startswith("_scan_")]:
            del st.session_state[_k]
        st.session_state["_active_page"] = page

    with st.sidebar:
        st.divider()
        account_size = float(st.number_input("Account size ($)", value=100_000, step=10_000,
                                             min_value=1000))
        # Free-tier API budget status — only shown for providers you've enabled.
        for _pk, _spec in PROVIDERS.items():
            if _provider_on(_pk) and _spec.per_minute:
                _stt = get_api_budget().status(_pk)
                _used = int(_stt.get("used_minute", 0))
                _dot = "🟢" if _used < 0.8 * _spec.per_minute else ("🟡" if _used < _spec.per_minute else "🔴")
                st.caption(f"{_dot} {_spec.name}: {_used}/{_spec.per_minute} calls/min")

    # Data-source + timestamp strip on every screen (free vs paid, and data freshness).
    render_data_status()

    # ═══════════════════════ MIGRATED PAGES (screens/ package) ═══════════════
    # Screener, Morning Insights, YouTube, Signal Stack, Alpha Engine and Settings
    # live in screens/*.py now; the elif chain below migrates there one page at a time.
    handler = screens.PAGES.get(page)
    if handler is not None:
        handler(screens.Ctx(config=config, account_size=account_size,
                            watchlist_mgr=watchlist_mgr))

    # ═══════════════════════ RALLY RADAR (EARLY MOMENTUM) ════════════════════
    elif page == "Rally Radar":
        st.header("🚀 Rally Radar")
        st.caption("Stocks whose momentum is *igniting* — about to start a move, not ones that "
                   "already ran. Looks for the pre-breakout setup: a volatility squeeze, volume "
                   "starting to build, a MACD/RSI turn, a 20-day-MA reclaim, and price pressing "
                   "the top of its base. Each gets a 0–100 **Rally Readiness** score and a stage: "
                   "**Coiling** → **Igniting** → **Breaking out**.")

        rr1, rr2, rr3 = st.columns(3)
        with rr1:
            rr_n = st.slider("Universe size", 40, 250, 150, 10,
                             help="Most-active names are scanned first")
        with rr2:
            rr_min = st.slider("Min Rally Readiness", 30, 80, 45, 5,
                               help="Higher = only the strongest setups")
        with rr3:
            rr_stage = st.selectbox("Stage", ["All", "Coiling", "Igniting", "Breaking out"],
                                    help="Coiling = earliest (still quiet); Breaking out = latest")

        rr_penny = st.checkbox("Include sub-$5 names", value=False, key="rr_penny",
                               help="Penny stocks are skipped by default — thin books make their signals unreliable")
        if not scan_gate("rallyradar", RECOMMENDED_TIMES["Rally Radar"], clear=scan_rally_radar.clear):
            st.stop()
        prefetch_histories(get_screen_universe()[:rr_n], 160, "rally-radar data")
        _t0 = time.perf_counter()
        with st.spinner("Scanning for igniting momentum…"):
            rally = scan_rally_radar(sample_size=rr_n, min_score=float(rr_min),
                                     min_price=0.0 if rr_penny else 5.0)
        st.caption(f"⏱ Scanned {rr_n} symbols in {time.perf_counter() - _t0:.1f}s")

        if rally.empty:
            st.info("No early-momentum setups cleared the bar right now. Lower the minimum score "
                    "or widen the universe — quiet tapes simply have fewer coiled springs.")
        else:
            view = rally if rr_stage == "All" else rally[rally["stage"] == rr_stage]
            if view.empty:
                st.info(f"No '{rr_stage}' setups right now — try another stage or 'All'.")
            else:
                rm1, rm2, rm3, rm4 = st.columns(4)
                rm1.metric("Setups", len(view))
                rm2.metric("Coiling", int((rally["stage"] == "Coiling").sum()))
                rm3.metric("Igniting", int((rally["stage"] == "Igniting").sum()))
                rm4.metric("Breaking out", int((rally["stage"] == "Breaking out").sum()))

                disp = view.rename(columns={
                    "symbol": "Symbol", "rally_score": "Readiness", "stage": "Stage",
                    "price": "Price", "change_pct": "Change %", "rsi": "RSI",
                    "rvol5": "Rel Vol", "bb_pctile": "Squeeze %ile",
                    "dist_to_high_pct": "% to High", "reasons": "Why",
                })[["Symbol", "Readiness", "Stage", "Price", "Change %", "RSI", "Rel Vol",
                    "Squeeze %ile", "% to High", "Why"]]

                def _stage_color(v: str) -> str:
                    return {"Breaking out": "color:#00c851;font-weight:bold",
                            "Igniting": "color:#00a843;font-weight:bold",
                            "Coiling": "color:#0b6cad"}.get(v, "color:#888888")

                def _chg_color(v):
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return ""
                    return "color:#00c851" if v > 0 else ("color:#ff4444" if v < 0 else "")

                st.dataframe(
                    disp.style.format({
                        "Readiness": "{:.1f}", "Price": "${:.2f}", "Change %": "{:+.2f}%",
                        "RSI": "{:.0f}", "Rel Vol": "{:.2f}x", "Squeeze %ile": "{:.0f}",
                        "% to High": "{:.2f}%",
                    }, na_rep="—")
                    .map(_stage_color, subset=["Stage"])
                    .map(_chg_color, subset=["Change %"]),
                    use_container_width=True, height=520,
                )

                st.download_button(
                    "Download CSV", view.to_csv(index=False),
                    file_name=f"rally_radar_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )

                with st.expander("How to read Rally Radar"):
                    st.markdown(
                        "- **Readiness (0–100)** — how many early-rally signals are lining up "
                        "(squeeze + volume + MACD/RSI turn + MA reclaim + base breakout).\n"
                        "- **Stage** — **Coiling** (tight, quiet base, hasn't moved yet — earliest "
                        "and least confirmed) → **Igniting** (momentum turning up, volume building) "
                        "→ **Breaking out** (pressing the recent high with volume).\n"
                        "- **Squeeze %ile** — where today's Bollinger-band *width* sits vs its own "
                        "recent range. **Low = tightly coiled** (a quiet base that tends to precede "
                        "an expansion move).\n"
                        "- **Rel Vol** — last 5 days' volume ÷ the 20-day average (>1 = building).\n"
                        "- **% to High** — distance below the 20-day high (small = pressing "
                        "resistance / about to break out).\n"
                        "- **Why** — the specific signals that fired for that name.\n\n"
                        "These are *early* setups — higher reward but less confirmed than the "
                        "Screener's established trends. Pair with the Screener and your own "
                        "risk plan; this is not advice."
                    )

    # ═══════════════════════ PRE-MARKET MOVERS ═══════════════════════════════
    elif page == "Pre-Market Movers":
        st.header("Pre-Market Movers")
        st.caption("Biggest gainers and losers right now (pre-market quotes when the pre-market "
                   "session is open, otherwise the regular session). Live from Yahoo Finance.")

        mc1, mc2 = st.columns(2)
        with mc1:
            mv_n = st.slider("Show top", 10, 50, 25, 5)
        with mc2:
            mv_min = st.slider("Min move %", 0.0, 10.0, 1.0, 0.5)

        if not scan_gate("premarket", RECOMMENDED_TIMES["Pre-Market Movers"], clear=fetch_movers.clear):
            st.stop()
        with st.spinner("Fetching movers…"):
            mv = fetch_movers(top_n=mv_n, min_change_pct=mv_min)

        if mv.empty:
            st.warning("No movers returned — the market data feed may be unavailable right now.")
        else:
            gainers = mv[mv["change_pct"] > 0]
            losers = mv[mv["change_pct"] < 0]
            s1, s2, s3 = st.columns(3)
            s1.metric("Gainers", len(gainers))
            s2.metric("Losers", len(losers))
            s3.metric("Session", mv["session"].mode().iloc[0] if not mv.empty else "—")

            disp = mv.rename(columns={
                "symbol": "Symbol", "name": "Name", "change_pct": "Move %",
                "session": "Session", "price": "Price", "regular_change_pct": "Reg %",
                "volume": "Volume", "market_cap": "Mkt Cap",
            })[["Symbol", "Name", "Move %", "Session", "Price", "Reg %", "Volume", "Mkt Cap"]]

            def _hl(v):
                return "color:#00c851" if v > 0 else ("color:#ff4444" if v < 0 else "")

            st.dataframe(
                disp.style.format({
                    "Move %": "{:+.2f}%", "Price": "${:.2f}", "Reg %": "{:+.2f}%",
                    "Volume": "{:,.0f}", "Mkt Cap": "${:,.0f}",
                }, na_rep="—").map(_hl, subset=["Move %", "Reg %"]),
                use_container_width=True, height=520,
            )

            gl, gr = st.columns(2)
            with gl:
                st.subheader("Top Gainers")
                if not gainers.empty:
                    gfig = px.bar(gainers.head(12).sort_values("change_pct"),
                                  x="change_pct", y="symbol", orientation="h",
                                  color="change_pct", color_continuous_scale="Greens",
                                  labels={"change_pct": "% move", "symbol": ""})
                    gfig.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(gfig, use_container_width=True)
            with gr:
                st.subheader("Top Losers")
                if not losers.empty:
                    lfig = px.bar(losers.head(12).sort_values("change_pct", ascending=False),
                                  x="change_pct", y="symbol", orientation="h",
                                  color="change_pct", color_continuous_scale="Reds_r",
                                  labels={"change_pct": "% move", "symbol": ""})
                    lfig.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(lfig, use_container_width=True)

    # ═══════════════════════ LIVE MOVERS (RAW) ═══════════════════════════════
    elif page == "Live Movers":
        st.header("⚡ Live Movers")
        st.caption("Raw Yahoo Finance screener feed — no filters, no signals, no recommendations. "
                   "Just who's moving right now. Pick a screener and look.")

        lc1, lc2 = st.columns([2, 1])
        with lc1:
            screen_name = st.selectbox("Screener", list(RAW_SCREENS.keys()))
        with lc2:
            lm_count = st.slider("How many", 10, 100, 50, 10)

        if not scan_gate("livemovers", RECOMMENDED_TIMES["Live Movers"], clear=fetch_raw_movers.clear):
            st.stop()
        with st.spinner(f"Fetching {screen_name}…"):
            raw = fetch_raw_movers(RAW_SCREENS[screen_name], count=lm_count)

        if raw.empty:
            st.warning("No data returned — the screener feed may be unavailable right now.")
        else:
            st.caption(f"{len(raw)} names · live from Yahoo Finance · cached up to 3 min")

            def _chg(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return ""
                return "color:#00c851" if v > 0 else ("color:#ff4444" if v < 0 else "")

            st.dataframe(
                raw.style.format({
                    "Price": "${:.2f}", "Change %": "{:+.2f}%", "Pre-mkt %": "{:+.2f}%",
                    "Volume": "{:,.0f}", "Avg Vol (3M)": "{:,.0f}", "Mkt Cap": "${:,.0f}",
                }, na_rep="—").map(_chg, subset=["Change %", "Pre-mkt %"]),
                use_container_width=True, height=560,
            )

    # ═══════════════════════ AFTER-HOURS & IPOs ══════════════════════════════
    elif page == "After-Hours & IPOs":
        st.header("🌙 After-Hours & IPOs")
        st.caption("Post-market movers (extended-hours gappers that often set up the next session) "
                   "and a tracker for notable recent IPOs.")

        st.subheader("After-Hours Movers")
        ah1, ah2 = st.columns(2)
        with ah1:
            ah_n = st.slider("Show top", 10, 50, 25, 5, key="ah_n")
        with ah2:
            ah_min = st.slider("Min move %", 0.0, 10.0, 1.0, 0.5, key="ah_min")

        if not scan_gate("afterhours", RECOMMENDED_TIMES["After-Hours & IPOs"],
                         clear=lambda: (fetch_afterhours.clear(), ipo_table.clear())):
            st.stop()
        with st.spinner("Fetching after-hours movers…"):
            ah = fetch_afterhours(top_n=ah_n, min_change_pct=ah_min)

        if ah.empty:
            st.info("No after-hours movers right now — the post-market session is likely closed "
                    "(it runs ~4–8pm ET). Yahoo only reports post-market quotes during/after that "
                    "window.")
        else:
            ah_g = ah[ah["change_pct"] > 0]
            ah_l = ah[ah["change_pct"] < 0]
            s1, s2, s3 = st.columns(3)
            s1.metric("AH Gainers", len(ah_g))
            s2.metric("AH Losers", len(ah_l))
            s3.metric("Biggest move", f"{ah['change_pct'].abs().max():.2f}%")

            ah_disp = ah.rename(columns={
                "symbol": "Symbol", "name": "Name", "change_pct": "AH %",
                "price": "AH Price", "regular_change_pct": "Reg %",
                "volume": "Volume", "market_cap": "Mkt Cap",
            })[["Symbol", "Name", "AH %", "AH Price", "Reg %", "Volume", "Mkt Cap"]]

            def _hl_ah(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return ""
                return "color:#00c851" if v > 0 else ("color:#ff4444" if v < 0 else "")

            st.dataframe(
                ah_disp.style.format({
                    "AH %": "{:+.2f}%", "AH Price": "${:.2f}", "Reg %": "{:+.2f}%",
                    "Volume": "{:,.0f}", "Mkt Cap": "${:,.0f}",
                }, na_rep="—").map(_hl_ah, subset=["AH %", "Reg %"]),
                use_container_width=True, height=440,
            )

            agl, agr = st.columns(2)
            with agl:
                st.markdown("**Top AH Gainers**")
                if not ah_g.empty:
                    gfig = px.bar(ah_g.head(12).sort_values("change_pct"),
                                  x="change_pct", y="symbol", orientation="h",
                                  color="change_pct", color_continuous_scale="Greens",
                                  labels={"change_pct": "AH %", "symbol": ""})
                    gfig.update_layout(height=360, showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(gfig, use_container_width=True, key="ah_up")
            with agr:
                st.markdown("**Top AH Losers**")
                if not ah_l.empty:
                    lfig = px.bar(ah_l.head(12).sort_values("change_pct", ascending=False),
                                  x="change_pct", y="symbol", orientation="h",
                                  color="change_pct", color_continuous_scale="Reds_r",
                                  labels={"change_pct": "AH %", "symbol": ""})
                    lfig.update_layout(height=360, showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(lfig, use_container_width=True, key="ah_dn")

        st.markdown("---")
        st.subheader("Recent IPOs")
        st.caption("A **hand-curated sample** of notable recent IPOs (not an exhaustive feed — the "
                   "list is maintained manually and can lag new listings). **IPO price is "
                   "approximate** — the earliest close yfinance returns (~2y of history), not the "
                   "true offer price.")
        with st.spinner("Loading IPO performance…"):
            ipos = ipo_table()
        if ipos.empty:
            st.warning("IPO performance data is unavailable right now.")
        else:
            def _hl_gain(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return ""
                return "color:#00c851" if v > 0 else ("color:#ff4444" if v < 0 else "")

            st.dataframe(
                ipos.style.format({
                    "IPO price ≈": "${:.2f}", "Current": "${:.2f}",
                    "Gain %": "{:+.2f}%", "Days since IPO": "{:,.0f}",
                }, na_rep="—").map(_hl_gain, subset=["Gain %"]),
                use_container_width=True, height=460,
            )

        with st.expander("🔎 Look up any ticker's run since its earliest data"):
            ip_sym = st.text_input("Symbol", key="ipo_lookup").strip().upper()
            if ip_sym:
                perf = get_ipo_tracker().fetch_ipo_performance(ip_sym)
                if perf:
                    ic1, ic2, ic3 = st.columns(3)
                    ic1.metric("Earliest close ≈", f"${perf['ipo_price_approx']:.2f}")
                    ic2.metric("Current", f"${perf['current_price']:.2f}")
                    ic3.metric("Change", f"{perf['gain_pct']:+.2f}%")
                    st.caption("⚠️ 'Earliest close' is the oldest price in ~2y of history — a rough "
                               "IPO-price proxy only for stocks that listed within that window.")
                else:
                    st.info("No history available for that symbol.")

    # ═══════════════════════ WHALE MOVEMENTS ═════════════════════════════════
    elif page == "Whale Movements":
        st.header("🐋 Whale Movements")
        st.caption("Large-money footprints inferred from the public tape: names trading on an "
                   "outsized volume surge vs their own 20-day baseline, scaled by the dollars "
                   "changing hands and where the day closed in its range. No dark-pool / Level 2 "
                   "data — this is smart-money *inference* from free Yahoo Finance OHLCV.")

        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            w_n = st.slider("Universe size", 40, 250, 120, 20,
                            help="Most-active names are scanned first")
        with wc2:
            w_rvol = st.slider("Min relative volume", 1.5, 8.0, 2.0, 0.5,
                               help="Today's volume vs its 20-day average")
        with wc3:
            w_dollar = st.slider("Min $ traded ($M)", 10, 500, 50, 10,
                                 help="Minimum dollar volume to count as whale size")

        w_penny = st.checkbox("Include sub-$5 names", value=False, key="w_penny",
                              help="Penny stocks are skipped by default — thin books make their signals unreliable")
        if not scan_gate("whale", RECOMMENDED_TIMES["Whale Movements"], clear=scan_whale_activity.clear):
            st.stop()
        prefetch_histories(get_screen_universe()[:w_n], 60, "whale-scan data")
        _t0 = time.perf_counter()
        with st.spinner("Scanning the tape for whale activity…"):
            whales = scan_whale_activity(sample_size=w_n, min_rvol=w_rvol,
                                         min_dollar_vol_m=float(w_dollar),
                                         min_price=0.0 if w_penny else 5.0)
        st.caption(f"⏱ Scanned {w_n} symbols in {time.perf_counter() - _t0:.1f}s")

        if whales.empty:
            st.info("No whale-sized volume events right now. Lower the relative-volume or "
                    "$-traded thresholds, or widen the universe.")
        else:
            buying = whales[whales["signal"].isin(["Heavy Buying", "Accumulation"])]
            selling = whales[whales["signal"].isin(["Heavy Selling", "Distribution"])]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Whale events", len(whales))
            m2.metric("Bullish footprints", len(buying))
            m3.metric("Bearish footprints", len(selling))
            m4.metric("Top $ traded", f"${whales['dollar_vol'].max()/1e6:,.0f}M")

            disp = whales.rename(columns={
                "symbol": "Symbol", "signal": "Signal", "whale_score": "Whale Score",
                "rvol": "Rel Vol", "price": "Price", "change_pct": "Change %",
                "dollar_vol": "$ Traded", "close_strength": "Close Str",
                "accum_days": "Accum d", "distrib_days": "Distrib d",
            })[["Symbol", "Signal", "Whale Score", "Rel Vol", "Price", "Change %",
                "$ Traded", "Close Str", "Accum d", "Distrib d"]]

            def _whale_sig_color(v: str) -> str:
                if v in ("Heavy Buying", "Accumulation"):
                    return "color:#00c851;font-weight:bold" if v == "Heavy Buying" else "color:#00c851"
                if v in ("Heavy Selling", "Distribution"):
                    return "color:#ff4444;font-weight:bold" if v == "Heavy Selling" else "color:#ff4444"
                return "color:#888888"

            def _chg_color(v):
                return "color:#00c851" if v > 0 else ("color:#ff4444" if v < 0 else "")

            st.dataframe(
                disp.style.format({
                    "Whale Score": "{:.1f}", "Rel Vol": "{:.2f}x", "Price": "${:.2f}",
                    "Change %": "{:+.2f}%", "$ Traded": "${:,.0f}", "Close Str": "{:.2f}",
                }, na_rep="—")
                .map(_whale_sig_color, subset=["Signal"])
                .map(_chg_color, subset=["Change %"]),
                use_container_width=True, height=520,
            )

            bub = whales.copy()
            bub["dollar_vol_m"] = bub["dollar_vol"] / 1e6
            bfig = px.scatter(
                bub, x="rvol", y="change_pct", size="dollar_vol_m", color="signal",
                hover_name="symbol", size_max=46,
                labels={"rvol": "Relative volume (×)", "change_pct": "Day change %",
                        "dollar_vol_m": "$ traded (M)", "signal": "Signal"},
                color_discrete_map={
                    "Heavy Buying": "#00c851", "Accumulation": "#7cd992",
                    "Heavy Selling": "#ff4444", "Distribution": "#ff8a80", "Churn": "#9e9e9e",
                },
                title="Whale map — volume surge vs price impact (bubble = $ traded)",
            )
            bfig.add_hline(y=0, line_dash="dot", line_color="#888")
            bfig.update_layout(height=440, margin=dict(t=46, l=10, r=10, b=10))
            st.plotly_chart(bfig, use_container_width=True, key="whale_bubble")

            with st.expander("How to read this"):
                st.markdown(
                    "- **Heavy Buying / Accumulation** — big volume on an up day; whales likely "
                    "building positions. Heavy = closed in the top third of the day's range.\n"
                    "- **Heavy Selling / Distribution** — big volume on a down day; likely "
                    "offloading. Heavy = closed in the bottom third.\n"
                    "- **Churn** — outsized volume but the close reverses the day's range "
                    "(indecision / two-sided fight).\n"
                    "- **Rel Vol** — today's volume ÷ its 20-day average (2.00x = double normal).\n"
                    "- **Close Str** — where the close sat in the day's range (1.00 = on the high, "
                    "0.00 = on the low).\n"
                    "- **Accum d / Distrib d** — high-volume up vs down days over the last 20 "
                    "sessions (the multi-day balance behind the latest bar)."
                )

    # ═══════════════════════ OPTIONS FLOW ════════════════════════════════════
    elif page == "Options Flow":
        st.header("🎯 Options Flow")
        st.caption("Options-derived sentiment from live chains: IV rank, put/call ratio, and "
                   "unusual volume (a contract trading above 10% of its open interest = possible "
                   "smart-money positioning). Plus an earnings IV-crush estimate and a Greeks calc.")

        # ── Single-ticker analysis ──────────────────────────────────────────
        oc1, oc2 = st.columns([3, 1])
        with oc1:
            opt_sym = st.text_input("🔍 Analyze a ticker's options", placeholder="e.g. AAPL, NVDA",
                                    key="opt_search").strip().upper()
        with oc2:
            st.write("")
            do_opt = st.button("Analyze", use_container_width=True, key="opt_btn")

        if opt_sym and (do_opt or st.session_state.get("opt_last") == opt_sym):
            st.session_state["opt_last"] = opt_sym
            with st.spinner(f"Pulling option chains for {opt_sym}…"):
                od = analyze_options(opt_sym)

            gcol, mcol = st.columns([1, 1])
            with gcol:
                if od["iv_rank"] is not None:
                    ivfig = go.Figure(go.Indicator(
                        mode="gauge+number", value=od["iv_rank"],
                        title={"text": "IV Rank"},
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
                               "steps": [
                                   {"range": [0, 30], "color": "#9ccc65"},
                                   {"range": [30, 70], "color": "#ffe066"},
                                   {"range": [70, 100], "color": "#ff4444"}]},
                    ))
                    ivfig.update_layout(height=260, margin=dict(t=50, l=30, r=30, b=10))
                    st.plotly_chart(ivfig, use_container_width=True, key=f"iv_gauge_{opt_sym}")
                    st.caption("Low IV (<30) = cheaper premiums (favor buying); High IV (>70) = "
                               "expensive (favor selling).")
                else:
                    st.info("IV rank unavailable (no options chain for this symbol).")
            with mcol:
                pc = od["pc_ratio"]
                sent = _pc_sentiment(pc)
                sent_color = {"Bullish": "#00c851", "Bearish": "#ff4444"}.get(sent, "#888888")
                st.metric("Put/Call ratio", f"{pc:.2f}" if pc is not None else "—")
                st.markdown(f"Sentiment: <span style='color:{sent_color};font-weight:bold'>{sent}"
                            f"</span>", unsafe_allow_html=True)
                st.caption("<0.70 call-heavy (bullish) · >1.00 put-heavy (bearish)")
                if od["current_iv"] is not None:
                    st.metric("Current IV", f"{od['current_iv'] * 100:.2f}%")
                crush = od["iv_crush"]
                if crush:
                    st.metric("Days to earnings", f"{crush['days_to_earnings']}")
                    st.caption(f"Implied move ≈ ±{crush['estimated_move_pct'] * 100:.2f}% · "
                               f"est. IV crush ≈ {crush['iv_crush_pct']:.0f}% post-earnings")
                elif od["earnings_date"]:
                    st.caption(f"Next earnings: {od['earnings_date']:%Y-%m-%d}")

            # ── Key strikes & positioning (the "what does this mean" analytics) ──
            ks = od.get("key_strikes")
            if ks:
                st.markdown("#### 📍 Key strikes & positioning")
                spot = ks.get("spot")
                pw, cw, mp = ks.get("put_wall"), ks.get("call_wall"), ks.get("max_pain")
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    if pw:
                        st.metric("Put OI wall", f"${pw['strike']:.2f}",
                                  help="Strike with the most put open interest — often acts as "
                                       "support (put sellers defend it).")
                        tag = " · support" if (spot and pw["strike"] <= spot) else ""
                        st.caption(f"{pw['oi']:,} OI{tag}")
                    else:
                        st.metric("Put OI wall", "—")
                with k2:
                    if cw:
                        st.metric("Call OI wall", f"${cw['strike']:.2f}",
                                  help="Strike with the most call open interest — often acts as "
                                       "resistance.")
                        tag = " · resistance" if (spot and cw["strike"] >= spot) else ""
                        st.caption(f"{cw['oi']:,} OI{tag}")
                    else:
                        st.metric("Call OI wall", "—")
                with k3:
                    if mp is not None:
                        st.metric("Max pain", f"${mp:.2f}",
                                  help="Strike where the most option value expires worthless; "
                                       "price often gravitates here into expiration.")
                        if spot:
                            st.caption(f"{(mp - spot) / spot * 100:+.2f}% vs spot")
                    else:
                        st.metric("Max pain", "—")
                with k4:
                    pcoi = ks.get("pc_oi_ratio")
                    st.metric("Put/Call OI", f"{pcoi:.2f}" if pcoi is not None else "—",
                              help="Total put ÷ call open interest. >1 = more puts open (defensive "
                                   "/ bearish positioning); <1 = call-heavy.")
                    st.caption(f"{ks.get('total_put_oi', 0):,}P / {ks.get('total_call_oi', 0):,}C")

                # OI-by-strike around spot — the visual answer to "which strikes hold the puts?"
                oi_rows = ks.get("oi_by_strike") or []
                if oi_rows and spot:
                    lo, hi = spot * 0.8, spot * 1.2
                    near = [r for r in oi_rows if lo <= r["strike"] <= hi
                            and (r["call_oi"] or r["put_oi"])]
                    if near:
                        odf = pd.DataFrame(near)
                        ofig = go.Figure()
                        ofig.add_bar(x=odf["strike"], y=odf["call_oi"], name="Call OI",
                                     marker_color="#00c851")
                        ofig.add_bar(x=odf["strike"], y=odf["put_oi"], name="Put OI",
                                     marker_color="#ff4444")
                        ofig.add_vline(x=spot, line_dash="dot", line_color="#888",
                                       annotation_text="Spot")
                        ofig.update_layout(
                            barmode="group", height=300,
                            margin=dict(t=30, l=10, r=10, b=10),
                            xaxis_title="Strike", yaxis_title="Open interest",
                            legend=dict(orientation="h", y=1.12, x=0),
                        )
                        st.plotly_chart(ofig, use_container_width=True,
                                        key=f"oi_strikes_{opt_sym}")
                st.caption(f"Nearest expiration: {ks.get('expiration', '—')}. Open-interest "
                           "concentrations and max pain are positioning context, not a forecast.")
            else:
                st.markdown("#### 📍 Key strikes & positioning")
                st.info("Options positioning unavailable for this symbol right now — it may have "
                        "no listed options, an illiquid/empty chain, or the feed is throttling "
                        "(try again in a moment).")

            unusual = od["unusual"]
            if unusual and (unusual.get("unusual_calls") or unusual.get("unusual_puts")):
                st.markdown("#### Unusual options activity")
                ucol, pcol = st.columns(2)
                with ucol:
                    st.markdown("**Calls** (vol > 10% of OI)")
                    uc = pd.DataFrame(unusual.get("unusual_calls", []))
                    if not uc.empty:
                        st.dataframe(uc[["strike", "volume", "openInterest"]].rename(columns={
                            "strike": "Strike", "volume": "Volume", "openInterest": "Open Int"}),
                            use_container_width=True, hide_index=True)
                    else:
                        st.caption("None")
                with pcol:
                    st.markdown("**Puts** (vol > 10% of OI)")
                    up = pd.DataFrame(unusual.get("unusual_puts", []))
                    if not up.empty:
                        st.dataframe(up[["strike", "volume", "openInterest"]].rename(columns={
                            "strike": "Strike", "volume": "Volume", "openInterest": "Open Int"}),
                            use_container_width=True, hide_index=True)
                    else:
                        st.caption("None")
            else:
                st.caption("No unusual options volume detected on the nearest expiration.")

            with st.expander("🧮 Greeks calculator (Black-Scholes call)"):
                spot = (fetch_symbol_info(opt_sym).get("price")) or 100.0
                seed_iv = od["current_iv"] if od["current_iv"] else 0.30
                seed_iv_pct = min(300.0, max(1.0, round(seed_iv * 100, 1)))
                gk1, gk2, gk3 = st.columns(3)
                with gk1:
                    g_strike = st.number_input("Strike $", value=float(round(spot, 2)), step=1.0,
                                               key="g_strike")
                with gk2:
                    g_dte = st.number_input("Days to expiry", 1, 730, 30, key="g_dte")
                with gk3:
                    g_iv = st.number_input("IV (%)", 1.0, 300.0, float(seed_iv_pct),
                                           step=1.0, key="g_iv") / 100.0
                greeks = get_options_analyzer().estimate_greeks(
                    opt_sym, float(spot), float(g_strike), int(g_dte), float(g_iv))
                if greeks:
                    e1, e2, e3, e4, e5 = st.columns(5)
                    e1.metric("Call price", f"${greeks['call_price']:.2f}")
                    e2.metric("Delta", f"{greeks['delta']:.3f}")
                    e3.metric("Gamma", f"{greeks['gamma']:.4f}")
                    e4.metric("Vega", f"{greeks['vega']:.3f}")
                    e5.metric("Theta", f"{greeks['theta']:.3f}")
                    st.caption(f"Spot ${spot:.2f} · simplified estimate, not a pricing engine.")

        st.markdown("---")

        # ── Unusual-flow scan across the most-actives ───────────────────────
        st.subheader("Unusual flow scan")
        st.caption("Scans the most-active names for options sentiment + unusual volume. **Slow** — "
                   "each symbol pulls a live option chain — so it's a small sample, cached 20 min.")
        sf_n = st.slider("Symbols to scan", 5, 30, 15, 5)

        if not scan_gate("optionsflow", RECOMMENDED_TIMES["Options Flow"], clear=scan_options_flow.clear):
            st.stop()
        with st.spinner("Scanning live option chains…"):
            flow = scan_options_flow(sample_size=sf_n)

        if flow.empty:
            st.info("No options data returned for the scanned names right now.")
        else:
            def _sent_color(v: str) -> str:
                if v == "Bullish":
                    return "color:#00c851;font-weight:bold"
                if v == "Bearish":
                    return "color:#ff4444;font-weight:bold"
                return "color:#888888"

            st.dataframe(
                flow.style.format({"P/C Ratio": "{:.2f}", "# Unusual": "{:,.0f}"}, na_rep="—")
                .map(_sent_color, subset=["Sentiment"]),
                use_container_width=True, height=440,
            )

    # ═══════════════════════ PREDICTIONS (TOMORROW) ══════════════════════════
    elif page == "Predictions":
        st.header("🔮 Predictions for Tomorrow")
        st.caption("Next-session probabilistic price forecast for the most-active names. Uses "
                   "Amazon Chronos when the AI extras are installed, otherwise a Monte-Carlo "
                   "random-walk from recent returns. Shows the median predicted close, the "
                   "p10–p90 range, and the implied next-day return. **Model output, not advice.**")

        pc1, pc2 = st.columns(2)
        with pc1:
            pr_n = st.slider("Universe size", 20, 150, 60, 10,
                             help="Most-active names are scanned first")
        with pc2:
            pr_dir = st.selectbox("Show", ["All", "Bullish only", "Bearish only"])

        if not scan_gate("predictions", RECOMMENDED_TIMES["Predictions"], clear=predict_tomorrow.clear):
            st.stop()
        prefetch_histories(get_screen_universe()[:pr_n], 200, "forecast data")
        _t0 = time.perf_counter()
        with st.spinner("Forecasting tomorrow's moves…"):
            preds = predict_tomorrow(sample_size=pr_n)
        st.caption(f"⏱ Forecast {pr_n} symbols in {time.perf_counter() - _t0:.1f}s")

        if preds.empty:
            st.warning("No forecasts available right now — the data feed may be unavailable.")
        else:
            view = preds
            if pr_dir == "Bullish only":
                view = preds[preds["Pred Return %"] > 0]
            elif pr_dir == "Bearish only":
                view = preds[preds["Pred Return %"] < 0]

            ups = int((preds["Pred Return %"] > 0).sum())
            downs = int((preds["Pred Return %"] < 0).sum())
            model = preds["Model"].mode().iloc[0] if not preds.empty else "—"
            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("Forecasts", len(preds))
            pm2.metric("Predicted up", ups)
            pm3.metric("Predicted down", downs)
            pm4.metric("Model", "Chronos AI" if model == "chronos" else "Heuristic MC")
            if model != "chronos":
                st.caption("💡 Install the AI extras (`pip install -r requirements-ai.txt`) to use "
                           "the Chronos foundation model instead of the heuristic.")

            def _dir_color(v: str) -> str:
                if v == "Up":
                    return "color:#00c851;font-weight:bold"
                if v == "Down":
                    return "color:#ff4444;font-weight:bold"
                return "color:#888888"

            def _ret_color(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return ""
                return "color:#00c851" if v > 0 else ("color:#ff4444" if v < 0 else "")

            st.dataframe(
                view.style.format({
                    "Price": "${:.2f}", "Pred Close": "${:.2f}", "Pred Return %": "{:+.2f}%",
                    "Low (p10)": "${:.2f}", "High (p90)": "${:.2f}", "Uncertainty %": "{:.2f}%",
                }, na_rep="—")
                .map(_dir_color, subset=["Direction"])
                .map(_ret_color, subset=["Pred Return %"]),
                use_container_width=True, height=480,
            )

            tcol, bcol = st.columns(2)
            with tcol:
                st.subheader("Top predicted gainers")
                top_up = preds[preds["Pred Return %"] > 0].head(12)
                if not top_up.empty:
                    ufig = px.bar(top_up.sort_values("Pred Return %"),
                                  x="Pred Return %", y="Symbol", orientation="h",
                                  color="Pred Return %", color_continuous_scale="Greens",
                                  labels={"Pred Return %": "Pred. next-day %", "Symbol": ""})
                    ufig.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(ufig, use_container_width=True, key="pred_up")
            with bcol:
                st.subheader("Top predicted losers")
                top_dn = preds[preds["Pred Return %"] < 0].tail(12)
                if not top_dn.empty:
                    dfig = px.bar(top_dn.sort_values("Pred Return %", ascending=False),
                                  x="Pred Return %", y="Symbol", orientation="h",
                                  color="Pred Return %", color_continuous_scale="Reds_r",
                                  labels={"Pred Return %": "Pred. next-day %", "Symbol": ""})
                    dfig.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(dfig, use_container_width=True, key="pred_dn")

            # ── Drill-down: forecast cone for one ticker (next 5 sessions for context) ──
            st.markdown("---")
            st.subheader("Inspect a forecast")
            insp = st.selectbox("Ticker", view["Symbol"].tolist(), key="pred_inspect")
            if insp:
                fc5 = forecast_symbol(insp, horizon=5)
                if fc5:
                    st.plotly_chart(create_forecast_chart(insp, fc5),
                                    use_container_width=True, key=f"pred_cone_{insp}")
                    er = expected_return_pct(fc5)
                    st.caption(f"5-session median path · expected return {er:+.2f}% · "
                               f"source: {fc5['source']}")
                else:
                    st.info("Not enough history to chart a forecast for this ticker.")

            st.caption("⚠️ Forecasts are probabilistic and frequently wrong, especially around "
                       "news/earnings. Use as one input, not a guarantee.")

    # ═══════════════════════ AUTO WATCHLIST ══════════════════════════════════
    elif page == "Auto Watchlist":
        st.header("Auto Watchlist — Pre-Market Bulls")
        st.caption("Auto-built from today's bullish pre-market movers, with recommended "
                   "entry / stop / target and the key momentum indicators.")

        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            aw_n = st.slider("Max names", 5, 40, 20, 5)
        with ac2:
            aw_min = st.slider("Min pre-market move %", 1.0, 10.0, 2.0, 0.5)
        with ac3:
            st.write("")
            auto_save = st.checkbox("Save to watchlist", value=True,
                                    help="Overwrites the 'Pre-Market Bulls' watchlist on each run")

        use_fc = _ai_on("ai_forecast")
        if not scan_gate("autowatchlist", RECOMMENDED_TIMES["Auto Watchlist"],
                         clear=build_auto_watchlist.clear):
            st.stop()
        with st.spinner("Scanning bullish movers and building signals…"):
            _movers = fetch_movers(top_n=60, min_change_pct=aw_min)  # cached; same call the builder makes
            if not _movers.empty:
                bulls = _movers[_movers["change_pct"] > 0].head(aw_n)
                prefetch_histories(list(bulls["symbol"]), 120, "mover data")
            awl = build_auto_watchlist(config, top_n=aw_n, min_change_pct=aw_min,
                                       with_forecast=use_fc)

        if awl.empty:
            st.warning("No bullish pre-market movers found right now. Try lowering the min move %.")
        else:
            buys = int((awl["Recommendation"] == "Buy").sum())
            s1, s2, s3 = st.columns(3)
            s1.metric("Candidates", len(awl))
            s2.metric("Buy-rated", buys)
            s3.metric("Avg pre-mkt move", f"{awl['Pre-mkt %'].mean():+.2f}%")

            awl_fmt = {
                "Pre-mkt %": "{:+.2f}%", "Price": "${:.2f}", "Score": "{:.2f}",
                "RSI": "{:.2f}", "MACD hist": "{:.4f}", "Vol surge": "{:.2f}x",
                "Entry": "${:.2f}", "Stop": "${:.2f}", "Target": "${:.2f}", "R:R": "{:.2f}x",
            }
            styler = awl.style.format({**awl_fmt, **({"Fcst ret %": "{:+.2f}%"} if use_fc else {})},
                                      na_rep="—").map(_reco_color, subset=["Recommendation"])
            if use_fc and "Forecast" in awl.columns:
                styler = styler.map(_forecast_color, subset=["Forecast"])
            st.dataframe(styler, use_container_width=True, height=460)

            if auto_save:
                wl_name = "Pre-Market Bulls"
                existing = set(watchlist_mgr.get_watchlist(wl_name))
                for sym in awl["Symbol"]:
                    if sym not in existing:
                        watchlist_mgr.add_symbol(wl_name, sym)
                st.success(f"Saved {len(awl)} symbols to watchlist '{wl_name}'.")

            st.download_button("Download CSV", awl.to_csv(index=False),
                               file_name=f"auto_watchlist_{datetime.now():%Y%m%d_%H%M}.csv",
                               mime="text/csv")
            _render_legend()

    # ═══════════════════════ ETF SCREENER ════════════════════════════════════
    elif page == "ETF Screener":
        st.header("ETF Screener — by Category")
        st.caption("Famous ETFs across sectors, themes, factors, regions, commodities, bonds, "
                   "crypto and more — each with daily change and a trend signal. Pick categories "
                   "below; loading everything cold takes a moment, then it's cached.")

        all_cats = list(ETF_CATEGORIES)
        default_cats = ["Broad Market", "Sector", "Thematic"]
        ec1, ec2 = st.columns([4, 1])
        with ec1:
            sel_cats = st.multiselect("Categories", all_cats, default=default_cats,
                                      help="Each ETF is fetched individually, so more categories = slower first load.")
        with ec2:
            st.write("")
            if st.button("Select all", use_container_width=True):
                sel_cats = all_cats
        if not sel_cats:
            st.info("Pick at least one category to scan.")
            st.stop()

        if not scan_gate("etf", RECOMMENDED_TIMES["ETF Screener"], clear=fetch_etf_table.clear):
            st.stop()
        total = sum(len(ETF_CATEGORIES[c]) for c in sel_cats)
        with st.spinner(f"Loading signals for {total} ETFs across {len(sel_cats)} categories…"):
            etf_df = fetch_etf_table(config, tuple(sel_cats))

        if etf_df.empty:
            st.warning("No ETF data available right now.")
        else:
            # Sector rotation summary from the sector rows.
            sectors = etf_df[etf_df["Category"] == "Sector"].dropna(subset=["Change %"])
            if not sectors.empty:
                best = sectors.loc[sectors["Change %"].idxmax()]
                worst = sectors.loc[sectors["Change %"].idxmin()]
                r1, r2, r3 = st.columns(3)
                r1.metric("Leading sector", best["Name"], f"{best['Change %']:+.2f}%")
                r2.metric("Lagging sector", worst["Name"], f"{worst['Change %']:+.2f}%")
                r3.metric("Avg sector move", f"{sectors['Change %'].mean():+.2f}%")

            etf_fmt = {"Price": "${:.2f}", "Change %": "{:+.2f}%",
                       "Score": "{:.2f}", "RSI": "{:.2f}"}

            def _trend_color(v):
                return ("color:#00c851" if v == "long"
                        else ("color:#ff4444" if v == "short" else ""))

            for category in sel_cats:
                cat_df = etf_df[etf_df["Category"] == category]
                if cat_df.empty:
                    continue
                st.subheader(category)
                disp = cat_df[["Symbol", "Name", "Price", "Change %", "Trend",
                               "Score", "RSI", "Recommendation"]]
                st.dataframe(
                    disp.style.format(etf_fmt, na_rep="—")
                        .map(_trend_color, subset=["Trend"])
                        .map(_reco_color, subset=["Recommendation"]),
                    use_container_width=True,
                )
            _render_legend()

    # ═══════════════════════ MARKET EVENTS ═══════════════════════════════════
    elif page == "Market Events":
        st.header("Market Events — Global & Local")
        st.caption("Macro regime, the scheduled economic calendar, and live news across "
                   "equities, rates, commodities, currencies and global markets — the events "
                   "most likely to move stocks.")

        macro = get_macro_context()

        if not scan_gate("marketevents", RECOMMENDED_TIMES["Market Events"],
                         clear=lambda: (compute_market_mood.clear(), scan_market_events.clear())):
            st.stop()

        # ── Market Mood (news tone + VIX + breadth + trend) ──────────────────
        st.subheader("Market Mood")
        with st.spinner("Gauging market mood…"):
            mood = compute_market_mood(config)
        mg1, mg2 = st.columns([1, 1])
        with mg1:
            st.plotly_chart(create_mood_gauge(mood), use_container_width=True)
        with mg2:
            st.markdown(f"**Overall: {mood['label']} ({mood['score']:.1f}/100)**")
            st.caption("Composite of live news tone and known quant gauges. "
                       "0 = extreme fear · 100 = extreme greed.")
            comp_df = pd.DataFrame(
                [{"Gauge": k, "Score": round(v, 1)} for k, v in mood["components"].items()]
            )
            st.dataframe(
                comp_df.style.format({"Score": "{:.1f}"}).map(_mood_cell, subset=["Score"]),
                use_container_width=True, hide_index=True,
            )
            st.caption(f"As of {mood['as_of']}")

        st.divider()

        # Macro regime snapshot.
        vix = macro.fetch_vix()
        breadth = macro.fetch_market_breadth()
        skip, reason = macro.should_skip_entries()
        e1, e2, e3 = st.columns(3)
        if vix is not None:
            regime = ("High stress" if vix > 25 else "Complacent" if vix < 12 else "Normal")
            e1.metric("VIX", f"{vix:.2f}", regime)
        else:
            e1.metric("VIX", "—")
        e2.metric("Bullish breadth", f"{breadth['bullish_breadth_pct']:.0%}" if breadth else "—")
        e3.metric("Entry climate", "Caution" if skip else "OK", reason)

        # Scheduled economic calendar (programmatic — never goes stale).
        st.subheader("Upcoming economic calendar")
        events = macro.get_upcoming_macro_events(days_ahead=45)
        if events:
            cal = pd.DataFrame(
                [{"Date": d.strftime("%a %b %d, %Y"),
                  "In days": (d.date() - datetime.now().date()).days,
                  "Event": name} for d, name in events]
            )
            st.dataframe(cal, use_container_width=True, hide_index=True)
        else:
            st.caption("No scheduled events in the next 45 days.")
        if macro.fomc_schedule_stale():
            st.caption("⚠ The known FOMC schedule has run out — FOMC dates above are missing until "
                       "the calendar in `swingtradeapp/macro_filters.py` is updated from "
                       "federalreserve.gov (Jobs/CPI dates are generated and stay current).")

        # Live news scan across macro proxies.
        st.subheader("Live market-moving news")
        with st.spinner("Scanning global & local market news…"):
            ev = scan_market_events(config)
        if ev.empty:
            st.warning("No market news available right now.")
        else:
            areas = st.multiselect("Filter areas", list(MARKET_EVENT_TICKERS.keys()),
                                   default=list(MARKET_EVENT_TICKERS.keys()))
            shown = ev[ev["Area"].isin(areas)] if areas else ev

            net = {"positive": 0, "negative": 0, "neutral": 0}
            for s in shown["Sentiment"]:
                key = "positive" if "pos" in s else "negative" if "neg" in s else "neutral"
                net[key] += 1
            n1, n2, n3 = st.columns(3)
            n1.metric("Positive", net["positive"])
            n2.metric("Negative", net["negative"])
            n3.metric("Neutral", net["neutral"])

            def _sent_color(v: str) -> str:
                return ("color:#00c851" if "pos" in v
                        else "color:#ff4444" if "neg" in v else "color:#888888")

            st.dataframe(
                shown[["Area", "Source", "Event", "Sentiment", "Score", "Headline"]]
                    .style.format({"Score": "{:.2f}"})
                    .map(_sent_color, subset=["Sentiment"]),
                use_container_width=True, height=460,
            )
            if not _ai_on("ai_events"):
                st.caption("Tip: enable **News event tagging** in Settings for model-based "
                           "event classification (currently using keyword heuristics).")
        _render_legend()

    # ═══════════════════════ HEAT MAP ════════════════════════════════════════
    elif page == "Heat Map":
        st.header("Market Heat Map")
        st.caption("Tiles sized by market cap, colored by today's % change — green up, red down. Grouped by sector.")
        hm_size = st.slider("Universe size", 30, 300, 120, 10)

        if not scan_gate("heatmap", RECOMMENDED_TIMES["Heat Map"], clear=fetch_heatmap_data.clear):
            st.stop()
        tickers = get_screen_universe()[:hm_size]
        with st.spinner("Loading market data…"):
            hm_df = fetch_heatmap_data(tickers)
        if not hm_df.empty:
            adv = int((hm_df["change_pct"] > 0).sum())
            dec = int((hm_df["change_pct"] < 0).sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Advancing", adv)
            m2.metric("Declining", dec)
            m3.metric("Avg % change", f"{hm_df['change_pct'].mean():+.2f}%")
            st.plotly_chart(create_market_heatmap(hm_df), use_container_width=True)
            st.plotly_chart(create_sector_bar(hm_df), use_container_width=True)
        else:
            st.warning("No market data available — try a larger universe.")

    # ═══════════════════════ WATCHLISTS ══════════════════════════════════════
    elif page == "Watchlists":
        st.header("Watchlists")
        wl_names = watchlist_mgr.list_watchlists()
        wc1, wc2 = st.columns([2, 1])
        with wc1:
            sel_wl = st.selectbox("Select watchlist", wl_names)
        with wc2:
            new_wl = st.text_input("New watchlist name")
            if st.button("Create") and new_wl:
                watchlist_mgr.create_watchlist(new_wl)
                st.rerun()

        if sel_wl:
            syms = watchlist_mgr.get_watchlist(sel_wl)
            wa1, wa2 = st.columns([3, 1])
            with wa1:
                new_sym = st.text_input("Add symbol").upper()
            with wa2:
                if st.button("Add") and new_sym:
                    watchlist_mgr.add_symbol(sel_wl, new_sym)
                    st.rerun()

            if syms:
                trend_gen = get_signal_generator(config)
                wl_rows = []
                for sym in syms:
                    hist = fetch_symbol_history(sym, 90)
                    info = fetch_symbol_info(sym)
                    if not hist.empty:
                        closes = _to_series(hist, "Close")
                        volumes = _to_series(hist, "Volume") if "Volume" in hist.columns else [0]*len(closes)
                        highs = _to_series(hist, "High") if "High" in hist.columns else closes
                        lows = _to_series(hist, "Low") if "Low" in hist.columns else closes
                        sig = trend_gen.build_signal(sym, closes, volumes, highs=highs, lows=lows)
                        if sig:
                            risk = sig.entry_price - sig.stop_price
                            reward = sig.target_price - sig.entry_price
                            wl_rows.append({
                                "Symbol": sym, "Price": info.get("price"),
                                "Recommendation": recommend_label(sig.score, sig.signal_type),
                                "Score": sig.score,
                                "RSI": sig.metadata.get("rsi"), "Entry": sig.entry_price,
                                "Stop": sig.stop_price, "Target": sig.target_price,
                                "R:R": round(reward/risk, 2) if risk > 0 else 0,
                                "Sector": info.get("sector"),
                            })
                if wl_rows:
                    wl_df = pd.DataFrame(wl_rows)
                    st.dataframe(
                        wl_df.style.format({"Price": "${:.2f}", "Score": "{:.2f}",
                                            "RSI": "{:.1f}", "Entry": "${:.2f}",
                                            "Stop": "${:.2f}", "Target": "${:.2f}", "R:R": "{:.2f}x"}
                                           ).map(_reco_color, subset=["Recommendation"]),
                        use_container_width=True,
                    )
                # Remove buttons
                cols = st.columns(min(len(syms), 8))
                for i, sym in enumerate(syms):
                    with cols[i % len(cols)]:
                        if st.button(f"Remove {sym}", key=f"rm_{sym}"):
                            watchlist_mgr.remove_symbol(sel_wl, sym)
                            st.rerun()
            else:
                st.info("Watchlist is empty.")

            if sel_wl != "Default" and st.button(f"Delete watchlist '{sel_wl}'"):
                watchlist_mgr.delete_watchlist(sel_wl)
                st.rerun()

    # ═══════════════════════ COMPARE ═════════════════════════════════════════
    elif page == "Compare":
        st.header("Compare Stocks")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            sym1 = st.text_input("Stock 1", "AAPL").upper()
        with cc2:
            sym2 = st.text_input("Stock 2", "MSFT").upper()
        with cc3:
            sym3 = st.text_input("Stock 3 (optional)", "").upper()

        symbols = [s for s in [sym1, sym2, sym3] if s]
        if symbols:
            for i, sym in enumerate(symbols):
                st.plotly_chart(create_price_chart(sym, days=180), use_container_width=True,
                                key=f"cmp_price_{i}_{sym}")

            fundamentals = get_fundamentals()
            trend_gen = get_signal_generator(config)
            cmp_rows = []
            for sym in symbols:
                info = fetch_symbol_info(sym)
                fund = fundamentals.get_fundamentals(sym)
                hist = fetch_symbol_history(sym, 90)
                if not hist.empty:
                    closes = _to_series(hist, "Close")
                    volumes = _to_series(hist, "Volume") if "Volume" in hist.columns else [0]*len(closes)
                    highs = _to_series(hist, "High") if "High" in hist.columns else closes
                    lows = _to_series(hist, "Low") if "Low" in hist.columns else closes
                    sig = trend_gen.build_signal(sym, closes, volumes, highs=highs, lows=lows)
                    bt = run_symbol_backtest(sym, config)
                    crow = {
                        "Symbol": sym, "Price": info.get("price"),
                        "Recommendation": recommend_label(sig.score, sig.signal_type) if sig else "—",
                        "Score": sig.score if sig else None,
                        "RSI": sig.metadata.get("rsi") if sig else None,
                        "Entry": sig.entry_price if sig else None,
                        "Stop": sig.stop_price if sig else None,
                        "Target": sig.target_price if sig else None,
                        "P/E": fund.get("pe_ratio"), "Div Yield": fund.get("dividend_yield"),
                        "BT Win Rate": bt["win_rate"] if bt else None,
                        "BT PF": bt["profit_factor"] if bt else None,
                        "Sector": info.get("sector"),
                    }
                    if _ai_on("ai_forecast"):
                        fc = get_forecaster().forecast(closes, horizon=5)
                        crow["Fcst ret %"] = expected_return_pct(fc)
                        crow["Forecast"] = forecast_confirms(sig, fc)
                    cmp_rows.append(crow)
            if cmp_rows:
                cmp_styler = pd.DataFrame(cmp_rows).style.format({
                    "Price": "${:.2f}", "Score": "{:.2f}", "RSI": "{:.1f}",
                    "Entry": "${:.2f}", "Stop": "${:.2f}", "Target": "${:.2f}",
                    "P/E": "{:.1f}", "Div Yield": "{:.2%}",
                    "BT Win Rate": "{:.2%}", "BT PF": "{:.2f}",
                    **({"Fcst ret %": "{:+.2f}%"} if _ai_on("ai_forecast") else {}),
                }, na_rep="—").map(_reco_color, subset=["Recommendation"])
                if _ai_on("ai_forecast"):
                    cmp_styler = cmp_styler.map(_forecast_color, subset=["Forecast"])
                st.dataframe(cmp_styler, use_container_width=True)

    # ═══════════════════════ P&L TRACKER ═════════════════════════════════════
    elif page == "P&L Tracker":
        st.header("P&L Tracker — Trade Journal")

        journal = load_journal()

        # ── Add trade manually ─────────────────────────────────────────────
        with st.expander("Add Trade"):
            ta1, ta2, ta3, ta4, ta5, ta6 = st.columns(6)
            with ta1:
                t_sym = st.text_input("Symbol", key="t_sym").upper()
            with ta2:
                t_entry = st.number_input("Entry $", value=0.0, step=0.01, key="t_entry")
            with ta3:
                t_stop = st.number_input("Stop $", value=0.0, step=0.01, key="t_stop")
            with ta4:
                t_target = st.number_input("Target $", value=0.0, step=0.01, key="t_target")
            with ta5:
                t_qty = st.number_input("Qty", min_value=1, value=1, key="t_qty")
            with ta6:
                t_score = st.number_input("Score", 0.0, 1.0, 0.5, 0.01, key="t_score")
            if st.button("Log Trade") and t_sym and t_entry > 0:
                add_trade(journal, t_sym, "long", t_entry, t_stop, t_target, t_qty, t_score)
                save_journal(journal)
                st.success(f"Logged {t_sym}")
                st.rerun()

        # ── Close open trade ───────────────────────────────────────────────
        open_trades = [t for t in journal if t["status"] == "open"]
        if open_trades:
            with st.expander("Close a Trade"):
                open_syms = [f"#{t['id']} {t['symbol']} @ {t['entry_price']}" for t in open_trades]
                sel = st.selectbox("Select trade to close", open_syms, key="close_sel")
                exit_px = st.number_input("Exit price $", value=0.0, step=0.01, key="exit_px")
                if st.button("Close Trade") and exit_px > 0:
                    trade_id = int(sel.split("#")[1].split(" ")[0])
                    close_trade(journal, trade_id, exit_px)
                    save_journal(journal)
                    st.rerun()

        if not journal:
            st.info("No trades logged yet. Use the Screener to log a signal, or add one manually above.")
        else:
            df_j = pd.DataFrame(journal)

            # ── Summary metrics ────────────────────────────────────────────
            closed = df_j[df_j["status"] == "closed"]
            if not closed.empty:
                total_pnl = closed["pnl"].sum()
                wins = closed[closed["pnl"] > 0]
                losses = closed[closed["pnl"] <= 0]
                win_rate = len(wins) / len(closed)
                avg_win = wins["pnl"].mean() if not wins.empty else 0
                avg_loss = losses["pnl"].mean() if not losses.empty else 0
                profit_factor = (wins["pnl"].sum() / abs(losses["pnl"].sum())
                                 if not losses.empty and losses["pnl"].sum() != 0 else float("inf"))

                pm1, pm2, pm3, pm4, pm5 = st.columns(5)
                pm1.metric("Total P&L", f"${total_pnl:,.2f}",
                           delta_color="normal" if total_pnl >= 0 else "inverse")
                pm2.metric("Win Rate", f"{win_rate:.2%}")
                pm3.metric("Avg Win", f"${avg_win:,.2f}")
                pm4.metric("Avg Loss", f"${avg_loss:,.2f}")
                pm5.metric("Profit Factor", f"{profit_factor:.2f}")

                # Equity curve
                closed_sorted = closed.sort_values("exit_date")
                eq = closed_sorted["pnl"].cumsum()
                fig_eq = go.Figure(go.Scatter(
                    x=list(range(len(eq))), y=eq,
                    mode="lines+markers", fill="tonexty",
                    line=dict(color="green" if total_pnl >= 0 else "red", width=2)
                ))
                fig_eq.update_layout(title="Equity Curve (closed trades)", height=300,
                                     xaxis_title="Trade #", yaxis_title="Cumulative P&L ($)")
                st.plotly_chart(fig_eq, use_container_width=True)

            # ── Trade table ────────────────────────────────────────────────
            st.subheader("All Trades")
            show_cols = ["id", "symbol", "side", "entry_price", "stop_price", "target_price",
                         "qty", "score", "entry_date", "exit_price", "exit_date", "status", "pnl", "pnl_pct"]
            disp_j = df_j[show_cols].copy()
            st.dataframe(
                disp_j.style.format({
                    "entry_price": "${:.2f}", "stop_price": "${:.2f}",
                    "target_price": "${:.2f}", "exit_price": "${:.2f}",
                    "score": "{:.2f}", "pnl": "${:,.2f}", "pnl_pct": "{:.2f}%",
                }, na_rep="—"),
                use_container_width=True,
            )
            # Export journal
            st.download_button("Download Journal CSV", df_j.to_csv(index=False),
                               file_name="trade_journal.csv", mime="text/csv")

    # ═══════════════════════ ALERTS ══════════════════════════════════════════
    elif page == "Alerts":
        st.header("Price & Metric Alerts")
        ac1, ac2 = st.columns([2, 1])
        with ac1:
            alert_sym = st.text_input("Symbol").upper()
        with ac2:
            alert_type = st.selectbox("Alert type", ["Price Above", "Price Below", "RSI Above", "RSI Below"])
        alert_val = st.number_input("Value", value=0.0, step=0.1)
        if st.button("Create Alert") and alert_sym and alert_val:
            key_map = {"Price Above": ("price", "above"), "Price Below": ("price", "below"),
                       "RSI Above": ("rsi", "above"), "RSI Below": ("rsi", "below")}
            atype, acomp = key_map[alert_type]
            watchlist_mgr.add_alert(alert_sym, atype, alert_val, acomp)
            st.success(f"Alert created for {alert_sym}")

        st.subheader("Active Alerts")
        all_alerts = watchlist_mgr.get_all_alerts()
        if all_alerts:
            for sym, alerts in all_alerts.items():
                with st.expander(f"{sym} ({len(alerts)} alerts)"):
                    for i, alert in enumerate(alerts):
                        ai1, ai2 = st.columns([4, 1])
                        with ai1:
                            st.write(f"**{alert['type'].upper()}** {alert['comparison']} {alert['value']}")
                        with ai2:
                            if st.button("Delete", key=f"del_{sym}_{i}"):
                                watchlist_mgr.remove_alert(sym, i)
                                st.rerun()
        else:
            st.info("No alerts set.")

    # ═══════════════════════ SETTINGS ════════════════════════════════════════
    elif page == "How to Analyze":
        st.header("🎓 How to Analyze a Stock")
        st.caption("The method first, then a live worked example. Analysis = combining trend, "
                   "momentum, volume, risk:reward, fundamentals, sentiment and position sizing into "
                   "one checklist — no single number tells you everything.")

        # ── Part A: the framework ────────────────────────────────────────────
        st.subheader("The method — 7 steps")
        for s in ag.STEPS:
            with st.expander(s["title"]):
                st.markdown(f"**What it is:** {s['what']}")
                st.markdown(f"**Why it matters:** {s['why']}")
                st.markdown(f"**How to read it:** {s['how']}")
                st.caption(f"In this app → use the **{s['screen']}** screen.")
        _render_legend()

        st.markdown("---")
        # ── Part B: live worked example ──────────────────────────────────────
        st.subheader("Worked example — grade any ticker")
        ec1, ec2 = st.columns([3, 1])
        with ec1:
            tkr = st.text_input("Ticker", value=st.session_state.get("last_search", "AAPL"),
                                key="howto_ticker").strip().upper()
        with ec2:
            acct = st.number_input("Account size ($)", value=100_000, step=10_000, key="howto_acct")

        def _days_to_earnings(ed):
            if ed is None:
                return None
            try:
                ts = pd.Timestamp(ed, unit="s") if isinstance(ed, (int, float)) else pd.Timestamp(ed)
                ts = ts.tz_localize(None) if ts.tzinfo else ts
                return int((ts.normalize() - pd.Timestamp.now().normalize()).days)
            except Exception:
                return None

        if tkr:
            hist = fetch_symbol_history(tkr, days=160)
            if hist.empty or len(hist) < 26:
                st.warning(f"No usable price history for '{tkr}'. Check the symbol.")
            else:
                closes = _to_series(hist, "Close")
                volumes = _to_series(hist, "Volume") if "Volume" in hist.columns else [0] * len(closes)
                highs = _to_series(hist, "High") if "High" in hist.columns else closes
                lows = _to_series(hist, "Low") if "Low" in hist.columns else closes
                opens = _to_series(hist, "Open") if "Open" in hist.columns else closes

                sig = get_signal_generator(config).build_signal(
                    tkr, closes, volumes, highs=highs, lows=lows, min_score=0.0)
                info = fetch_symbol_info(tkr)
                price = info.get("price") or closes[-1]
                sma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else None
                sma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else None
                fund = get_fundamentals().get_fundamentals(tkr)
                sent = aggregate_news_sentiment(tkr, get_sentiment_analyzer(config))
                pos_pct = sent.get("positive_pct", 0) * 100 if sent.get("count") else None
                opt = analyze_options(tkr)
                days_earn = _days_to_earnings(opt.get("earnings_date"))
                whale = WhaleDetector(WhaleConfig(min_rvol=0.0, min_dollar_vol=0.0)).analyze(
                    tkr, opens, highs, lows, closes, volumes)
                whale_sig = whale.get("signal") if whale else None

                rsi = sig.metadata.get("rsi") if sig else None
                macd_h = sig.metadata.get("macd_hist") if sig else None
                vol_surge = sig.metadata.get("vol_surge") if sig else None
                rr = ((sig.target_price - sig.entry_price) / (sig.entry_price - sig.stop_price)
                      if sig and sig.entry_price > sig.stop_price else None)
                kelly_frac = None
                if sig:
                    kelly_frac = float(get_sizer(config).size_position(sig, account_size=float(acct)).fraction)

                grades = {
                    "trend": ag.grade_trend(price, sma20, sma50),
                    "momentum": ag.grade_momentum(rsi, macd_h),
                    "volume": ag.grade_volume(vol_surge, whale_sig),
                    "rr": ag.grade_rr(rr),
                    "fundamentals": ag.grade_fundamentals(
                        fund.get("pe_ratio"), fund.get("profit_margin"),
                        fund.get("roe"), fund.get("debt_to_equity")),
                    "sentiment": ag.grade_sentiment(pos_pct, days_earn),
                    "risk": ag.grade_risk(kelly_frac),
                }
                extras = {
                    "trend": (f"price ${price:.2f} · SMA20 ${sma20:.2f} · SMA50 ${sma50:.2f}"
                              if sma20 and sma50 else None),
                    "rr": ((f"entry ${sig.entry_price:.2f} (limit) · stop ${sig.stop_price:.2f} · "
                            f"target ${sig.target_price:.2f} · live ${price:.2f}")
                           if sig and rr else None),
                    "risk": (f"≈ ${kelly_frac * acct:,.0f} of ${acct:,.0f}" if kelly_frac else None),
                }
                step_by_key = {s["key"]: s for s in ag.STEPS}

                reco = recommend_label(sig.score, sig.signal_type) if sig else "—"
                st.markdown(f"### {tkr} — ${price:,.2f} · {reco}")
                if sig:
                    st.plotly_chart(
                        create_price_chart(tkr, signal_row=pd.Series(
                            {"entry": sig.entry_price, "stop": sig.stop_price,
                             "target": sig.target_price})),
                        use_container_width=True, key=f"howto_price_{tkr}")

                for s in ag.STEPS:
                    g = grades[s["key"]]
                    ic, body = st.columns([0.07, 0.93])
                    ic.markdown(f"<div style='font-size:1.7rem;line-height:1'>{ag.MARKS[g.mark]}</div>",
                                unsafe_allow_html=True)
                    line = f"**{s['title']}** — {g.note}"
                    if extras.get(s["key"]):
                        line += (f"  \n<span style='color:#888;font-size:0.85em'>"
                                 f"{extras[s['key']]}</span>")
                    body.markdown(line, unsafe_allow_html=True)
                    body.caption(s["how"])

                with st.expander("📰 Headlines (sentiment detail)"):
                    render_ticker_news(tkr, config)
                with st.expander("🔮 Model forecast (bonus)"):
                    render_forecast_panel(tkr, config,
                                          entry=(sig.entry_price if sig else None),
                                          stop=(sig.stop_price if sig else None),
                                          key_prefix="howto")

                n_pass, n_graded, verdict = ag.summarize({k: g.mark for k, g in grades.items()})
                st.markdown("---")
                st.subheader(f"📋 Scorecard — {n_pass}/{n_graded} · {verdict}")
                score_df = pd.DataFrame([
                    {"Step": step_by_key[k]["title"], "Read": ag.MARKS[grades[k].mark],
                     "Detail": grades[k].note}
                    for k in [s["key"] for s in ag.STEPS]
                ])
                st.dataframe(score_df, hide_index=True, use_container_width=True)
                st.caption("Educational framework with rule-of-thumb thresholds — **not financial "
                           "advice**. A high score is a starting point for your own research, not a "
                           "signal to buy.")

    elif page == "Information":
        st.header("ℹ️ Information & Guide")
        st.caption("How SwingTrade Pro works, what each page is for, when to run it, "
                   "and the limits of what it can — and can't — tell you.")

        st.warning(
            "**Not financial advice.** This is a research and screening tool, not a proven "
            "profitable strategy. It has **no live track record**, and its backtest metrics are "
            "**survivorship-biased** (the universe is today's listed names). Treat every signal as "
            "a starting point for your own research — never an instruction to trade."
        )

        tab_guide, tab_timing, tab_glossary, tab_data, tab_method = st.tabs(
            ["📄 Page guide", "⏰ When to run", "📖 Glossary", "🗄️ Data & sources", "⚖️ Method & limits"]
        )

        with tab_guide:
            st.markdown(
                "| Page | What it's for |\n"
                "|---|---|\n"
                "| **Screener** | The core workflow: scans the universe for trend setups, sizes "
                "positions (half-Kelly), and drills into any ticker with chart, news & forecast. |\n"
                "| **Pre-Market Movers** | Biggest gainers/losers right now (pre-market quotes when "
                "that session is open, else regular session). |\n"
                "| **Live Movers** | Raw Yahoo predefined-screener feed — no signals, no filters. "
                "Just who's moving. |\n"
                "| **After-Hours & IPOs** | Post-market movers (~4–8pm ET) plus a curated recent-IPO "
                "tracker. |\n"
                "| **Whale Movements** | Infers large-money footprints from daily volume/$-traded/"
                "closing-strength → a 0–100 whale score. |\n"
                "| **Options Flow** | Single-ticker options analysis (IV rank, put/call, unusual "
                "volume, Greeks) + a small flow scan. |\n"
                "| **Predictions** | Next-session forecast (Chronos → heuristic fallback) + a "
                "5-session drill-down cone. |\n"
                "| **Auto Watchlist** | Auto-built watchlist of the strongest current signals. |\n"
                "| **ETF Screener** | Screen ETFs by category (sector, broad market, volatility, "
                "commodities, bonds). |\n"
                "| **Market Events** | Market-wide news + a mood gauge (news tone + VIX + breadth + "
                "SPY trend). |\n"
                "| **Heat Map** | Visual map of moves across the universe. |\n"
                "| **Watchlists / Compare** | Save names to track; compare several side by side. |\n"
                "| **P&L Tracker** | Manual trade journal — log signals and track realized P&L. |\n"
                "| **Alerts** | Threshold alerts on watched names. |\n"
                "| **Settings** | Cost model, optional local AI toggles, cache/data management. |\n"
            )

        with tab_timing:
            st.markdown("#### Best time to run each screen *(all times ET)*")
            st.markdown(
                "| Screen | Best window | Why |\n"
                "|---|---|---|\n"
                "| **Pre-Market Movers** | **8:00–9:15 AM** | Overnight news, earnings, European "
                "session and 8:30 econ data are priced in; real volume, gap mostly formed. |\n"
                "| **Screener** | After 9:45 AM, or evening | Lets the opening gap settle; or scan "
                "after the close to plan tomorrow. |\n"
                "| **Live Movers** | Intraday (9:30 AM–4:00 PM) | It's a live session feed. |\n"
                "| **After-Hours & IPOs** | **4:00–8:00 PM** | Post-market fields are empty outside "
                "this window. |\n"
                "| **Predictions / Auto Watchlist** | After the close | Uses completed daily bars; "
                "stable until the next session. |\n"
            )
            st.info(
                "**On data days, wait until after 8:30 AM.** CPI / jobs / FOMC releases reshuffle "
                "the movers completely. And always check the **Volume** column — a big % move on "
                "tiny volume usually fades at the open."
            )

        with tab_glossary:
            st.markdown("Every label used across the dashboard:")
            _render_legend()

        with tab_data:
            st.markdown(
                "- **yfinance** — quotes, history, fundamentals, news, predefined screeners.\n"
                "- **Nasdaq Trader symbol directory** — the tradable universe (cached 24h).\n"
                "- **Google News RSS** — broad free news (on-demand single-ticker & market views only).\n"
                "- **Alpaca** — execution (paper/live bracket orders); offline if no keys in `.env`.\n"
            )
            st.caption(
                "All network calls are retry-wrapped (Yahoo 401/429 are usually transient). Data is "
                "free, delayed/best-effort, and occasionally wrong — corporate actions, thin-volume "
                "prints and bad ticks happen. Caches refresh on their own TTLs; force a refresh per "
                "page or via Settings → Clear all caches."
            )

        with tab_method:
            st.markdown("""
**Signals** — Trend + RSI · MACD · Bollinger Bands · ATR-based stops · Volume surge, combined
into a 0–1 score with an entry / stop / target and a reward-to-risk ratio.

**Risk** — Half-Kelly position sizing with shrinkage, a portfolio heat limit, and a daily
circuit breaker. Kelly priors are calibrated **once, out-of-sample** on walk-forward trades —
not on the window being traded.

**Backtest** — Vectorized walk-forward simulation, **net of costs** (slippage + commission,
editable in Settings). Reports win rate, profit factor and Sharpe.

**Optional AI** — Local, open-source models (price forecasting, news event tagging,
summarization, novelty), each with a heuristic fallback. Off by default; toggle in Settings.
            """)
            st.error(
                "**Read this before trusting any number.** The backtest is survivorship-biased "
                "(point-in-time data isn't wired up yet), so its win rate / profit factor are "
                "optimistic. The signals are standard, widely-arbitraged indicators with no proven "
                "edge, and there is no forward paper-trading record. The only honest way to know if "
                "the strategy works is to paper-trade it forward for months and compare to SPY "
                "buy-and-hold **after costs**."
            )

        st.markdown("---")
        st.caption("SwingTrade Pro · for research & education only · you alone are responsible for "
                   "your trades.")



    # ═══════════════════════ INSIDER ACTIVITY ════════════════════════════════
    elif page == "Insider Activity":
        st.header("🕵️ Insider Activity")
        st.caption("Real SEC **Form 4** filings — corporate insiders (officers, directors, 10%+ "
                   "owners) buying or selling their own stock. Free via Yahoo/yfinance. Routine "
                   "selling is normal (comp); **cluster buying** by several insiders is the bullish tell.")

        ic1, ic2 = st.columns([3, 1])
        with ic1:
            isym = st.text_input("Ticker", value=st.session_state.get("last_search", "AAPL"),
                                 key="insider_sym").strip().upper()
        with ic2:
            st.write("")
            do_lookup = st.button("Look up", use_container_width=True, type="primary")

        if isym and (do_lookup or st.session_state.get("_insider_last") == isym):
            st.session_state["_insider_last"] = isym
            with st.spinner(f"Fetching insider filings for {isym}…"):
                data = fetch_insider(isym)
            tidy, summ = data["tidy"], data["summary"]

            if tidy.empty:
                st.info(f"No insider transactions found for {isym} (or the feed is unavailable).")
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Net insider (180d)", summ["label"], f"{summ['score']:+d}")
                m2.metric("Buys", summ["buys"])
                m3.metric("Sells", summ["sells"])
                m4.metric("Distinct buyers", summ["cluster_buyers"])

                if summ["cluster_buyers"] >= 2 and summ["score"] > 0:
                    st.success(f"🟢 **Cluster buy** — {summ['cluster_buyers']} different insiders "
                               f"bought in the last {summ['window_days']} days.")
                st.caption(f"Buys ${summ['buy_value']:,.0f} vs sells ${summ['sell_value']:,.0f} · "
                           f"net ${summ['net_value']:,.0f} over the last {summ['window_days']} days.")

                def _act_color(v):
                    return ("color:#00c851;font-weight:bold" if v == "Buy"
                            else ("color:#ff4444;font-weight:bold" if v == "Sell" else "color:#888"))

                disp = tidy.head(50).copy()
                disp["Date"] = disp["Date"].dt.strftime("%Y-%m-%d")
                st.dataframe(
                    disp[["Date", "Insider", "Position", "Action", "Shares", "Value", "Note"]]
                        .style.format({"Shares": "{:,.0f}", "Value": "${:,.0f}"}, na_rep="—")
                        .map(_act_color, subset=["Action"]),
                    use_container_width=True, height=460, hide_index=True,
                )

                if not data["purchases"].empty:
                    with st.expander("6-month buy/sell summary (Yahoo)"):
                        st.dataframe(data["purchases"], use_container_width=True, hide_index=True)

                st.download_button("Download CSV", tidy.to_csv(index=False),
                                   file_name=f"insider_{isym}.csv", mime="text/csv")
        else:
            st.info("Enter a ticker and press **Look up** to see its insider filings.")

    # ═══════════════════════ SETUP SCANNER ═══════════════════════════════════
    elif page == "Setup Scanner":
        st.header("🧭 Setup Scanner")
        st.caption("Named, research-backed swing setups — **VCP breakout**, **20-EMA pullback**, "
                   "**double bottom**, **liquidity sweep (Turtle Soup)** and **RSI(2) reversion** "
                   "— each with a concrete entry / stop / target and the reasons it fired. Long "
                   "ideas are gated by the market regime. Educational, not financial advice.")
        _render_legend()

        reg = get_regime_read()
        badge = {"Trade": "🟢", "Caution": "🟡", "Stand-aside": "🔴"}.get(reg.verdict, "⚪")
        gate_msg = (f"{badge} **Market regime: {reg.verdict}** ({reg.score}/100) — "
                    f"{', '.join(reg.drivers)}. ")
        gate_msg += ("Long setups are sanctioned." if reg.allows_long
                     else "Regime says **stand aside** — setups below are for study only.")
        (st.success if reg.verdict == "Trade" else st.warning if reg.verdict == "Caution"
         else st.error)(gate_msg)

        ssc1, ssc2 = st.columns([1, 2])
        with ssc1:
            ss_n = st.slider("Universe size", 40, 250, 150, 10,
                             help="Most-active names are scanned first")
        with ssc2:
            all_names = [s.name for s in setups_mod.ALL_SETUPS]
            ss_pick = st.multiselect("Setups to show", all_names, default=all_names)

        ss_penny = st.checkbox("Include sub-$5 names", value=False, key="ss_penny",
                               help="Penny stocks are skipped by default — thin books make their signals unreliable")
        if not scan_gate("setupscanner", RECOMMENDED_TIMES["Setup Scanner"], clear=scan_setups.clear):
            st.stop()
        prefetch_histories(get_screen_universe()[:ss_n], 400, "setup-scan data")
        _t0 = time.perf_counter()
        with st.spinner("Scanning for setups…"):
            sdf = scan_setups(sample_size=ss_n, min_price=0.0 if ss_penny else 5.0)
        st.caption(f"⏱ Scanned {ss_n} symbols in {time.perf_counter() - _t0:.1f}s")

        if sdf.empty:
            st.info("No setups cleared the rules right now. Widen the universe, or wait for the "
                    "tape to set up — clean trends and quiet ranges simply produce fewer triggers.")
        else:
            view = sdf[sdf["setup"].isin(ss_pick)].copy() if ss_pick else sdf.copy()
            if view.empty:
                st.info("No hits for the selected setups — try selecting more.")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Setups found", len(view))
                m2.metric("Unique symbols", view["symbol"].nunique())
                m3.metric("Avg R:R", f"{view['rr'].mean():.2f}")

                sizer = get_sizer(config)

                def _alloc_pct(r) -> float:
                    sig = SimpleNamespace(score=float(r["score"]), signal_type="long",
                                          entry_price=float(r["entry"]), stop_price=float(r["stop"]),
                                          target_price=float(r["target"]))
                    try:
                        return sizer.size_position(sig, account_size=account_size).fraction * 100.0
                    except Exception:
                        return 0.0

                view["Reco"] = view["score"].map(lambda s: recommend_label(s, "long"))
                view["Alloc %"] = view.apply(_alloc_pct, axis=1)

                disp = view.rename(columns={
                    "symbol": "Symbol", "setup": "Setup", "price": "Price", "entry": "Entry",
                    "stop": "Stop", "target": "Target", "rr": "R:R", "tags": "Tags", "why": "Why",
                })[["Symbol", "Setup", "Reco", "Price", "Entry", "Stop", "Target", "R:R",
                    "Alloc %", "Tags", "Why"]]

                st.dataframe(
                    disp.style.format({
                        "Price": "${:.2f}", "Entry": "${:.2f}", "Stop": "${:.2f}",
                        "Target": "${:.2f}", "R:R": "{:.2f}", "Alloc %": "{:.2f}%",
                    }, na_rep="—").map(_reco_color, subset=["Reco"]),
                    use_container_width=True, height=480, hide_index=True,
                )
                st.download_button("Download CSV", view.to_csv(index=False),
                                   file_name=f"setups_{datetime.now():%Y%m%d_%H%M}.csv",
                                   mime="text/csv")

                # Drill-down chart with pattern overlays.
                labels = [f"{r.symbol} · {r.setup}" for r in view.itertuples()]
                pick = st.selectbox("Inspect a setup", labels)
                sel_row = view.iloc[labels.index(pick)]
                st.plotly_chart(create_setup_chart(sel_row["symbol"], sel_row),
                                use_container_width=True,
                                key=f"setup_chart_{sel_row['symbol']}_{sel_row['setup']}")
                note = _limit_note(sel_row["entry"], sel_row["price"])
                if note:
                    st.caption(note)
                st.caption(f"**Why:** {sel_row['why']}")

    # ═══════════════════════ BACKTEST LAB ════════════════════════════════════
    elif page == "Backtest Lab":
        st.header("🧪 Backtest Lab")
        st.caption("Validate a setup **honestly**: a high win rate alone is not an edge — "
                   "**expectancy** (average R per trade) and **profit factor** are. Trades are "
                   "simulated net of costs on completed daily bars; entries are causal (no "
                   "lookahead) and use each setup's own stop/target.")

        bl1, bl2 = st.columns([2, 1])
        with bl1:
            bl_setup = st.selectbox("Setup", [s.name for s in setups_mod.ALL_SETUPS])
        with bl2:
            bl_mode = st.radio("Scope", ["Single symbol", "Universe sample"], index=1)

        if bl_mode == "Single symbol":
            bl_sym = st.text_input("Symbol", "AAPL").strip().upper()
            bl_symbols = [bl_sym] if bl_sym else []
        else:
            bl_k = st.slider("Sample size (most-active names)", 10, 120, 40, 10)
            bl_symbols = get_screen_universe()[:bl_k]

        s_obj = setups_mod.SETUP_BY_NAME.get(bl_setup)
        if s_obj is not None and isinstance(s_obj, setups_mod.RSI2Setup):
            st.caption("⚠️ RSI(2) exits are condition-based (RSI>70 / close back above the 5-EMA); "
                       "here they're approximated by a tight bracket, so results are indicative.")

        if st.button("▶ Run backtest", type="primary", key="run_bt"):
            prefetch_histories(bl_symbols, 800, "backtest data")
            _t0 = time.perf_counter()
            with st.spinner(f"Backtesting {bl_setup} across {len(bl_symbols)} symbol(s)…"):
                res = run_setup_backtest(bl_setup, bl_symbols)
            st.caption(f"⏱ Backtested {len(bl_symbols)} symbol(s) in {time.perf_counter() - _t0:.1f}s")
            st.session_state["_bt_result"] = res
            st.session_state["_bt_label"] = f"{bl_setup} · {len(bl_symbols)} symbol(s)"

        res = st.session_state.get("_bt_result")
        if not res:
            st.info("Pick a setup and scope, then press **▶ Run backtest**.")
        elif res["n"] == 0:
            st.warning(f"No trades triggered for **{bl_setup}** on {res['n_symbols']} symbol(s) "
                       "with data. Try a larger sample or a different setup.")
        else:
            st.caption(f"**{st.session_state.get('_bt_label', '')}** · "
                       f"{res['n']} trades across {res['n_symbols']} symbols")
            e_color = "normal" if res["expectancy_r"] >= 0 else "inverse"
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            r1c1.metric("Expectancy", f"{res['expectancy_r']:+.2f} R",
                        help="Avg profit per trade in units of risk. >0 = a positive edge.")
            r1c2.metric("Profit factor", f"{res['profit_factor']:.2f}",
                        help="Gross win ÷ gross loss. >1 = profitable.")
            r1c3.metric("Win rate", f"{res['win_rate']*100:.2f}%")
            r1c4.metric("Trades", res["n"])
            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            r2c1.metric("Avg win", f"{res['avg_win']*100:+.2f}%")
            r2c2.metric("Avg loss", f"{res['avg_loss']*100:+.2f}%")
            r2c3.metric("Sharpe", f"{res['sharpe']:.2f}")
            r2c4.metric("Max drawdown", f"{res['max_dd']*100:.2f}%")

            verdict = ("✅ Positive expectancy — a real edge in this sample."
                       if res["expectancy_r"] > 0 and res["profit_factor"] > 1
                       else "⚠️ Negative/zero expectancy — win rate alone is misleading here.")
            (st.success if "✅" in verdict else st.warning)(verdict)

            eq = res.get("equity", [])
            if eq:
                fig = go.Figure(go.Scatter(y=eq, mode="lines", line=dict(color="#00c851", width=2),
                                           name="Equity (×)"))
                fig.update_layout(title="Equity curve (1.0 = start, trades chained)",
                                  xaxis_title="Trade #", yaxis_title="Equity (×)", height=320)
                st.plotly_chart(fig, use_container_width=True, key=f"bt_equity_{bl_setup}")

            trades = res["trades"]
            tdf = pd.DataFrame([{
                "Symbol": t.symbol, "Entry $": round(t.entry_price, 2),
                "Exit $": round(t.exit_price, 2), "Stop $": round(t.stop_price, 2),
                "Target $": round(t.target_price, 2), "P&L %": t.pnl_pct * 100.0,
                "Won": "✅" if t.won else "❌",
            } for t in trades[-60:]])
            st.dataframe(tdf.style.format({"P&L %": "{:+.2f}%"}, na_rep="—")
                         .map(_hl_pct, subset=["P&L %"]),
                         use_container_width=True, height=360, hide_index=True)
            st.caption("Showing up to the last 60 trades. Costs (slippage + commission) are applied "
                       "to every fill via the Settings cost model.")

    # ═══════════════════════ MARKET REGIME ═══════════════════════════════════
    elif page == "Market Regime":
        st.header("🌡️ Market Regime")
        st.caption("Should you be taking new long setups at all? This is the daily-bias gate the "
                   "research insists on (SPY vs its 200-SMA + slope, breadth, VIX) — plus weekday "
                   "seasonality (the Tue/Wed/Thu/Fri edges). Auto-runs; cached 10 min.")

        reg = get_regime_read()
        banner = {"Trade": "🟢 TRADE", "Caution": "🟡 CAUTION", "Stand-aside": "🔴 STAND ASIDE"}.get(reg.verdict, reg.verdict)
        verdict_fn = (st.success if reg.verdict == "Trade" else
                      st.warning if reg.verdict == "Caution" else st.error)
        verdict_fn(f"### {banner} — risk-on score {reg.score}/100\n\n" + " · ".join(reg.drivers))

        g1, g2, g3 = st.columns(3)
        g1.metric("SPY vs 200-SMA", "Above ✅" if reg.spy_above_200 else "Below ❌")
        g2.metric("Market breadth",
                  f"{reg.breadth_pct*100:.0f}%" if reg.breadth_pct is not None else "—",
                  help="Share of recent SPY closes above the 200-day SMA")
        g3.metric("VIX", f"{reg.vix:.1f}" if reg.vix is not None else "—")

        st.divider()
        st.subheader("Weekday seasonality")
        ws_sym = st.text_input("Symbol", "SPY", key="regime_sym").strip().upper() or "SPY"
        wsh = fetch_symbol_history(ws_sym, days=800)
        if wsh.empty or "Open" not in wsh.columns:
            st.info(f"No daily history for **{ws_sym}**.")
        else:
            ws = regime_mod.weekday_seasonality(
                wsh.index, _to_series(wsh, "Open"), _to_series(wsh, "High"),
                _to_series(wsh, "Low"), _to_series(wsh, "Close"))
            st.dataframe(
                ws.style.format({"Up day %": "{:.2f}%", "Avg return %": "{:+.2f}%",
                                 "Gap-fill %": "{:.2f}%"}, na_rep="—")
                  .map(_hl_pct, subset=["Avg return %"]),
                use_container_width=True,
            )
            st.caption(f"Over {int(ws['Days'].sum())} trading days. **Up day %** = close > prior "
                       "close · **Gap-fill %** = the overnight gap traded back through the prior "
                       "close intraday. Patterns drift — treat as context, not a guarantee.")



if __name__ == "__main__":
    run_dashboard()
