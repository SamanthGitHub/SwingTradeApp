"""Morning Insights page (extracted verbatim from the pre-split app.py monolith)."""

from services import *  # noqa: F401,F403 — the shared data/scan/chart layer


def render(ctx) -> None:
    config = ctx.config
    account_size = ctx.account_size
    watchlist_mgr = ctx.watchlist_mgr
    phase, greeting, data_session = _market_phase()
    st.header(f"☀️ Morning Insights — {greeting}")
    st.caption("Your personal briefing — **loads automatically, no Run needed**. A decisive read "
               "on the day, what's happening with *your* names (watchlists + open positions), and "
               "the best ranked setups. Works any time of day.")

    mh1, mh2 = st.columns([3, 1])
    with mh1:
        st.caption(f"Phase: **{phase}** · as of {datetime.now():%a %b %d, %I:%M %p}")
    with mh2:
        if st.button("↻ Refresh", use_container_width=True, help="Pull fresh data"):
            for _clear in (compute_market_mood.clear, fetch_movers.clear, fetch_afterhours.clear,
                           score_symbols.clear, names_headlines.clear, _spy_trend.clear,
                           get_market_news.clear, predict_tomorrow.clear, pricestore.clear):
                try:
                    _clear()
                except Exception:
                    pass
            st.rerun()

    macro = get_macro_context()

    # ── The verdict: a plan, not gauges ──────────────────────────────────
    try:
        with st.spinner("Reading the tape…"):
            mood = compute_market_mood(config)
            spy = _spy_trend()
        vix = macro.fetch_vix()
        skip, reason = macro.should_skip_entries()
        score = float(mood.get("score", 50))
        risk_on = score >= 58 and spy in ("up", "mixed") and not skip
        risk_off = score <= 42 or spy == "down" or skip
        bias = "Long" if risk_on else "Defensive" if risk_off else "Neutral"
        events = macro.get_upcoming_macro_events(days_ahead=10) or []
        nxt = events[0] if events else None
        nxt_days = (nxt[0].date() - datetime.now().date()).days if nxt else None
        vix_txt = (f"VIX {vix:.1f}" + (" (stress)" if vix > 25 else " (calm)" if vix < 13 else "")
                   ) if vix is not None else ""
        spy_txt = {"up": "SPY uptrend", "down": "SPY downtrend",
                   "mixed": "SPY mixed"}.get(spy, "")
        evt_txt = ""
        if nxt:
            when = "today" if nxt_days == 0 else "tomorrow" if nxt_days == 1 else f"in {nxt_days}d"
            evt_txt = f"next catalyst: {nxt[1]} {when}"
        line = " · ".join(x for x in [f"Bias **{bias}**", mood.get("label", ""),
                                      vix_txt, spy_txt, evt_txt] if x)
        (st.success if bias == "Long" else st.warning if bias == "Defensive"
         else st.info)(f"🧭 {line}")
        if bias == "Long":
            st.markdown("✅ Conditions favor longs — take **A+ setups**, size normally.")
        elif bias == "Defensive":
            st.markdown("⛔ Risk-off — **trim size / wait**; only the strongest names, or cash.")
        else:
            st.markdown("➖ Mixed tape — be **selective**; let the open settle before adding.")
        if skip:
            st.markdown(f"⚠️ Macro caution: {reason}")
        if nxt and nxt_days == 0:
            st.markdown(f"📅 **{nxt[1]} today** — expect volatility; avoid fresh risk into it.")
        if st.session_state.get("analyst_briefs", True):
            try:
                reg = get_regime_read()
                mb = analyst_mod.market_brief(
                    score, bias,
                    {"verdict": reg.verdict, "score": reg.score, "drivers": list(reg.drivers)},
                    vix, reg.breadth_pct,
                    next_event=(nxt[1], nxt_days) if nxt else None)
                with st.container(border=True):
                    st.markdown(analyst_mod.render_markdown(mb))
            except Exception as exc:
                errlog.record("morning_market_brief", exc)
    except Exception as exc:
        errlog.record("morning_market_read", exc)
        st.info("Market read unavailable right now.")

    # ── My universe: open positions (first) + watchlists ─────────────────
    journal = load_journal()
    open_pos = [t for t in journal if t.get("status") == "open"]
    pos_syms = [t["symbol"] for t in open_pos]
    wl_syms: List[str] = []
    for _wl in watchlist_mgr.list_watchlists():
        wl_syms.extend(watchlist_mgr.get_watchlist(_wl))
    my_syms = list(dict.fromkeys([*pos_syms, *wl_syms]))  # de-dup, positions first

    st.divider()

    # ── What's moving (session-aware) → also the setup candidate pool ────
    st.subheader("What's moving" + (f" · {data_session}" if data_session != "regular" else ""))
    movers = pd.DataFrame()
    try:
        with st.spinner("Fetching movers…"):
            if data_session == "after-hours":
                movers = fetch_afterhours(top_n=25, min_change_pct=1.0)
            else:
                movers = fetch_movers(top_n=25, min_change_pct=1.0)
        if movers.empty:
            st.caption("No live movers for this session — using your names + forecasts below.")
        else:
            if data_session == "closed":
                st.caption("Markets closed — showing the **last regular session**.")

            def _hl(v):
                return "color:#00c851" if v > 0 else ("color:#ff4444" if v < 0 else "")

            disp = movers.rename(columns={
                "symbol": "Symbol", "name": "Name", "change_pct": "Move %", "session": "Session",
                "price": "Price", "regular_change_pct": "Reg %", "volume": "Volume",
                "market_cap": "Mkt Cap"})
            cols = [c for c in ["Symbol", "Name", "Move %", "Session", "Price", "Reg %",
                                "Volume", "Mkt Cap"] if c in disp.columns]
            st.dataframe(
                disp[cols].style.format({"Move %": "{:+.2f}%", "Price": "${:.2f}",
                    "Reg %": "{:+.2f}%", "Volume": "{:,.0f}", "Mkt Cap": "${:,.0f}"}, na_rep="—")
                    .map(_hl, subset=[c for c in ["Move %", "Reg %"] if c in cols]),
                use_container_width=True, height=300)
    except Exception as exc:
        errlog.record("morning_movers", exc)
        st.caption("Movers feed unavailable right now.")

    bull_movers = (movers[movers["change_pct"] > 0]["symbol"].tolist()
                   if not movers.empty and "change_pct" in movers.columns else [])

    # Score (my names ∪ bullish movers) once — cached, sorted key for cache hits.
    pool = list(dict.fromkeys([*my_syms, *bull_movers[:20]]))
    scored = pd.DataFrame()
    if pool:
        prefetch_histories(pool, 120, "morning-brief data")
        with st.spinner(f"Scoring {len(pool)} names…"):
            scored = score_symbols(config, tuple(sorted(set(pool))))

    st.divider()

    # ── Your names (personalized: move, reco, alerts fired, earnings, news) ─
    st.subheader(f"Your names ({len(my_syms)})")
    if not my_syms:
        st.info("No watchlists or open positions yet. Add names on **Watchlists** or log trades "
                "on **P&L Tracker**, and they'll show up here with moves, alerts & news.")
    else:
        alerts_all = watchlist_mgr.get_all_alerts()
        heads = names_headlines(tuple(my_syms[:25]))
        mine = (scored.reindex([s for s in my_syms if s in scored.index])
                if not scored.empty else pd.DataFrame())
        if mine.empty:
            st.caption("Couldn't score your names right now (data feed) — try Refresh.")
        else:
            def _hl2(v):
                return "color:#00c851" if v > 0 else ("color:#ff4444" if v < 0 else "")

            rows = []
            for sym, r in mine.iterrows():
                hl, earn = heads.get(sym, ("", False))
                fired = _eval_alerts(alerts_all.get(sym, []), r["Last"], r["RSI"])
                flags = " ".join(x for x in [("🔔" if fired else ""),
                                             ("📅ER" if earn else "")] if x)
                rows.append({"Symbol": sym, "Last": r["Last"], "Move %": r["Move %"],
                             "Reco": r["Reco"], "Score": r["Score"], "RSI": r["RSI"],
                             "Flags": flags, "Alert": fired, "Latest headline": hl})
            pdf = pd.DataFrame(rows)
            st.dataframe(
                pdf.style.format({"Last": "${:.2f}", "Move %": "{:+.2f}%", "Score": "{:.2f}",
                    "RSI": "{:.1f}"}, na_rep="—").map(_reco_color, subset=["Reco"])
                    .map(_hl2, subset=["Move %"]),
                use_container_width=True, height=min(420, 80 + 38 * len(pdf)), hide_index=True)
            fired_n = int((pdf["Alert"] != "").sum())
            if fired_n:
                st.caption(f"🔔 {fired_n} alert(s) triggered · 📅ER = earnings in recent news")

        if open_pos:
            st.markdown("**Open positions**")

            def _pnl_color(v):
                try:
                    return ("color:#00c851" if float(v) > 0
                            else "color:#ff4444" if float(v) < 0 else "")
                except (TypeError, ValueError):
                    return ""

            prows = []
            for t in open_pos:
                sym = t["symbol"]
                cur = (float(scored.loc[sym, "Last"])
                       if (not scored.empty and sym in scored.index) else None)
                entry, stop, tgt = t.get("entry_price"), t.get("stop_price"), t.get("target_price")
                qty = t.get("qty", 0)
                pnl = (cur - entry) * qty if cur and entry else None
                pnl_pct = (cur / entry - 1) * 100 if cur and entry else None
                status = ""
                if cur and stop and cur <= stop:
                    status = "⛔ stop hit"
                elif cur and tgt and cur >= tgt:
                    status = "🎯 target hit"
                elif cur and stop and (cur - stop) / cur <= 0.02:
                    status = "⚠️ near stop"
                elif cur and tgt and (tgt - cur) / cur <= 0.02:
                    status = "🟢 near target"
                prows.append({"Symbol": sym, "Qty": qty, "Entry": entry, "Last": cur,
                              "P&L $": pnl, "P&L %": pnl_pct, "Stop": stop, "Target": tgt,
                              "Status": status})
            ppdf = pd.DataFrame(prows)
            st.dataframe(
                ppdf.style.format({"Entry": "${:.2f}", "Last": "${:.2f}", "P&L $": "${:,.0f}",
                    "P&L %": "{:+.2f}%", "Stop": "${:.2f}", "Target": "${:.2f}"}, na_rep="—")
                    .map(_pnl_color, subset=["P&L $", "P&L %"]),
                use_container_width=True, hide_index=True)

    st.divider()

    # ── Today's best setups (ranked across my names + bullish movers) ────
    st.subheader("Today's best setups")
    st.caption("Ranked by signal score across your names + today's bullish movers, with levels. "
               "⭐ = one of your names. For full cross-screen conviction, open **Signal Stack**.")
    if scored.empty:
        st.caption("No scorable candidates right now.")
    else:
        setups = (scored[scored["Reco"].isin(["Buy", "Watch"])]
                  .sort_values("Score", ascending=False).head(8))
        if setups.empty:
            st.caption("No Buy/Watch-grade setups in the current pool — tape may be weak.")
        else:
            mine_set = set(my_syms)
            view = setups.copy()
            view.insert(1, "Yours", ["⭐" if s in mine_set else "" for s in view["Symbol"]])
            st.dataframe(
                view[["Symbol", "Yours", "Reco", "Score", "Last", "Entry", "Stop", "Target",
                      "R:R", "Why"]].style.format({"Score": "{:.2f}", "Last": "${:.2f}",
                    "Entry": "${:.2f}", "Stop": "${:.2f}", "Target": "${:.2f}", "R:R": "{:.2f}x"},
                    na_rep="—").map(_reco_color, subset=["Reco"]),
                use_container_width=True, hide_index=True)
            if st.session_state.get("analyst_briefs", True):
                for _sym in list(view["Symbol"].head(3)):
                    with st.expander(f"🧠 Analyst brief — {_sym}"):
                        render_analyst_brief(_sym, config, show_header=False)

    # Evening / weekend: plan tomorrow with the next-session forecast.
    if data_session == "closed":
        st.divider()
        st.subheader("Plan for tomorrow")
        try:
            with st.spinner("Forecasting next session…"):
                fc = predict_tomorrow(sample_size=60)
            if fc.empty:
                st.caption("Forecast unavailable right now.")
            else:
                st.dataframe(fc.head(8), use_container_width=True, hide_index=True)
        except Exception:
            st.caption("Forecast unavailable right now.")

    st.divider()

    # ── Catalysts & headlines (compact, tucked away) ─────────────────────
    with st.expander("📅 Calendar & 📰 market headlines"):
        try:
            evs = macro.get_upcoming_macro_events(days_ahead=10)
            if evs:
                st.dataframe(pd.DataFrame([{"Date": d.strftime("%a %b %d"),
                    "In days": (d.date() - datetime.now().date()).days, "Event": nm}
                    for d, nm in evs]), use_container_width=True, hide_index=True)
        except Exception:
            pass
        try:
            render_market_news(config, top=6)
        except Exception:
            st.caption("Market news unavailable right now.")

    _render_legend()
