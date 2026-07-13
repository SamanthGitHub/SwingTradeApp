"""Signal Stack page (extracted verbatim from the pre-split app.py monolith)."""

from services import *  # noqa: F401,F403 — the shared data/scan/chart layer


def render(ctx) -> None:
    config = ctx.config
    account_size = ctx.account_size
    watchlist_mgr = ctx.watchlist_mgr
    st.header("🧩 Signal Stack — cross-screen conviction")
    st.caption("Where the screens agree. Each name is scored by how many independent signals — "
               "technical · whale flow · forecast · options · news · YouTube · sector — line up, "
               "long or short. Built by joining the other screens' (mostly cached) outputs.")
    st.caption("⚠️ The signals aren't fully independent (technicals & forecast both use price; "
               "news & social are both sentiment), so confluence is suggestive, not proof.")

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        n = st.slider("Universe size", 40, 150, 80, 10)
    with sc2:
        dir_filter = st.selectbox("Direction", ["All", "Long", "Short"])
    with sc3:
        min_conv = st.slider("Min conviction", 0, 100, 20, 5)
    with sc4:
        min_agree = st.slider("Min signals agreeing", 1, 7, 2, 1)

    if not scan_gate("signalstack", RECOMMENDED_TIMES["Signal Stack"], clear=build_signal_stack.clear):
        st.stop()
    # 200d covers everything the stack fans out to (signals 120d, whale 60d, forecast 200d).
    prefetch_histories(get_screen_universe()[:n], 200, "signal-stack data")
    _t0 = time.perf_counter()
    with st.spinner(f"Stacking signals across {n} names (joins several scans)…"):
        stack = build_signal_stack(config, n)
    st.caption(f"⏱ Stacked {n} names in {time.perf_counter() - _t0:.1f}s")

    if stack.empty:
        st.warning("No signals to stack right now — try a larger universe or refresh.")
    else:
        view = stack.copy()
        if dir_filter == "Long":
            view = view[view["Direction"] == "long"]
        elif dir_filter == "Short":
            view = view[view["Direction"] == "short"]
        view = view[(view["Conviction"] >= min_conv) & (view["_agree"] >= min_agree)]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Names", len(view))
        m2.metric("Long", int((view["Direction"] == "long").sum()))
        m3.metric("Short", int((view["Direction"] == "short").sum()))
        m4.metric("Avg conviction", f"{view['Conviction'].mean():.0f}" if not view.empty else "—")

        if view.empty:
            st.info("Nothing clears those filters — lower the minimums.")
        else:
            sig_cols = [k.title() for k in cf.SIGNAL_ORDER]
            disp_cols = (["Symbol", "Price", "Direction", "Conviction", "Confluence", "Coverage"]
                         + sig_cols + ["Why"])

            def _dir_color(v):
                return ("color:#00c851;font-weight:bold" if v == "long"
                        else ("color:#ff4444;font-weight:bold" if v == "short" else "color:#888"))

            def _arrow_color(v):
                return ("color:#00c851" if v == "▲" else ("color:#ff4444" if v == "▼" else "color:#888"))

            st.dataframe(
                view[disp_cols].style.format({"Price": "${:.2f}", "Conviction": "{:.0f}"}, na_rep="—")
                    .map(_dir_color, subset=["Direction"])
                    .map(_arrow_color, subset=sig_cols),
                use_container_width=True, height=460, hide_index=True,
            )
            st.caption("**Conviction** = signal strength × breadth · **Coverage** = how many of "
                       "7 signals had a read · **Confluence** = how many of those agree. "
                       "▲ bullish · ▼ bearish · · neutral · blank = no read.")

            st.download_button("Download CSV", view[disp_cols].to_csv(index=False),
                               file_name=f"signal_stack_{datetime.now():%Y%m%d_%H%M}.csv",
                               mime="text/csv")

            # ── Drill-down ───────────────────────────────────────────────
            st.markdown("---")
            pick = st.selectbox("Inspect ticker", view["Symbol"].tolist())
            if pick:
                r = view[view["Symbol"] == pick].iloc[0]
                badge = r["Direction"].upper()
                st.markdown(f"### {pick} — **{badge}** · conviction {r['Conviction']} · "
                            f"confluence {r['Confluence']}")
                st.plotly_chart(create_price_chart(pick), use_container_width=True,
                                key=f"stack_price_{pick}")
                st.markdown("**Per-signal read:**")
                for k in cf.SIGNAL_ORDER:
                    arrow = r[k.title()]
                    det = r.get(f"_d_{k}", "")
                    st.markdown(f"- **{k.title()}** {arrow or '—'} · {det or '_no read_'}")
                if r["Why"]:
                    st.success(f"Agreeing screens ({badge}): {r['Why']}")
                render_analyst_brief(pick, config)

        _render_legend()
