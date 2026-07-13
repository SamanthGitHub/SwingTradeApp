"""YouTube page (extracted verbatim from the pre-split app.py monolith)."""

from services import *  # noqa: F401,F403 — the shared data/scan/chart layer


def render(ctx) -> None:
    config = ctx.config
    account_size = ctx.account_size
    watchlist_mgr = ctx.watchlist_mgr
    st.header("📺 YouTube — what top traders are saying")
    st.caption("Scans recent uploads from a curated set of finance YouTubers, reads the full "
               "transcript, and surfaces tickers, calls, pullback/merger chatter — and a "
               "running track record of who's actually right. Free & key-less.")
    st.warning("**Opinions, not signals.** Finfluencer picks frequently underperform the "
               "market. Treat mentions as leads to research, and watch the track record below "
               "before weighting anyone's view.")

    yc1, yc2 = st.columns([1, 2])
    with yc1:
        within_days = st.slider("Lookback (days)", 1, 30, 3, 1,
                                help="How many days back to scan each channel's uploads. "
                                     "(YouTube's per-channel feed only carries the latest "
                                     "~15 videos, so very active channels may not reach the "
                                     "full window.)")
        within_hours = within_days * 24.0
    with yc2:
        picked = st.multiselect("Channels", list(yt.TRADER_CHANNELS.values()),
                                default=list(yt.TRADER_CHANNELS.values()))

    handle_by_name = {name: handle for handle, name in yt.TRADER_CHANNELS.items()}
    selected_handles = [(handle_by_name[n], n) for n in picked]

    if not scan_gate("youtube", RECOMMENDED_TIMES["YouTube"],
                     clear=lambda: (fetch_yt_uploads.clear(), get_yt_transcript.clear(),
                                    get_yt_channel_id.clear())):
        st.stop()

    universe = get_universe_set()
    analyzer = get_sentiment_analyzer(config)
    event_clf = get_event_classifier() if _ai_on("ai_events") else None
    summarizer = get_summarizer() if _ai_on("ai_summary") else None
    cleaner = get_transcript_cleaner() if _ai_on("ai_clean") else None

    # ── Fetch + analyze ──────────────────────────────────────────────────
    analyses: List[yt.VideoAnalysis] = []
    skipped_channels: List[str] = []
    with st.spinner("Resolving channels & fetching recent uploads…"):
        uploads: List[yt.Upload] = []
        for handle, name in selected_handles:
            cid = get_yt_channel_id(handle)
            if not cid:
                skipped_channels.append(name)
                continue
            uploads += fetch_yt_uploads(cid, name, float(within_hours))

    uploads.sort(key=lambda u: u.published_ts, reverse=True)
    uploads = uploads[:60]  # cap transcript fetches so the scan stays responsive (newest first)

    if uploads:
        prog = st.progress(0.0, text="Reading transcripts…")
        for i, up in enumerate(uploads):
            segs = get_yt_transcript(up.video_id)
            analyses.append(yt.analyze_video(
                up, segs, universe,
                analyzer=analyzer, event_classifier=event_clf, summarizer=summarizer,
                cleaner=cleaner))
            prog.progress((i + 1) / len(uploads), text=f"Analyzed {i + 1}/{len(uploads)} videos")
        prog.empty()

    if skipped_channels:
        st.caption("Couldn't resolve: " + ", ".join(skipped_channels) +
                   " (handle renamed or rate-limited).")

    if not analyses:
        st.info("No uploads from the selected channels in the last "
                f"{within_days} day(s). Widen the lookback or pick more channels.")
    else:
        # Persist any new extracted picks (idempotent), then grade the whole history.
        store = load_yt_store()
        new_picks = sum(yt.record_picks(store, a, yt_price_on) for a in analyses)
        if new_picks:
            save_yt_store(store)
        graded = yt.grade_picks(store.get("picks", []), yt_current_price,
                                yt_spy_return_since, yt_peak_since)
        board = yt.creator_leaderboard(graded)

        transcribed = sum(1 for a in analyses if a.has_transcript)
        mm1, mm2, mm3, mm4 = st.columns(4)
        mm1.metric("Videos", len(analyses))
        mm2.metric("With transcript", f"{transcribed}/{len(analyses)}")
        mm3.metric("Tickers mentioned", len({t for a in analyses for t in a.tickers}))
        mm4.metric("Picks tracked", len(store.get("picks", [])))

        # ── Creator track record ─────────────────────────────────────────
        st.subheader("🏆 Creator track record")
        st.caption("Graded vs actual price move and SPY over the same window. Builds up as "
                   "picks accrue — sparse at first.")
        if board:
            bdf = pd.DataFrame(board)[["channel", "picks", "graded", "win_rate", "avg_alpha_pct"]]
            bdf = bdf.rename(columns={"channel": "Creator", "picks": "Picks", "graded": "Graded",
                                      "win_rate": "Win rate", "avg_alpha_pct": "Avg alpha"})
            st.dataframe(
                bdf.style.format({"Win rate": "{:.2%}", "Avg alpha": "{:+.2f}%"}, na_rep="—")
                   .map(lambda v: _hl_pct(v), subset=["Avg alpha"]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("No graded picks yet — check back after a few scans across some days.")

        # ── Consensus: what they're talking about ────────────────────────
        st.subheader("🗣️ What they're talking about")
        consensus = yt.ticker_consensus(analyses)
        wl_mgr = get_watchlist_manager()
        for c in consensus[:8]:
            sym = c["ticker"]
            rc1, rc2, rc3, rc4, rc5 = st.columns([1, 1.4, 1.4, 1.4, 1.2])
            rc1.markdown(f"**{sym}**")
            price = yt_current_price(sym)
            rc2.markdown(f"${price:.2f}" if price else "—")
            bull = c["bull_pct"]
            rc3.markdown(f"{bull:.0%} bull" if bull is not None else "—")
            rc4.markdown(f"{c['videos']} vids · {c['channels']} ch · conv {c['conviction']:.2f}")
            if rc5.button("+ Watchlist", key=f"yt_wl_{sym}"):
                wl_mgr.add_symbol("YouTube", sym)
                st.toast(f"Added {sym} to YouTube watchlist")

        # ── Pullbacks & mergers ──────────────────────────────────────────
        pc1, pc2 = st.columns(2)
        with pc1:
            st.subheader("📉 Pullbacks & impacting statements")
            hits = [a for a in analyses if a.flags.get("pullback")]
            if not hits:
                st.caption("Nothing flagged.")
            for a in hits[:6]:
                st.markdown(f"**{a.upload.channel}** — [{a.upload.title[:70]}]({a.upload.url})")
                for snip in a.flags["pullback"][:2]:
                    st.caption(snip)
        with pc2:
            st.subheader("🤝 Merger news & rumors")
            hits = [a for a in analyses if a.flags.get("merger") or "M&A" in a.events]
            if not hits:
                st.caption("Nothing flagged.")
            for a in hits[:6]:
                st.markdown(f"**{a.upload.channel}** — [{a.upload.title[:70]}]({a.upload.url})")
                for snip in a.flags.get("merger", [])[:2]:
                    st.caption(snip)

        # ── Extracted calls this scan ────────────────────────────────────
        st.subheader("🎯 Extracted calls (this scan)")
        pick_rows = [{
            "Creator": a.upload.channel, "Ticker": p.ticker,
            "Dir": (p.direction or "—").upper(), "Target": p.price_target,
            "Horizon": p.horizon or "—",
            "At": f"{p.timestamp_sec // 60}:{p.timestamp_sec % 60:02d}",
        } for a in analyses for p in a.picks]
        if pick_rows:
            st.dataframe(pd.DataFrame(pick_rows).style.format({"Target": "${:.2f}"}, na_rep="—"),
                         use_container_width=True, hide_index=True)
        else:
            st.caption("No explicit buy/sell/target calls parsed this scan.")

        # ── Per-video detail ─────────────────────────────────────────────
        st.subheader("📂 Videos")
        _sent_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}
        for a in analyses:
            up = a.upload
            head = f"{_sent_emoji.get(a.sentiment, '⚪')} {up.channel} · {up.title[:80]}"
            with st.expander(head):
                tx = "transcript" if a.has_transcript else "title+description (no transcript)"
                st.caption(f"{up.published_str} · {a.sentiment} ({a.sentiment_score:.2f}) · {tx}"
                           + (f" · events: {', '.join(a.events)}" if a.events else ""))
                st.markdown(f"[▶ Open video]({up.url})")
                if a.digest:
                    st.markdown(f"**Digest:** {a.digest}")
                if a.analysis_points:
                    st.markdown("**Key analysis points:**")
                    for pt in a.analysis_points:
                        st.markdown(f"- {pt}")
                if a.mentions:
                    st.markdown("**Mentions** (jump to the moment):")
                    for mn in a.mentions[:10]:
                        ts = f"{mn.timestamp_sec // 60}:{mn.timestamp_sec % 60:02d}"
                        st.markdown(f"- **{mn.ticker}** [{ts}]({mn.deep_link}) — {mn.snippet[:140]}")

    _render_legend()
