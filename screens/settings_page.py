"""Settings page (extracted verbatim from the pre-split app.py monolith)."""

from services import *  # noqa: F401,F403 — the shared data/scan/chart layer


def render(ctx) -> None:
    config = ctx.config
    account_size = ctx.account_size
    watchlist_mgr = ctx.watchlist_mgr
    st.header("Settings")

    st.subheader("Backtest Cost Model")
    st.caption("Backtest metrics are net of these costs, and the position sizer is "
               "calibrated on out-of-sample walk-forward trades (not the window being traded).")
    bc1, bc2 = st.columns(2)
    with bc1:
        slip = st.number_input("Slippage (bps per fill)", 0.0, 100.0,
                               float(config.slippage_bps), 1.0)
    with bc2:
        comm = st.number_input("Commission (bps per side)", 0.0, 50.0,
                               float(config.commission_bps), 0.5)
    if (slip, comm) != (config.slippage_bps, config.commission_bps):
        st.session_state["slippage_bps"] = slip
        st.session_state["commission_bps"] = comm
        st.cache_data.clear()
        st.cache_resource.clear()
        st.info("Cost model updated — caches cleared. Re-run the Screener to recalibrate.")
        st.rerun()

    st.subheader("AI Features (open-source, local)")
    st.caption("Each loads a Hugging Face model on first use (CPU). If a model isn't "
               "installed, the feature falls back to a heuristic. Install extras with: "
               "`pip install transformers torch chronos-forecasting sentence-transformers`.")
    ai1, ai2 = st.columns(2)
    with ai1:
        st.session_state["ai_forecast"] = st.checkbox(
            "Price forecasting (Chronos)", value=_ai_on("ai_forecast"),
            help="Probabilistic p10/p50/p90 forecast + signal confirmation")
        st.session_state["ai_events"] = st.checkbox(
            "News event tagging", value=_ai_on("ai_events"),
            help="Zero-shot earnings / M&A / downgrade / lawsuit tags")
    with ai2:
        st.session_state["ai_summary"] = st.checkbox(
            "News summarization digest", value=_ai_on("ai_summary"),
            help="distilbart digest of recent headlines")
        st.session_state["ai_novelty"] = st.checkbox(
            "Semantic news novelty", value=_ai_on("ai_novelty"),
            help="Dedup recycled headlines via sentence-transformers")
        st.session_state["ai_clean"] = st.checkbox(
            "Transcript cleanup (punctuation & casing)", value=_ai_on("ai_clean"),
            help="Restore punctuation + casing to messy YouTube auto-captions so the digest "
                 "and key analysis points read cleanly (deepmultilingualpunctuation). Falls "
                 "back to the heuristic tidy when the model isn't installed.")
    st.caption("Forecast appears on Auto Watchlist + Screener drill-down; news AI on the "
               "Screener drill-down; transcript cleanup on the YouTube screen.")

    st.subheader("🧠 Analyst briefs")
    st.session_state["analyst_briefs"] = st.checkbox(
        "Plain-English analyst briefs (free, instant, template-generated)",
        value=bool(st.session_state.get("analyst_briefs", True)),
        help="A multi-paragraph trade thesis composed from the app's own signals — trend, "
             "smart money, trade plan, catalysts, risks. Appears on the Screener drill-down, "
             "Signal Stack drill-down and Morning Insights. No model download; can't "
             "hallucinate numbers because it only interpolates the app's own values.")
    with st.expander("Optional: polish briefs with a local LLM (Ollama)"):
        st.caption("Free and fully local. Install [Ollama](https://ollama.com), run "
                   "`ollama pull llama3.2:3b`, then enable below. The LLM only *rephrases* the "
                   "finished brief — it's instructed to add no numbers or claims — and any "
                   "failure silently falls back to the template text.")
        oc1, oc2 = st.columns(2)
        with oc1:
            st.session_state["ollama_host"] = st.text_input(
                "Ollama host", st.session_state.get("ollama_host", "http://localhost:11434"))
        with oc2:
            st.session_state["ollama_model"] = st.text_input(
                "Model", st.session_state.get("ollama_model", "llama3.2:3b"))
        st.session_state["ollama_polish"] = st.checkbox(
            "Polish analyst briefs with the local LLM",
            value=bool(st.session_state.get("ollama_polish", False)))
        if st.button("Test connection", key="ollama_test"):
            client = OllamaClient(host=st.session_state.get("ollama_host", ""),
                                  model=st.session_state.get("ollama_model", ""))
            if client.available():
                st.success(f"Ollama is answering at {client.host} ✓")
            else:
                st.error(f"No Ollama server at {client.host} — is `ollama serve` running?")

    st.markdown("**ML signal model** — uses scikit-learn (already installed; no extra download)")
    st.session_state["ai_ml_signal"] = st.checkbox(
        "Add ML P(up) to the Screener + drive Kelly sizing", value=_ai_on("ai_ml_signal"),
        help="A calibrated, walk-forward-trained probability of an up move over the swing "
             "horizon, learned from the same technical features the Screener uses.")
    if _ai_on("ai_ml_signal"):
        _mlm = get_ml_signal_model()
        if _mlm is not None and _mlm.metrics:
            mm = _mlm.metrics
            ms1, ms2, ms3 = st.columns(3)
            ms1.metric("OOS AUC", f"{mm.get('auc', float('nan')):.3f}")
            ms2.metric("OOS accuracy", f"{mm.get('accuracy', float('nan')):.2%}")
            ms3.metric("Brier (lower=better)", f"{mm.get('brier', float('nan')):.3f}")
            st.caption(f"Trained on {mm.get('n_train', 0):,} samples · validated on "
                       f"{mm.get('n_val', 0):,} (held-out, base rate "
                       f"{mm.get('base_rate', float('nan')):.0%}) · horizon ~{_mlm.horizon} "
                       "sessions. AUC just above 0.50 is normal for noisy daily data.")
        else:
            st.caption("Model not trained yet — it builds on the next Screener scan "
                       "(or scikit-learn/data is unavailable).")
        if st.button("Retrain ML model"):
            try:
                ML_MODEL_PATH.unlink(missing_ok=True)
            except Exception:
                pass
            get_ml_signal_model.clear()
            st.success("Cleared — the model retrains on the next Screener scan.")

    st.subheader("Data & APIs")
    st.caption("Add free-tier API keys to bring in extra data sources. Usage is **hard-capped** "
               "to each provider's free limit — when you reach it, screens automatically fall "
               "back to free Yahoo data, so you're **never charged**. Keys are stored in the "
               "gitignored `.env`.")
    _budget = get_api_budget()
    for _pk, _spec in PROVIDERS.items():
        with st.container(border=True):
            st.markdown(f"**{_spec.name}** — best for {_spec.recommended_use}  ·  "
                        f"[plan & limits]({_spec.docs_url})")
            _has_key = bool(_polygon_key()) if _pk == "polygon" else bool(
                st.session_state.get(f"{_pk}_api_key"))
            _new = st.text_input(f"{_spec.name} API key", value="", type="password",
                                 placeholder=("✓ key on file — paste a new one to replace"
                                              if _has_key else "paste your key…"),
                                 key=f"{_pk}_api_key_input")
            if _new.strip():
                st.session_state[f"{_pk}_api_key"] = _new.strip()
                _persist_env(_spec.env_var, _new.strip())
                _has_key = True
                st.success(f"{_spec.name} key saved to .env.")
            st.session_state[f"use_{_pk}"] = st.checkbox(
                f"Use {_spec.name} for single-symbol lookups", value=_provider_on(_pk),
                disabled=not _has_key, key=f"use_{_pk}_box",
                help="Drill-down charts & one-ticker lookups use this provider within its free "
                     "limit; bulk scans always stay on free Yahoo data.")
            # Live usage meters.
            _stt = _budget.status(_pk)
            if _spec.per_minute:
                _used = int(_stt.get("used_minute", 0))
                st.progress(min(_used / _spec.per_minute, 1.0),
                            text=f"{_used}/{_spec.per_minute} calls this minute")
                if _used >= _spec.per_minute:
                    st.error(f"Minute limit reached — resets in ~{int(_stt.get('reset_in_s', 0))}s. "
                             "Falling back to free data.")
                elif _used >= 0.8 * _spec.per_minute:
                    st.warning(f"Near the minute limit ({_used}/{_spec.per_minute}).")

    st.subheader("Data health")
    n_issues = errlog.count()
    if n_issues == 0:
        st.caption("✅ No data issues recorded this session.")
    else:
        st.caption(f"⚠ {n_issues} data issue{'s' if n_issues != 1 else ''} recorded this "
                   "session. Failures never crash a page (the app falls back to cached/free "
                   "data), but a repeating source below usually means a provider outage or "
                   "throttling.")
        roll = errlog.summary()
        st.dataframe(pd.DataFrame(
            [{"source": s, "count": c} for s, c in
             sorted(roll.items(), key=lambda kv: -kv[1])]),
            use_container_width=True, hide_index=True, height=160)
        with st.expander("Recent entries"):
            ent = errlog.entries()[:50]
            st.dataframe(pd.DataFrame([{
                "time": datetime.fromtimestamp(e["ts"]).strftime("%H:%M:%S"),
                "source": e["source"], "note": e["note"], "error": e["error"],
            } for e in ent]), use_container_width=True, hide_index=True, height=240)
        if st.button("Clear data-health log"):
            errlog.clear()
            st.rerun()

    st.subheader("Cache Management")
    if st.button("Clear all caches"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Caches cleared!")

    st.subheader("About")
    st.markdown("""
**SwingTrade Pro Dashboard**

Signal engine: RSI · MACD · Bollinger Bands · ATR stops · Volume surge
Risk: Half-Kelly sizing · Portfolio heat limit · Daily circuit breaker
Backtest: Walk-forward vectorized simulation · Win rate · Profit factor · Sharpe
Execution: Alpaca bracket orders (entry / stop-loss / take-profit)
    """)

    data_dir = Path(".data")
    if data_dir.exists():
        st.subheader("Data Files")
        for f in data_dir.iterdir():
            if f.is_file():
                st.write(f"✅ `{f.name}` — {f.stat().st_size:,} bytes")
        if st.button("Reset all data"):
            import shutil
            shutil.rmtree(data_dir)
            st.success("Data reset!")
            st.rerun()
