# Deploying SwingTrade Pro — free hosting (Streamlit Community Cloud)

The app is **cloud-ready as-is**: single entry point `app.py`, a light `requirements.txt`, every
heavy AI dependency is lazy-loaded with a heuristic fallback (so it fits the free-tier RAM), and it
runs in **offline mode** when no broker keys are present. A fully-interactive read-only deploy needs
**no code changes**.

## 1. Get the code on GitHub
Already done — `github.com/SamanthGitHub/SwingTradeApp`. Deploy from branch **`SwingTradeAppV1`**
(it has every feature), or merge it into `main` first if you'd rather deploy `main`.

> `.env`, `.data/`, `.venv/`, `__pycache__/` are gitignored — no secrets or caches are committed.

## 2. Deploy on Streamlit Community Cloud (free)
1. Go to **https://share.streamlit.io** and **sign in with GitHub** (authorize it — it can read
   your **private** repo).
2. **Create app → Deploy from a repo**:
   - **Repository:** `SamanthGitHub/SwingTradeApp`
   - **Branch:** `SwingTradeAppV1`
   - **Main file path:** `app.py`
3. **Advanced settings → Python version: `3.12`** (or `3.13`). **Not 3.14** — some wheels aren't
   published for it yet.
4. Leave everything else default. **Do not** add `requirements-ai.txt` — keeping the optional AI
   models out is what keeps the app inside the free ~1 GB RAM. The core screens (Screener, Signal
   Stack, How to Analyze, YouTube, ETF, etc.) all work on the built-in heuristic fallbacks.
5. Click **Deploy**. The first build installs dependencies (~2–3 min), then you get a public URL:
   `https://<your-app-name>.streamlit.app`.

That's it — share the URL.

## 3. (Optional) Enable Alpaca order placement
The dashboard works fully without this; it just won't place orders. To turn execution on later:

1. In the app's **Settings → Secrets**, paste:
   ```toml
   ALPACA_API_KEY = "your-key"
   ALPACA_SECRET_KEY = "your-secret"
   ALPACA_BASE_URL = "https://paper-api.alpaca.markets"   # paper trading
   ```
2. Save. The app already copies `st.secrets` into the environment at startup, so the config picks
   these up automatically — no redeploy/code change needed. (Use **paper** keys for safety.)

## 4. Things to know on the free tier
- **Storage is ephemeral.** Caches in `.data/` regenerate fine, but the **P&L journal, watchlists,
  and alerts** (JSON files) **reset on every reboot/redeploy**. Persisting them needs an external
  store (database / cloud bucket) — not included.
- **yfinance shares a cloud IP**, so 401/429 throttling is more likely than on your laptop. Keep the
  Screener **Universe size** modest (e.g. 30–50) for snappier, more reliable scans. The retry
  wrapper already handles transient errors.
- **The app sleeps after inactivity** and cold-starts on the next visit (a few seconds).
- AI features show as "not installed → heuristic fallback" on the Settings page — expected and free.

## 5. Verify before you deploy (optional local cloud-parity check)
Reproduce exactly what the cloud does — a clean install of only `requirements.txt`, no `.env`:
```bash
python3.12 -m venv /tmp/stp
/tmp/stp/bin/pip install -r requirements.txt
( unset ALPACA_API_KEY ALPACA_SECRET_KEY; /tmp/stp/bin/streamlit run app.py )
```
If it boots in offline mode and the pages load without an ImportError, Streamlit Cloud will too.

## Alternatives (not covered here)
- **Hugging Face Spaces** (Streamlit SDK) — free with ~16 GB RAM, enough to even enable the optional
  AI models; needs an HF Space + a small README header.
- A database/bucket for persistent journals/watchlists; a custom domain.
