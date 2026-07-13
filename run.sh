#!/usr/bin/env bash
# SwingTrade Pro — self-contained launcher.
# Creates an isolated Python virtual environment (OUTSIDE the synced Drive folder),
# installs pinned dependencies once, then runs the Streamlit dashboard.
#
# Usage:   ./run.sh           (or double-click run.command in Finder)
#          SETUP_ONLY=1 ./run.sh   to build the venv without launching.
set -euo pipefail
cd "$(dirname "$0")"

# ── 1. Pick a Python with broad wheel support (3.14 is too new for some wheels) ──
PYTHON=""
for cand in python3.12 python3.11 python3.13 python3.10 python3.14 python3; do
  if command -v "$cand" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
done
if [ -z "$PYTHON" ]; then
  echo "❌ No python3 found. Install Python 3.11+ from https://www.python.org/downloads/"
  exit 1
fi
echo "🐍 Using $("$PYTHON" --version) ($(command -v "$PYTHON"))"

# ── 2. venv lives outside Google Drive to avoid sync churn / file locks ──────────
VENV="${HOME}/.swingtradeapp/venv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "📦 Creating isolated virtual environment at $VENV ..."
  mkdir -p "${HOME}/.swingtradeapp"
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ── 3. Install/refresh deps only when requirements.txt changes ───────────────────
python -m pip install --quiet --upgrade pip
# shasum on macOS; sha1sum on most Linux distros.
if command -v shasum >/dev/null 2>&1; then
  REQ_HASH="$(shasum requirements.txt | awk '{print $1}')"
else
  REQ_HASH="$(sha1sum requirements.txt | awk '{print $1}')"
fi
STAMP="$VENV/.req-${REQ_HASH}"
if [ ! -f "$STAMP" ]; then
  echo "⬇️  Installing dependencies (first run or requirements changed)…"
  pip install -r requirements.txt
  rm -f "$VENV"/.req-* 2>/dev/null || true
  touch "$STAMP"
fi

# ── 4. First-launch nicety: pre-seed Streamlit credentials so the app never blocks
# on the interactive "enter your email" prompt (a desktop-launcher run has no one
# typing at stdin — without this the first launch appears to hang).
if [ ! -f "${HOME}/.streamlit/credentials.toml" ]; then
  mkdir -p "${HOME}/.streamlit"
  printf '[general]\nemail = ""\n' > "${HOME}/.streamlit/credentials.toml"
fi

# ── 5. Launch (or just verify, under SETUP_ONLY) ─────────────────────────────────
if [ "${SETUP_ONLY:-0}" = "1" ]; then
  python -c "import streamlit, plotly, yfinance, pandas, numpy; print('✅ core imports OK')"
  echo "✅ Setup complete. venv: $VENV"
  exit 0
fi
echo "🚀 Launching SwingTrade Pro → http://localhost:8501  (Ctrl+C to stop)"
exec streamlit run app.py
