#!/usr/bin/env bash
# SwingTrade Pro — Ubuntu/GNOME launcher installer.
#
# Installs a desktop entry so the app shows up in the Activities/app grid (search
# "SwingTrade") and, if a ~/Desktop folder exists, drops a trusted icon there too.
# The launcher just runs ./run.sh in a terminal, so first-run setup, logs and
# Ctrl+C-to-stop all behave exactly like launching from a shell.
#
# Usage:    ./install-ubuntu-launcher.sh
# Remove:   ./install-ubuntu-launcher.sh --uninstall
set -euo pipefail
cd "$(dirname "$0")"
APP_DIR="$(pwd)"
APPS_DIR="${HOME}/.local/share/applications"
ENTRY="swingtrade-pro.desktop"
DESKTOP_FILE="${APPS_DIR}/${ENTRY}"

if [ "${1:-}" = "--uninstall" ]; then
  rm -f "$DESKTOP_FILE" "${HOME}/Desktop/${ENTRY}"
  command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS_DIR" || true
  echo "🗑  Launcher removed."
  exit 0
fi

chmod +x run.sh
mkdir -p "$APPS_DIR"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=SwingTrade Pro
GenericName=Swing-trading dashboard
Comment=Screeners, signals, backtests and analyst briefs (opens http://localhost:8501)
Exec="${APP_DIR}/run.sh"
Path=${APP_DIR}
Icon=${APP_DIR}/ubuntu/swingtrade-pro.svg
Terminal=true
Categories=Office;Finance;
Keywords=stocks;trading;screener;finance;streamlit;
Actions=tests;

[Desktop Action tests]
Name=Run test suite
Exec=bash -c "cd '${APP_DIR}' && '${HOME}/.swingtradeapp/venv/bin/python' -m pytest; echo; read -r -p 'Done — press Enter to close.'"
EOF

# Validate + refresh the menu database when the tools are available (best-effort).
if command -v desktop-file-validate >/dev/null 2>&1; then
  desktop-file-validate "$DESKTOP_FILE" && echo "✅ desktop entry validates clean"
fi
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS_DIR" || true

# Optional Desktop icon (GNOME needs the 'trusted' flag for double-click launching).
if [ -d "${HOME}/Desktop" ]; then
  cp "$DESKTOP_FILE" "${HOME}/Desktop/${ENTRY}"
  chmod +x "${HOME}/Desktop/${ENTRY}"
  command -v gio >/dev/null 2>&1 && \
    gio set "${HOME}/Desktop/${ENTRY}" metadata::trusted true 2>/dev/null || true
  echo "🖥  Desktop icon installed (right-click → Allow Launching if it looks untrusted)."
fi

echo "✅ Installed. Find “SwingTrade Pro” in the app grid / Activities search."
echo "   It runs: ${APP_DIR}/run.sh  →  http://localhost:8501  (Ctrl+C in the terminal stops it)"
