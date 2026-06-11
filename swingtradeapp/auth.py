"""App login via streamlit-authenticator — multi-user, bcrypt-hashed, signed-cookie ($0 / local).

Credentials live in ``.data/auth_config.yaml`` (gitignored). First run **seeds** an admin account
from ``APP_ADMIN_USER`` / ``APP_ADMIN_PASSWORD`` (environment or ``st.secrets`` — both reach
``os.environ`` by the time this runs); if no password is provided, a random one is generated, shown
once in the UI, and written to ``.data/INITIAL_ADMIN_PASSWORD.txt``. Passwords are stored
bcrypt-hashed (via the library's own hasher, so login verifies them natively).

Hosted deploys (Streamlit Community Cloud) have an ephemeral ``.data/`` — set ``APP_ADMIN_USER`` /
``APP_ADMIN_PASSWORD`` **and a fixed ``AUTH_COOKIE_KEY``** in *Secrets* so the seed is stable and the
session cookie survives restarts.

Graceful degradation (keeps a fresh clone runnable): if ``streamlit-authenticator`` / PyYAML aren't
installed, fall back to a single shared-password gate using ``APP_PASSWORD``; if that's unset too,
show a visible "login not configured" warning but allow access (local dev only).
"""

from __future__ import annotations

import hmac
import os
import secrets as _secrets
import string
from pathlib import Path
from typing import Dict, Optional, Tuple

import streamlit as st

CONFIG_PATH = Path(".data/auth_config.yaml")
INITIAL_PW_PATH = Path(".data/INITIAL_ADMIN_PASSWORD.txt")

_DEFAULT_COOKIE_NAME = "swingtrade_auth"
_DEFAULT_EXPIRY_DAYS = 30.0


def _gen_token(n: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(_secrets.choice(alphabet) for _ in range(n))


# ── Config load / seed / save ─────────────────────────────────────────────────────

def _seed_config() -> Dict:
    """Build a fresh config (one admin user + cookie settings), hashing the admin password."""
    import streamlit_authenticator as stauth

    user = (os.environ.get("APP_ADMIN_USER") or "admin").strip() or "admin"
    pw = (os.environ.get("APP_ADMIN_PASSWORD") or "").strip()
    generated = False
    if not pw:
        pw = _gen_token(14)
        generated = True

    cfg = {
        "credentials": {"usernames": {
            user: {"name": user.title(), "email": "", "password": stauth.Hasher.hash(pw)},
        }},
        "cookie": {
            "name": os.environ.get("AUTH_COOKIE_NAME") or _DEFAULT_COOKIE_NAME,
            "key": os.environ.get("AUTH_COOKIE_KEY") or _gen_token(32),
            "expiry_days": float(os.environ.get("AUTH_COOKIE_EXPIRY_DAYS") or _DEFAULT_EXPIRY_DAYS),
        },
    }
    if generated:
        try:
            INITIAL_PW_PATH.parent.mkdir(parents=True, exist_ok=True)
            INITIAL_PW_PATH.write_text(f"username: {user}\npassword: {pw}\n")
        except Exception:
            pass
        # Surfaced once by require_login() so the user can grab it on first run.
        st.session_state["_auth_generated_pw"] = (user, pw)
    return cfg


def save_config(cfg: Dict) -> bool:
    try:
        import yaml
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception:
        return False


def load_config() -> Dict:
    """Load the auth config from YAML, seeding (and saving) a fresh one if absent/invalid.

    An ``AUTH_COOKIE_KEY`` in the environment always wins, so a hosted deploy can pin a stable
    cookie-signing key via Secrets regardless of what's on disk.
    """
    cfg = None
    if CONFIG_PATH.exists():
        try:
            import yaml
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f)
        except Exception:
            cfg = None
    if not isinstance(cfg, dict) or "credentials" not in cfg or "cookie" not in cfg:
        cfg = _seed_config()
        save_config(cfg)

    env_key = os.environ.get("AUTH_COOKIE_KEY")
    if env_key:
        cfg.setdefault("cookie", {})["key"] = env_key
    return cfg


# ── The gate ────────────────────────────────────────────────────────────────────

def require_login() -> Optional[Tuple[Optional[str], Optional[str]]]:
    """Block the app until the user is authenticated.

    Renders the login form (under the branded header) and ``st.stop()``s when not logged in.
    Returns ``(name, username)`` once authenticated and renders a sidebar **Log out** button.
    """
    try:
        import streamlit_authenticator as stauth
    except Exception:
        return _password_gate_fallback()

    cfg = load_config()
    authenticator = stauth.Authenticate(
        cfg["credentials"],
        cfg["cookie"]["name"],
        cfg["cookie"]["key"],
        cfg["cookie"].get("expiry_days", _DEFAULT_EXPIRY_DAYS),
        auto_hash=False,  # passwords in the config are already bcrypt-hashed
    )

    # First-run: show the generated admin password exactly once.
    gen = st.session_state.pop("_auth_generated_pw", None)
    if gen:
        st.warning(f"Created admin account **{gen[0]}** with a generated password "
                   f"(also saved to `{INITIAL_PW_PATH}`). Copy it now, then log in:")
        st.code(gen[1], language=None)

    try:
        authenticator.login(location="main", key="Login")
    except Exception:
        # A corrupt/expired cookie can throw — clear it and ask again.
        st.session_state.pop("authentication_status", None)
        st.error("Session error — please log in again.")
        st.stop()

    status = st.session_state.get("authentication_status")
    if status is True:
        with st.sidebar:
            who = st.session_state.get("name") or st.session_state.get("username") or "user"
            st.caption(f"👤 Signed in as **{who}**")
            authenticator.logout("Log out", "sidebar", key="Logout")
        return st.session_state.get("name"), st.session_state.get("username")

    if status is False:
        st.error("Invalid username or password.")
    else:
        st.info("Please log in to use SwingTrade Pro.")
    st.stop()


def _password_gate_fallback() -> Optional[Tuple[str, str]]:
    """Single shared-password gate used only when streamlit-authenticator isn't installed."""
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected:
        st.warning("⚠️ Login is not configured (streamlit-authenticator not installed and "
                   "`APP_PASSWORD` unset). The app is running **open** — run "
                   "`pip install -r requirements.txt` to enable real login.")
        return ("guest", "guest")
    if st.session_state.get("_pw_ok"):
        with st.sidebar:
            if st.button("Log out", key="pw_logout"):
                st.session_state.pop("_pw_ok", None)
                st.rerun()
        return ("user", "user")
    pw = st.text_input("Password", type="password", key="_pw_input")
    if st.button("Log in", key="_pw_login"):
        if hmac.compare_digest(pw, expected):
            st.session_state["_pw_ok"] = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()
