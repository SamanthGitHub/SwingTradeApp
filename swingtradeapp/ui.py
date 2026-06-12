"""Presentation layer: app theming, branded header, and the icon-based sidebar navigation.

Kept separate from ``app.py`` so the 1900-line dashboard stays focused on data/logic. Everything
here is purely visual and **auto light/dark aware** — colors come from Streamlit theme CSS
variables (``var(--secondary-background-color)`` etc.) or ``@media (prefers-color-scheme)`` so the
app follows the OS setting (no hardcoded background that breaks in one mode).

The icon menu uses ``streamlit-option-menu`` when installed and **degrades gracefully** to a
styled ``st.radio`` when it isn't (same fallback philosophy as the optional AI features), so the
app never crashes if the dependency hasn't been installed into the venv yet.
"""

import html
from typing import Dict, List, Optional, Sequence, Tuple

import streamlit as st

# Brand accent (green = buy/positive, consistent with the rest of the palette).
ACCENT = "#00c851"
ACCENT_DARK = "#00a843"

# (page label, Bootstrap icon name, emoji fallback). The LABEL must stay byte-identical to the
# `if page == "..."` branches in app.py — it is the routing key.
# Ordered as a trader's workflow: find ideas → see what's moving → dig into signals →
# research the story → track your own positions → help & config. Labels MUST stay
# byte-identical to the `page == "…"` branches in app.py (they're the routing keys);
# only the order changes here.
NAV_ITEMS: List[Tuple[str, str, str]] = [
    # Find ideas
    ("Screener", "graph-up", "📈"),
    ("Rally Radar", "rocket-takeoff", "🚀"),  # early / pre-breakout momentum-ignition setups
    ("Signal Stack", "layers", "🧩"),  # synthesis of the other screens
    ("ETF Screener", "grid-3x3-gap", "🗂️"),
    ("Auto Watchlist", "stars", "⭐"),
    # What's moving now (by session) + market overview
    ("Pre-Market Movers", "sunrise", "🌅"),
    ("Live Movers", "lightning-charge", "⚡"),
    ("After-Hours & IPOs", "moon-stars", "🌙"),
    ("Heat Map", "fire", "🔥"),
    # Deeper signals & forecasts
    ("Whale Movements", "water", "🐋"),
    ("Insider Activity", "person-badge", "🕵️"),
    ("Options Flow", "graph-up-arrow", "🎯"),
    ("Predictions", "magic", "🔮"),
    # News & sentiment research
    ("Market Events", "newspaper", "📰"),
    ("YouTube", "youtube", "📺"),
    # Track & manage my positions
    ("Watchlists", "bookmark-star", "🔖"),
    ("Compare", "bar-chart-line", "📊"),
    ("P&L Tracker", "journal-text", "📒"),
    ("Alerts", "bell", "🔔"),
    # Help & config
    ("How to Analyze", "mortarboard", "🎓"),
    ("Information", "info-circle", "ℹ️"),
    ("Settings", "gear", "⚙️"),
]

_BRAND_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, button, input, select, textarea {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

/* Hide ONLY the cosmetic chrome (footer, the dev "Deploy" button, the ⋮ menu). Do NOT hide the
   whole toolbar/header — that's what removed Streamlit's native sidebar expand (») control and made
   a collapsed nav impossible to reopen. The header/toolbar structure must survive. */
footer {{visibility: hidden;}}
[data-testid="stHeader"] {{background: transparent;}}
[data-testid="stAppDeployButton"] {{display: none !important;}}
#MainMenu, [data-testid="stMainMenu"] {{visibility: hidden;}}

/* Make the expand-sidebar (») control unmistakable so a collapsed nav is always reopenable. */
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] {{
    visibility: visible !important; opacity: 1 !important;
    z-index: 1000000 !important; pointer-events: auto !important;
}}
[data-testid="stExpandSidebarButton"] {{
    background: var(--secondary-background-color) !important;
    border: 1px solid rgba(128,128,128,.30) !important; border-radius: 0.5rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,.14) !important;
}}

@keyframes brandShift {{ 0% {{background-position: 0% 50%;}} 50% {{background-position: 100% 50%;}} 100% {{background-position: 0% 50%;}} }}
@keyframes livePulse {{ 0%,100% {{opacity: 1; box-shadow: 0 0 0 0 rgba(0,255,106,.55);}} 50% {{opacity: .55; box-shadow: 0 0 0 7px rgba(0,255,106,0);}} }}

/* Branded animated gradient header band with a live status pill. */
.brand-header {{
    position: relative; overflow: hidden;
    display: flex; align-items: center; justify-content: space-between; gap: 1rem;
    padding: 1.15rem 1.5rem; border-radius: 1rem; margin-bottom: 1.15rem;
    background: linear-gradient(110deg, {ACCENT_DARK} 0%, #0b7d6f 42%, #0b6cad 74%, #5a3fb5 100%);
    background-size: 240% 240%; animation: brandShift 14s ease infinite;
    box-shadow: 0 10px 32px rgba(0,168,67,.26), 0 2px 8px rgba(0,0,0,.22);
}}
.brand-header::after {{
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background: radial-gradient(1100px 220px at 8% -30%, rgba(255,255,255,.20), transparent 60%);
}}
.brand-header h1 {{
    color: #fff; font-size: 1.8rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;
}}
.brand-header p {{
    color: rgba(255,255,255,.9); font-size: 0.92rem; font-weight: 500; margin: 0.25rem 0 0 0;
}}
.brand-live {{
    display: flex; align-items: center; gap: 0.5rem; flex-shrink: 0;
    color: #fff; font-weight: 700; font-size: 0.74rem; letter-spacing: 0.12em;
    background: rgba(255,255,255,.16); padding: 0.42rem 0.85rem; border-radius: 999px;
    backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,.22);
}}
.brand-live .dot {{
    width: 9px; height: 9px; border-radius: 50%; background: #00ff6a;
    animation: livePulse 1.8s ease-in-out infinite;
}}

/* Metric "cards" — accent bar + hover-lift, adapts to light/dark via theme vars. */
[data-testid="stMetric"] {{
    position: relative; overflow: hidden;
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,.16);
    border-radius: 0.85rem;
    padding: 0.9rem 1.0rem 0.9rem 1.15rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
    transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}}
[data-testid="stMetric"]::before {{
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg, {ACCENT}, #0b6cad);
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 24px rgba(0,0,0,.14);
    border-color: rgba(0,200,81,.5);
}}
[data-testid="stMetricValue"] {{ font-weight: 800; font-size: 1.65rem; letter-spacing: -0.02em; }}
[data-testid="stMetricLabel"] {{
    opacity: .7; font-weight: 600; text-transform: uppercase;
    font-size: 0.72rem; letter-spacing: 0.05em;
}}

/* Pill-style tabs. */
.stTabs [data-baseweb="tab-list"] {{ gap: 0.45rem; border-bottom: none; }}
.stTabs [data-baseweb="tab"] {{
    border-radius: 999px; padding: 0.35rem 0.95rem; font-size: 14px; font-weight: 600;
    background: var(--secondary-background-color); border: 1px solid rgba(128,128,128,.18);
}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(110deg, {ACCENT}, {ACCENT_DARK}) !important;
    border-color: transparent !important; box-shadow: 0 3px 12px rgba(0,200,81,.35);
}}
.stTabs [aria-selected="true"] p {{ color: #fff !important; }}

/* Buttons — rounded, lift + accent glow on hover; gradient for primary. */
.stButton > button, .stDownloadButton > button {{
    border-radius: 0.6rem; font-weight: 600; border: 1px solid rgba(128,128,128,.25);
    transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    transform: translateY(-1px); border-color: rgba(0,200,81,.6);
    box-shadow: 0 5px 16px rgba(0,200,81,.22);
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(110deg, {ACCENT}, {ACCENT_DARK}); border: none; color: #fff;
}}

/* Containers: rounded dataframes + expanders. */
[data-testid="stDataFrame"], [data-testid="stExpander"] {{
    border-radius: 0.6rem; overflow: hidden; border: 1px solid rgba(128,128,128,.16);
}}

/* Sidebar polish + subtle accent wash at the top. */
[data-testid="stSidebar"] {{
    border-right: 1px solid rgba(128,128,128,.15);
    background-image: linear-gradient(180deg, rgba(0,200,81,.05), transparent 28%);
}}
[data-testid="stSidebar"] .stRadio label {{
    font-size: 0.95rem; padding: 0.15rem 0; font-weight: 500;
}}

.metric-positive {{color: {ACCENT}; font-weight: bold;}}
.metric-negative {{color: #ff4444; font-weight: bold;}}

/* ── Ticker hover cards: a strip of pills that reveal a company info card on hover ──
   Pure CSS (no JS); theme-aware via Streamlit vars; pills are focusable so the card also
   shows on keyboard focus / tap (touch devices have no hover). */
.tk-hovers {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.1rem 0 0.7rem; }}
.tk-pill {{
    position: relative; display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.2rem 0.62rem; border-radius: 999px; cursor: default;
    font-weight: 700; font-size: 0.82rem; line-height: 1.4;
    background: var(--secondary-background-color); border: 1px solid rgba(128,128,128,.22);
    transition: border-color .12s ease, box-shadow .12s ease;
}}
.tk-pill:hover, .tk-pill:focus {{
    border-color: rgba(0,200,81,.6); box-shadow: 0 3px 10px rgba(0,200,81,.18);
    outline: none; z-index: 60;
}}
.tk-pill .tk-chg {{ font-size: 0.72rem; font-weight: 600; opacity: .9; }}
.tk-chg-pos {{ color: {ACCENT}; }}
.tk-chg-neg {{ color: #ff4444; }}
/* The card opens UPWARD (above the pill), over the metrics/caption — never into the table below.
   Streamlit renders st.dataframe in its own canvas that paints over absolutely-positioned HTML
   regardless of z-index, so opening downward got clipped/overlapped; opening up avoids it. */
.tk-card {{
    position: absolute; left: 0; bottom: calc(100% + 8px); top: auto; width: 280px; max-width: 78vw;
    z-index: 1000; text-align: left; font-weight: 400; white-space: normal;
    background: var(--background-color); color: var(--text-color);
    border: 1px solid rgba(128,128,128,.28); border-radius: 0.7rem; padding: 0.7rem 0.8rem;
    box-shadow: 0 -8px 28px rgba(0,0,0,.24);
    opacity: 0; visibility: hidden; transform: translateY(4px); pointer-events: none;
    transition: opacity .14s ease, transform .14s ease;
}}
.tk-pill:hover .tk-card, .tk-pill:focus .tk-card {{
    opacity: 1; visibility: visible; transform: translateY(0);
}}
.tk-card .tk-name {{ font-weight: 700; font-size: 0.92rem; margin-bottom: 0.12rem; }}
.tk-card .tk-meta {{ font-size: 0.73rem; opacity: .72; margin-bottom: 0.45rem; }}
.tk-card .tk-sum {{ font-size: 0.78rem; line-height: 1.38; opacity: .92; }}

/* ── Mobile / small-screen optimization ──────────────────────────────────────── */
@media (max-width: 640px) {{
    /* reclaim horizontal space (Streamlit's default desktop padding is large) */
    .block-container, [data-testid="stMainBlockContainer"] {{
        padding: 1rem 0.75rem 3rem !important;
    }}
    /* header stacks + shrinks instead of overflowing the viewport */
    .brand-header {{
        flex-direction: column; align-items: flex-start; gap: 0.5rem; padding: 0.85rem 1rem;
    }}
    .brand-header h1 {{ font-size: 1.35rem; }}
    .brand-header p {{ font-size: 0.78rem; }}
    /* let column rows wrap to 2-up instead of cramming many tiny columns across.
       Scoped to the MAIN area so the sidebar / nav is never affected. */
    [data-testid="stMain"] [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; gap: 0.5rem; }}
    [data-testid="stMain"] [data-testid="stColumn"],
    [data-testid="stMain"] [data-testid="column"] {{
        min-width: 45% !important; flex-basis: 45% !important;
    }}
    /* tighter but still-legible metric cards */
    [data-testid="stMetric"] {{ padding: 0.7rem 0.7rem 0.7rem 0.9rem; }}
    [data-testid="stMetricValue"] {{ font-size: 1.3rem; }}
    /* pill tabs scroll sideways rather than stacking into many rows */
    .stTabs [data-baseweb="tab-list"] {{
        overflow-x: auto; flex-wrap: nowrap; -webkit-overflow-scrolling: touch;
    }}
    h1 {{ font-size: 1.5rem; }}
    h2 {{ font-size: 1.2rem; }}
    h3 {{ font-size: 1.05rem; }}
    /* Mobile: the nav auto-collapses — make the » reopen control a big, obvious, pinned tap
       target so it's never missed. */
    [data-testid="stExpandSidebarButton"] {{
        position: fixed !important; top: 0.5rem; left: 0.5rem; z-index: 1000001 !important;
        padding: 0.45rem 0.6rem !important;
        background: linear-gradient(110deg, {ACCENT}, {ACCENT_DARK}) !important;
        border: none !important; box-shadow: 0 3px 12px rgba(0,0,0,.28) !important;
    }}
    [data-testid="stExpandSidebarButton"] svg {{ width: 1.5rem !important; height: 1.5rem !important; color: #fff !important; }}
    /* full-width, easy-to-tap buttons in the main area */
    [data-testid="stMain"] .stButton > button,
    [data-testid="stMain"] .stDownloadButton > button {{ width: 100% !important; }}
    /* the in-page Scan-now control rows shouldn't squeeze side-by-side on a phone */
    [data-testid="stMain"] [data-testid="stColumn"]:has(.stButton) {{ min-width: 100% !important; }}
}}
</style>
"""


def inject_theme() -> None:
    """Inject the global stylesheet. Call once, right after ``st.set_page_config``."""
    st.markdown(_BRAND_CSS, unsafe_allow_html=True)


def render_header() -> None:
    """Branded gradient title block, replacing the plain ``st.title``/markdown."""
    st.markdown(
        '<div class="brand-header">'
        '<div>'
        '<h1>📈 SwingTrade Pro</h1>'
        '<p>Multi-factor signals · Backtested edge · Dynamic Kelly sizing · Whale flow</p>'
        '</div>'
        '<div class="brand-live"><span class="dot"></span>LIVE</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _fmt_market_cap(value: Optional[float]) -> str:
    """Human-readable market cap (e.g. ``$1.2T`` / ``$840.0B`` / ``$95.0M``)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if v <= 0:
        return ""
    for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if v >= cutoff:
            return f"${v / cutoff:.1f}{suffix}"
    return f"${v:,.0f}"


def render_ticker_hovercards(profiles: Sequence[Dict]) -> None:
    """Render a strip of ticker pills, each revealing a company info card on hover/focus.

    ``profiles`` is a sequence of dicts with keys: ``symbol`` (required) and optional
    ``name``, ``sector``, ``industry``, ``market_cap``, ``change_pct``, ``summary``.
    Pure HTML/CSS (styles live in ``_BRAND_CSS``); no JS, no extra network calls — the caller
    passes data it already has. All text is HTML-escaped. Touch/keyboard users get the card on
    focus since the pills are ``tabindex=0``.
    """
    pills: List[str] = []
    for p in profiles:
        symbol = str(p.get("symbol") or "").strip()
        if not symbol:
            continue
        sym_e = html.escape(symbol)

        chg = p.get("change_pct")
        chg_html = ""
        try:
            if chg is not None:
                cls = "tk-chg-pos" if float(chg) >= 0 else "tk-chg-neg"
                chg_html = f'<span class="tk-chg {cls}">{float(chg):+.2f}%</span>'
        except (TypeError, ValueError):
            chg_html = ""

        name = html.escape(str(p.get("name") or symbol))
        meta_bits = [str(p.get("sector") or "").strip(), str(p.get("industry") or "").strip()]
        meta_bits = [b for b in meta_bits if b and b.lower() != "unknown"]
        mcap = _fmt_market_cap(p.get("market_cap"))
        if mcap:
            meta_bits.append(mcap)
        meta = html.escape(" · ".join(meta_bits)) if meta_bits else "—"

        summary = str(p.get("summary") or "").strip()
        if len(summary) > 240:
            summary = summary[:237].rstrip() + "…"
        summary_html = (
            f'<div class="tk-sum">{html.escape(summary)}</div>' if summary
            else '<div class="tk-sum" style="opacity:.6">No company description available.</div>'
        )

        pills.append(
            f'<span class="tk-pill" tabindex="0">{sym_e}{chg_html}'
            f'<span class="tk-card">'
            f'<div class="tk-name">{name}</div>'
            f'<div class="tk-meta">{meta}</div>'
            f'{summary_html}'
            f'</span></span>'
        )

    if pills:
        st.markdown(f'<div class="tk-hovers">{"".join(pills)}</div>', unsafe_allow_html=True)


def render_nav() -> str:
    """Render the sidebar navigation and return the selected page label.

    Prefers the icon-based ``streamlit-option-menu``; falls back to a styled ``st.radio`` (with
    emoji-prefixed labels) when the package isn't installed, so the app keeps working either way.
    The returned string always matches a ``NAV_ITEMS`` label / ``page ==`` branch in app.py.
    """
    labels = [item[0] for item in NAV_ITEMS]
    try:
        from streamlit_option_menu import option_menu
    except ImportError:
        return _render_radio_fallback(labels)

    icons = [item[1] for item in NAV_ITEMS]
    with st.sidebar:
        # Theme-agnostic styles: transparent container, neutral unselected text (readable on both
        # light and dark), primary-accent selection. Keeps "follow system" looking right.
        return option_menu(
            menu_title="Navigation",
            options=labels,
            icons=icons,
            menu_icon="compass",
            default_index=0,
            styles={
                "container": {"padding": "0.25rem 0", "background-color": "transparent"},
                "icon": {"color": ACCENT, "font-size": "1.0rem"},
                "nav-link": {
                    "font-size": "0.95rem",
                    "font-weight": "500",
                    "color": "#8a93a3",
                    "padding": "0.5rem 0.75rem",
                    "margin": "0.12rem 0",
                    "border-radius": "0.5rem",
                    "--hover-color": "rgba(0, 200, 81, 0.12)",
                },
                "nav-link-selected": {
                    "background-color": ACCENT,
                    "color": "#ffffff",
                    "font-weight": "600",
                    "box-shadow": "0 3px 14px rgba(0, 200, 81, 0.45)",
                },
            },
        )


def _render_radio_fallback(labels: List[str]) -> str:
    """Styled ``st.radio`` nav used when ``streamlit-option-menu`` isn't installed."""
    emoji = {item[0]: item[2] for item in NAV_ITEMS}
    with st.sidebar:
        st.markdown("#### Navigation")
        return st.radio(
            "Navigation",
            labels,
            format_func=lambda lbl: f"{emoji.get(lbl, '•')}  {lbl}",
            label_visibility="collapsed",
        )
