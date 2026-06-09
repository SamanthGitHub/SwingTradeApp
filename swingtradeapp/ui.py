"""Presentation layer: app theming, branded header, and the icon-based sidebar navigation.

Kept separate from ``app.py`` so the 1900-line dashboard stays focused on data/logic. Everything
here is purely visual and **auto light/dark aware** — colors come from Streamlit theme CSS
variables (``var(--secondary-background-color)`` etc.) or ``@media (prefers-color-scheme)`` so the
app follows the OS setting (no hardcoded background that breaks in one mode).

The icon menu uses ``streamlit-option-menu`` when installed and **degrades gracefully** to a
styled ``st.radio`` when it isn't (same fallback philosophy as the optional AI features), so the
app never crashes if the dependency hasn't been installed into the venv yet.
"""

from typing import List, Tuple

import streamlit as st

# Brand accent (green = buy/positive, consistent with the rest of the palette).
ACCENT = "#00c851"
ACCENT_DARK = "#00a843"

# (page label, Bootstrap icon name, emoji fallback). The LABEL must stay byte-identical to the
# `if page == "..."` branches in app.py — it is the routing key.
NAV_ITEMS: List[Tuple[str, str, str]] = [
    ("Screener", "graph-up", "📈"),
    ("Pre-Market Movers", "sunrise", "🌅"),
    ("Live Movers", "lightning-charge", "⚡"),
    ("After-Hours & IPOs", "moon-stars", "🌙"),
    ("Whale Movements", "water", "🐋"),
    ("Options Flow", "graph-up-arrow", "🎯"),
    ("Predictions", "magic", "🔮"),
    ("Auto Watchlist", "stars", "⭐"),
    ("ETF Screener", "grid-3x3-gap", "🗂️"),
    ("Market Events", "newspaper", "📰"),
    ("YouTube", "youtube", "📺"),
    ("Heat Map", "fire", "🔥"),
    ("Watchlists", "bookmark-star", "🔖"),
    ("Compare", "bar-chart-line", "📊"),
    ("P&L Tracker", "journal-text", "📒"),
    ("Alerts", "bell", "🔔"),
    ("Information", "info-circle", "ℹ️"),
    ("Settings", "gear", "⚙️"),
]

_BRAND_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, button, input, select, textarea {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

/* Hide default Streamlit chrome that reads as "unfinished". Keep the header transparent
   (not display:none) so the sidebar collapse control survives. */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
[data-testid="stToolbar"] {{display: none;}}
[data-testid="stHeader"] {{background: transparent;}}

/* Branded gradient header band. */
.brand-header {{
    padding: 1.0rem 1.4rem;
    border-radius: 0.8rem;
    margin-bottom: 1.0rem;
    background: linear-gradient(110deg, {ACCENT_DARK} 0%, #0b7d6f 55%, #0b6cad 100%);
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.18);
}}
.brand-header h1 {{
    color: #ffffff; font-size: 1.7rem; font-weight: 700; margin: 0; letter-spacing: -0.01em;
}}
.brand-header p {{
    color: rgba(255, 255, 255, 0.88); font-size: 0.92rem; font-weight: 500; margin: 0.25rem 0 0 0;
}}

/* Metric "cards" — adapt to light/dark via theme CSS variables. */
[data-testid="stMetric"] {{
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.18);
    border-radius: 0.7rem;
    padding: 0.85rem 1.0rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
}}
[data-testid="stMetricLabel"] {{opacity: 0.75; font-weight: 600;}}

/* Tabs + sidebar polish. */
.stTabs [data-baseweb="tab-list"] button {{font-size: 15px; font-weight: 600;}}
[data-testid="stSidebar"] {{border-right: 1px solid rgba(128, 128, 128, 0.15);}}
.metric-positive {{color: {ACCENT}; font-weight: bold;}}
.metric-negative {{color: #ff4444; font-weight: bold;}}

/* Fallback radio nav (used only when streamlit-option-menu is absent). */
[data-testid="stSidebar"] .stRadio label {{
    font-size: 0.95rem; padding: 0.15rem 0; font-weight: 500;
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
        '<h1>📈 SwingTrade Pro</h1>'
        '<p>Multi-factor signals · Backtested edge · Dynamic Kelly sizing · Whale flow</p>'
        '</div>',
        unsafe_allow_html=True,
    )


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
