"""Page renderers extracted from the app.py monolith, one module per screen.

``PAGES`` maps the **exact nav label** (must stay byte-identical to ``ui.NAV_ITEMS``)
to a ``render(ctx)`` callable. ``run_dashboard`` tries this registry first and falls
back to its remaining ``elif`` chain, so pages migrate here one at a time with the app
working at every step. Import discipline: screens import ``services``, never ``app``.

NOTE: this package is deliberately NOT named ``pages/`` — a ``pages/`` directory next
to the entry point would activate Streamlit's native multipage router and fight the
app's own sidebar nav.
"""

from screens._ctx import Ctx
from screens import (
    alpha_engine_page,
    morning_insights,
    screener,
    settings_page,
    signal_stack,
    youtube_page,
)

PAGES = {
    "Screener": screener.render,
    "Morning Insights": morning_insights.render,
    "YouTube": youtube_page.render,
    "Signal Stack": signal_stack.render,
    "Alpha Engine": alpha_engine_page.render,
    "Settings": settings_page.render,
}

__all__ = ["Ctx", "PAGES"]
