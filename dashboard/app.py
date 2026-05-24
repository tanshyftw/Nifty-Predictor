"""
Indian Stock Market Dashboard — Main Streamlit Application.

Run with:
    streamlit run dashboard/app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from dashboard.data_loader import (
    load_market_snapshot,
    _market_minute_bucket,
    attach_sectoral_data,
    attach_supply_chain_data,
)
from dashboard.components.header import render_header
from dashboard.components.market_overview import render_key_metrics, render_sectoral_heatmap
from dashboard.components.top_movers import render_top_movers
from dashboard.components.macro_panel import render_macro_panel
from dashboard.components.global_factors import render_global_indices, render_supply_chain
from dashboard.components.sector_deep_dive import render_sector_deep_dive
from dashboard.components.target_hunter import render_target_hunter_tiles
from dashboard.components.ema_chart import render_ema_chart
from dashboard.components.sector_momentum import render_sector_momentum
from dashboard.components.sector_monthly_returns import render_sector_monthly_returns
from dashboard.components.heatmap import render_heatmap
from dashboard.components.stock_view import render_stock_view
from dashboard.components.ownership_flows import render_ownership_flows
from dashboard.components.swing_ideas import render_swing_ideas_tiles


_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ========== Groww-inspired design system ========== */

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* Main page background — deep but not black */
.stApp { background: #0b0e14; }

/* Typography hierarchy */
h1 {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #e8ecf1 !important;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem !important;
}
h2 {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #d8dde5 !important;
    letter-spacing: -0.01em;
}
h3 {
    font-size: 1.0rem !important;
    font-weight: 600 !important;
    color: #d8dde5 !important;
}
h4 {
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    color: #c9cfd9 !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.005em;
}

p, span, label, div { color: #c9cfd9; }

/* Metric cards — clean, bordered, hover lift */
[data-testid="stMetric"] {
    background: #151922;
    border: 1px solid #232834;
    border-radius: 10px;
    padding: 14px 18px;
    transition: border-color 0.15s ease, transform 0.15s ease;
}
[data-testid="stMetric"]:hover {
    border-color: #2f3645;
    transform: translateY(-1px);
}
[data-testid="stMetric"] label {
    font-size: 0.68rem !important;
    color: #7a8294 !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: #e8ecf1 !important;
    letter-spacing: -0.01em;
}
[data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
    font-weight: 500;
}

/* Tabs — Groww-style underline tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: transparent !important;
    border-bottom: 1px solid #232834 !important;
    padding: 0 !important;
    border-radius: 0 !important;
    margin-bottom: 1rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 0 !important;
    padding: 10px 24px !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: #7a8294 !important;
    border-bottom: 2px solid transparent !important;
    margin: 0 !important;
    transition: all 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #c9cfd9 !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #00d09c !important;
    border-bottom: 2px solid #00d09c !important;
    background: transparent !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* Dataframes — clean borders, centered numeric cells */
[data-testid="stDataFrame"] {
    font-size: 0.82rem;
    border: 1px solid #232834;
    border-radius: 8px;
    overflow: hidden;
}
[data-testid="stDataFrame"] div[role="grid"] {
    background: #0f131c !important;
}
/* Center-align numeric column cells & headers */
[data-testid="stDataFrame"] [role="columnheader"],
[data-testid="stDataFrame"] [role="gridcell"] {
    display: flex !important;
    align-items: center !important;
}
[data-testid="stDataFrame"] [role="columnheader"] {
    font-weight: 600 !important;
    color: #c9cfd9 !important;
    letter-spacing: 0.02em;
}
/* Right-align numeric cells (Streamlit marks them with class containing "right") */
[data-testid="stDataFrame"] [role="gridcell"][class*="right"] {
    justify-content: flex-end !important;
    padding-right: 12px !important;
}
[data-testid="stDataFrame"] [role="columnheader"][class*="right"] {
    justify-content: flex-end !important;
    padding-right: 12px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a0d13 !important;
    border-right: 1px solid #232834;
}
section[data-testid="stSidebar"] h1 {
    font-size: 1.15rem !important;
    color: #e8ecf1 !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.88rem !important;
    color: #c9cfd9 !important;
}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
    padding: 4px 0;
}

/* Buttons */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    border: 1px solid #232834 !important;
    background: #151922 !important;
    color: #e8ecf1 !important;
    transition: all 0.15s ease;
    padding: 8px 16px !important;
}
.stButton > button:hover {
    border-color: #00d09c !important;
    color: #00d09c !important;
    background: #151922 !important;
}
.stButton > button[kind="primary"] {
    background: #00d09c !important;
    color: #0b0e14 !important;
    border: none !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #00b386 !important;
    color: #0b0e14 !important;
}

/* Expanders — clean cards */
[data-testid="stExpander"] {
    background: #151922;
    border: 1px solid #232834 !important;
    border-radius: 10px !important;
    margin-bottom: 8px;
}
[data-testid="stExpander"] summary {
    font-size: 0.9rem !important;
    padding: 10px 14px !important;
}
[data-testid="stExpander"] summary:hover {
    background: #1a1f2b;
}

/* Dividers — subtle */
hr { border-color: #232834 !important; margin: 1rem 0 !important; }

/* Captions — muted */
[data-testid="stCaptionContainer"], .caption, small {
    color: #6b7587 !important;
    font-size: 0.78rem !important;
}

/* Info/Success/Warning alerts */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border: 1px solid #232834 !important;
}

/* Progress bar in columns */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #00d09c 0%, #00b386 100%) !important;
}

/* Radio buttons in sidebar */
[role="radiogroup"] {
    gap: 2px !important;
}

/* Slider */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #00d09c !important;
    border: 2px solid #00d09c !important;
}

/* Hide Streamlit footer/menu for cleaner look */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

/* Tighter vertical rhythm */
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
.element-container { margin-bottom: 0.5rem; }

/* Column spacing */
[data-testid="column"] { padding: 0 0.5rem; }

/* ========== Pill-style Mode toggle (segmented control + horizontal radio) ========== */
.mode-toggle-wrap { margin-bottom: 14px; }

/* Segmented control (Streamlit ≥1.40) */
.mode-toggle-wrap [data-testid="stSegmentedControl"] {
    background: #151922;
    border: 1px solid #232834;
    border-radius: 10px;
    padding: 4px;
    display: inline-flex;
}
.mode-toggle-wrap [data-testid="stSegmentedControl"] button {
    background: transparent !important;
    border: none !important;
    color: #7a8294 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 8px 20px !important;
    border-radius: 7px !important;
    transition: all 0.15s ease;
}
.mode-toggle-wrap [data-testid="stSegmentedControl"] button[aria-pressed="true"],
.mode-toggle-wrap [data-testid="stSegmentedControl"] button[data-selected="true"] {
    background: #00d09c !important;
    color: #0b0e14 !important;
    font-weight: 600 !important;
}

/* Fallback: horizontal radio styled as pills */
.mode-toggle-wrap [data-testid="stRadio"] > div[role="radiogroup"] {
    flex-direction: row !important;
    gap: 0 !important;
    background: #151922;
    border: 1px solid #232834;
    border-radius: 10px;
    padding: 4px;
    display: inline-flex;
}
.mode-toggle-wrap [data-testid="stRadio"] label {
    margin: 0 !important;
    padding: 8px 20px !important;
    border-radius: 7px !important;
    cursor: pointer;
    color: #7a8294 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    transition: all 0.15s ease;
}
.mode-toggle-wrap [data-testid="stRadio"] label:hover {
    color: #c9cfd9 !important;
}
.mode-toggle-wrap [data-testid="stRadio"] label:has(input:checked) {
    background: #00d09c !important;
    color: #0b0e14 !important;
    font-weight: 600 !important;
}
.mode-toggle-wrap [data-testid="stRadio"] label > div:first-child {
    display: none !important;  /* hide the actual radio circle */
}

/* Top-of-page Refresh button — match pill toggle height/style */
.refresh-btn-wrap { margin-bottom: 14px; }
.refresh-btn-wrap .stButton > button {
    background: #151922 !important;
    border: 1px solid #232834 !important;
    color: #c9cfd9 !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    border-radius: 10px !important;
}
.refresh-btn-wrap .stButton > button:hover {
    border-color: #00d09c !important;
    color: #00d09c !important;
    background: #151922 !important;
}
</style>
"""


def main():
    st.set_page_config(
        page_title="Nifty Market Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(
            '<div style="padding:4px 0 12px 0;">'
            '<div style="font-size:1.15rem; font-weight:700; color:#e8ecf1; letter-spacing:-0.01em;">'
            '<span style="color:#00d09c;">●</span> Nifty Terminal</div>'
            '<div style="font-size:0.78rem; color:#7a8294; margin-top:2px;">Indian equity market intelligence</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        if st.button("Refresh data", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()

        st.markdown('<div style="margin-top:24px;"></div>', unsafe_allow_html=True)

        st.caption(
            "Data: Yahoo Finance, NSE India. "
            "Free sources, no API keys. Auto-refreshes every 5 min."
        )

    # --- Toggles (top of page, pill-style) ---
    toggle_left, toggle_mid, toggle_right = st.columns([3, 2, 1])
    mode_options = [
        "Market Overview",
        "Stock View",
        "Ownership Radar",
    ]
    view_options = ["Daily", "Weekly"]

    with toggle_left:
        st.markdown('<div class="mode-toggle-wrap">', unsafe_allow_html=True)
        mode = st.radio(
            "Mode",
            mode_options,
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key="mode_toggle",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with toggle_mid:
        if mode == "Market Overview":
            st.markdown('<div class="mode-toggle-wrap" style="text-align:right;">', unsafe_allow_html=True)
            view = st.radio(
                "Timeframe",
                view_options,
                index=0,
                horizontal=True,
                label_visibility="collapsed",
                key="view_toggle",
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            view = "Daily"

    with toggle_right:
        st.markdown('<div class="refresh-btn-wrap">', unsafe_allow_html=True)
        if st.button("↻  Refresh", use_container_width=True, key="refresh_top"):
            st.cache_data.clear()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Stock View mode ---
    if mode == "Stock View":
        # No snapshot or sector fan-out needed — the stock view is its own
        # standalone surface; pulling the full market snapshot would add
        # seconds of latency to a flow the user might use for a quick lookup.
        render_stock_view()
        return

    # --- Ownership Radar mode ---
    if mode == "Ownership Radar":
        render_ownership_flows()
        return

    # --- Market Overview mode ---
    view_key = view.lower()
    snapshot = load_market_snapshot(view=view_key, _bucket=_market_minute_bucket())

    render_header(snapshot)
    render_key_metrics(snapshot)

    st.markdown('<div style="margin-top:24px;"></div>', unsafe_allow_html=True)

    tab_market, tab_heatmap, tab_sectors, tab_macro = st.tabs([
        "Market",
        "Heatmap",
        "Sectors",
        "Macro & Global",
    ])

    with tab_market:
        render_ema_chart(view=view_key)
        _spacer()
        render_top_movers(snapshot)
        _spacer()
        render_target_hunter_tiles(limit=6)
        _spacer()
        render_swing_ideas_tiles(limit=6)

    with tab_heatmap:
        render_heatmap(view=view_key, snapshot=snapshot)

    with tab_sectors:
        attach_sectoral_data(snapshot)
        render_sectoral_heatmap(snapshot)
        _spacer()
        render_sector_monthly_returns()
        _spacer()
        render_sector_momentum()
        _spacer()
        render_sector_deep_dive(snapshot)

    with tab_macro:
        # render_supply_chain also reads snapshot.sectoral_data for the
        # cross-reference table, so both lazy loaders fire when this tab opens.
        attach_sectoral_data(snapshot)
        attach_supply_chain_data(snapshot)
        render_macro_panel(snapshot)
        _spacer()
        render_global_indices(snapshot)
        _spacer()
        render_supply_chain(snapshot)


def _spacer():
    """Consistent vertical gap between tab sections."""
    st.markdown('<div style="margin-top:32px;"></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
