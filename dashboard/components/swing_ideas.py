"""Rule-based swing trade ideas view."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from loguru import logger

from config.largecap_extra_tickers import LARGECAP_NEXT50_STOCKS
from config.midcap_tickers import MIDCAP_STOCKS
from config.nifty50_tickers import NIFTY50_STOCKS
from dashboard.config import CACHE_TTL_SECONDS
from dashboard.data_loader import _market_minute_bucket
from signals.swing_engine import build_swing_ideas


_UNIVERSES = {
    "Nifty 50": NIFTY50_STOCKS,
    "Largecap+": LARGECAP_NEXT50_STOCKS,
    "Midcap": MIDCAP_STOCKS,
}

_CSS = """
<style>
.swing-card {
    background:#11151d;
    border:1px solid #232834;
    border-left:3px solid #00d09c;
    border-radius:10px;
    padding:12px 14px;
    margin-bottom:10px;
}
.swing-card .top {
    display:flex;
    justify-content:space-between;
    gap:12px;
    align-items:flex-start;
}
.swing-card .symbol {
    color:#e8ecf1;
    font-size:1rem;
    font-weight:700;
}
.swing-card .meta {
    color:#7a8294;
    font-size:0.76rem;
    margin-top:2px;
}
.swing-card .score {
    color:#00d09c;
    font-weight:700;
    font-variant-numeric:tabular-nums;
}
.swing-card .levels {
    color:#c9cfd9;
    font-size:0.8rem;
    margin-top:8px;
    line-height:1.6;
}
.swing-card .why {
    color:#9aa3b2;
    font-size:0.78rem;
    margin-top:8px;
    line-height:1.5;
}
</style>
"""


def render_swing_ideas():
    """Render a daily action board for recurring swing logic."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.title("Swing Ideas")
    st.caption("Repeatable rule-based setups with entry, stop, targets, and invalidation.")

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.0])
    with c1:
        universe = st.selectbox("Universe", list(_UNIVERSES.keys()) + ["All"], key="swing_universe")
    with c2:
        min_score = st.slider("Min setup score", 40, 90, 60, 5, key="swing_min_score")
    with c3:
        setup_filter = st.multiselect(
            "Setup",
            ["Momentum breakout", "Trend pullback", "Mean-reversion bounce"],
            default=["Momentum breakout", "Trend pullback", "Mean-reversion bounce"],
            key="swing_setup_filter",
        )
    with c4:
        limit = st.number_input("Ideas", min_value=5, max_value=50, value=20, step=5, key="swing_limit")

    stocks = _stocks_for_universe(universe)
    with st.spinner(f"Scanning {len(stocks)} stocks for recurring swing setups..."):
        histories, index_close = _load_swing_history(universe, _market_minute_bucket())

    metadata = {symbol: (meta[1], meta[2]) for symbol, meta in stocks.items()}
    ideas = build_swing_ideas(histories, metadata, index_close=index_close, min_score=float(min_score))
    if setup_filter and not ideas.empty:
        ideas = ideas[ideas["setup"].isin(setup_filter)].copy()

    if ideas.empty:
        st.info("No swing ideas match the current rule filters. Lower the score or switch universe.")
        _render_rulebook()
        return

    ideas = ideas.head(int(limit))
    _render_summary(ideas)

    tab_board, tab_table, tab_rulebook = st.tabs(["Action board", "Table", "Rulebook"])
    with tab_board:
        _render_cards(ideas)
    with tab_table:
        _render_table(ideas)
    with tab_rulebook:
        _render_rulebook()


def _stocks_for_universe(universe: str) -> dict:
    if universe == "All":
        merged = dict(MIDCAP_STOCKS)
        merged.update(LARGECAP_NEXT50_STOCKS)
        merged.update(NIFTY50_STOCKS)
        return merged
    return _UNIVERSES[universe]


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _load_swing_history(universe: str, _bucket: str = "") -> tuple[dict[str, pd.DataFrame], pd.Series]:
    del _bucket
    stocks = _stocks_for_universe(universe)
    tickers = " ".join(meta[0] for meta in stocks.values())
    try:
        data = yf.download(
            tickers,
            period="1y",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        nifty = yf.download(
            "^NSEI",
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        logger.warning(f"swing history fetch failed: {exc}")
        return {}, pd.Series(dtype=float)

    histories: dict[str, pd.DataFrame] = {}
    for symbol, (ticker, _, _) in stocks.items():
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker not in data.columns.get_level_values(0):
                    continue
                sub = data[ticker].dropna(how="all")
            else:
                sub = data.dropna(how="all")
            if not sub.empty:
                histories[symbol] = sub
        except Exception:
            continue

    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = [c[0] if isinstance(c, tuple) else c for c in nifty.columns]
    index_close = nifty["Close"].dropna() if "Close" in nifty.columns else pd.Series(dtype=float)
    return histories, index_close


def _render_summary(ideas: pd.DataFrame):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ideas found", f"{len(ideas)}")
    m2.metric("Avg score", f"{ideas['score'].mean():.0f}")
    m3.metric("Best setup", str(ideas.iloc[0]["setup"]), f"{ideas.iloc[0]['symbol']}")
    top_sector = ideas["sector"].value_counts().idxmax() if not ideas.empty else "-"
    m4.metric("Most active sector", top_sector, f"{ideas['sector'].value_counts().max()} ideas")


def _render_cards(ideas: pd.DataFrame):
    for _, row in ideas.iterrows():
        st.markdown(
            f"""
            <div class="swing-card">
              <div class="top">
                <div>
                  <div class="symbol">{row['symbol']} · {row['setup']}</div>
                  <div class="meta">{row['company']} · {row['sector']} · {row['horizon']}</div>
                </div>
                <div class="score">{row['score']:.0f}/100</div>
              </div>
              <div class="levels">
                Close ₹{row['close']:,.2f} · Entry ₹{row['entry']:,.2f} ·
                Stop ₹{row['stop']:,.2f} · T1 ₹{row['target_1']:,.2f} ·
                T2 ₹{row['target_2']:,.2f} · R:R {row['reward_risk']:.1f}
              </div>
              <div class="why"><b>Why:</b> {row['rationale']}<br/><b>Invalidation:</b> {row['invalidation']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_table(ideas: pd.DataFrame):
    table = ideas[[
        "symbol",
        "company",
        "sector",
        "setup",
        "score",
        "close",
        "entry",
        "stop",
        "target_1",
        "target_2",
        "reward_risk",
        "horizon",
        "rationale",
        "invalidation",
    ]].copy()
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "score": st.column_config.NumberColumn(format="%.0f"),
            "close": st.column_config.NumberColumn(format="%.2f"),
            "entry": st.column_config.NumberColumn(format="%.2f"),
            "stop": st.column_config.NumberColumn(format="%.2f"),
            "target_1": st.column_config.NumberColumn(format="%.2f"),
            "target_2": st.column_config.NumberColumn(format="%.2f"),
            "reward_risk": st.column_config.NumberColumn(format="%.2f"),
        },
    )


def _render_rulebook():
    rows = pd.DataFrame([
        {
            "Setup": "Momentum breakout",
            "Entry logic": "Close clears prior 20-day high with trend, volume, RSI, and relative strength confirmation.",
            "Risk": "Stop below recent support or 1.5 ATR.",
            "Horizon": "2-4 weeks",
        },
        {
            "Setup": "Trend pullback",
            "Entry logic": "Primary trend remains up while price resets near EMA-20 with healthy RSI.",
            "Risk": "Stop below support or EMA-50.",
            "Horizon": "1-3 weeks",
        },
        {
            "Setup": "Mean-reversion bounce",
            "Entry logic": "Oversold stock near 20-day low, without deep long-trend damage.",
            "Risk": "Stop around 1.2 ATR below close.",
            "Horizon": "3-10 sessions",
        },
    ])
    st.dataframe(rows, use_container_width=True, hide_index=True)
