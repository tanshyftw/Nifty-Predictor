"""Rule-based swing trade ideas view."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from loguru import logger
from html import escape

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

# Color + plain-English mapping per setup type
_SETUP_META = {
    "Momentum breakout": {
        "color": "#00d09c",
        "icon": "▲",
        "tagline": "Stock is breaking out to new highs with strength",
    },
    "Trend pullback": {
        "color": "#5eb8ff",
        "icon": "↗",
        "tagline": "Healthy dip inside an ongoing uptrend",
    },
    "Mean-reversion bounce": {
        "color": "#f0b429",
        "icon": "⤴",
        "tagline": "Oversold name due for a bounce",
    },
}

_CSS = """
<style>
/* Hero explainer card */
.swing-hero {
    background: linear-gradient(135deg, #11151d 0%, #131826 60%, #0f1a25 100%);
    border: 1px solid #232834;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 18px;
    display: flex;
    gap: 18px;
    align-items: flex-start;
}
.swing-hero .icon {
    width: 44px; height: 44px;
    border-radius: 12px;
    background: rgba(0,208,156,0.10);
    border: 1px solid rgba(0,208,156,0.25);
    display:flex; align-items:center; justify-content:center;
    color:#00d09c; font-weight:700; font-size:1.05rem;
    flex-shrink:0;
}
.swing-hero .title { color:#e8ecf1; font-size:1.0rem; font-weight:600; margin-bottom:4px; }
.swing-hero .body  { color:#9aa3b2; font-size:0.84rem; line-height:1.55; }
.swing-hero .body b { color:#cfd6e1; }

/* Headline stat tiles */
.stat-row { display:flex; gap:12px; margin-bottom:14px; flex-wrap:wrap; }
.stat-tile {
    flex: 1 1 200px;
    background:#11151d;
    border:1px solid #232834;
    border-radius:12px;
    padding:14px 16px;
}
.stat-tile .label {
    color:#7a8294;
    font-size:0.7rem;
    font-weight:500;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom:8px;
}
.stat-tile .value {
    color:#e8ecf1;
    font-size:1.4rem;
    font-weight:700;
    letter-spacing:-0.01em;
    line-height:1.1;
}
.stat-tile .sub {
    color:#9aa3b2;
    font-size:0.78rem;
    margin-top:6px;
}
.stat-tile .sub b { color:#cfd6e1; }

/* Glossary chips */
.glossary-row {
    display:flex; gap:8px; flex-wrap:wrap; margin: 6px 0 16px 0;
}
.gl-chip {
    background:#11151d;
    border:1px solid #232834;
    border-radius:8px;
    padding:7px 11px;
    color:#9aa3b2;
    font-size:0.76rem;
    line-height:1.4;
}
.gl-chip b { color:#e8ecf1; font-weight:600; }

/* Setup type badges row (above the cards) */
.setup-legend { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:14px; }
.setup-badge {
    padding:6px 11px;
    border-radius:999px;
    font-size:0.76rem;
    font-weight:500;
    border:1px solid;
    background:rgba(255,255,255,0.02);
}

/* Lens / control strip */
.lens-strip {
    background:#11151d;
    border:1px solid #232834;
    border-radius:12px;
    padding:12px 14px;
    margin-bottom:14px;
}

/* Idea card — new visual style */
.swing-card {
    background: linear-gradient(180deg, #131722 0%, #11151d 100%);
    border: 1px solid #232834;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 12px;
    transition: border-color 0.18s ease, transform 0.18s ease;
    position: relative;
    overflow: hidden;
}
.swing-card:hover {
    border-color: #2f3645;
    transform: translateY(-1px);
}
.swing-card .accent-strip {
    position:absolute; left:0; top:0; bottom:0; width:3px;
}
.swing-card .head {
    display:flex; justify-content:space-between; align-items:flex-start;
    gap: 14px; margin-bottom: 12px;
}
.swing-card .symbol {
    color:#e8ecf1; font-size:1.05rem; font-weight:700; letter-spacing:-0.01em;
}
.swing-card .company {
    color:#9aa3b2; font-size:0.78rem; margin-top:2px;
}
.swing-card .setup-pill {
    display:inline-block; padding:3px 9px;
    border-radius:999px; font-size:0.7rem; font-weight:600;
    margin-left:8px; vertical-align:middle;
}
.swing-card .horizon-pill {
    display:inline-block; padding:3px 9px;
    border-radius:999px; font-size:0.7rem; font-weight:500;
    margin-left:6px; vertical-align:middle;
    background:#0d1119; border:1px solid #232834; color:#9aa3b2;
}
.swing-card .score-wrap {
    text-align:right; flex-shrink:0;
}
.swing-card .score-num {
    font-size:1.4rem; font-weight:700;
    color:#e8ecf1;
    font-variant-numeric: tabular-nums;
}
.swing-card .score-label {
    font-size:0.7rem; color:#7a8294;
    text-transform:uppercase; letter-spacing:0.06em;
    margin-top:-2px;
}
.swing-card .score-bar {
    margin-top:6px; height:5px; width:100px;
    background:#1a1f2b; border-radius:3px; overflow:hidden;
    margin-left:auto;
}
.swing-card .score-bar > div { height:100%; border-radius:3px; }

/* Price ladder visual */
.ladder { display:flex; gap:10px; margin: 10px 0 8px 0; flex-wrap:wrap; }
.lvl {
    flex: 1 1 110px;
    background:#0d1119;
    border:1px solid #232834;
    border-radius:8px;
    padding: 8px 10px;
}
.lvl .lvl-label {
    color:#7a8294; font-size:0.66rem;
    text-transform:uppercase; letter-spacing:0.06em;
    margin-bottom:3px; display:flex; justify-content:space-between; align-items:center;
}
.lvl .lvl-value {
    color:#e8ecf1; font-size:0.95rem; font-weight:600;
    font-variant-numeric: tabular-nums;
}
.lvl .lvl-help { color:#7a8294; font-size:0.66rem; }
.lvl-close { border-color:#2f3645; }
.lvl-entry { border-color:rgba(0,208,156,0.35); }
.lvl-stop  { border-color:rgba(235,87,87,0.30); }
.lvl-t1    { border-color:rgba(94,184,255,0.30); }
.lvl-t2    { border-color:rgba(94,184,255,0.45); }

.rr-pill {
    display:inline-block; padding:2px 9px;
    border-radius:999px; font-size:0.72rem;
    background:#0d1119; border:1px solid #232834;
    color:#cfd6e1; margin-left:6px;
}
.rr-pill.good { background:rgba(0,208,156,0.10); border-color:rgba(0,208,156,0.35); color:#00d09c; }
.rr-pill.ok   { background:rgba(240,180,41,0.10); border-color:rgba(240,180,41,0.35); color:#f0b429; }

.swing-card .why {
    color:#cfd6e1; font-size:0.82rem; line-height:1.55; margin-top:8px;
}
.swing-card .why b { color:#e8ecf1; }
.swing-card .invalid {
    color:#9aa3b2; font-size:0.78rem; line-height:1.5; margin-top:6px;
    padding-top:8px; border-top:1px dashed #232834;
}
.swing-card .invalid b { color:#eb5757; }
</style>
"""

_OVERVIEW_SWING_CSS = """
<style>
.swing-tile-grid {
    display:grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap:12px;
    margin:10px 0 4px 0;
}
.swing-tile {
    background:linear-gradient(180deg, #141924 0%, #0f131c 100%);
    border:1px solid #232834;
    border-radius:10px;
    padding:14px 16px;
    position:relative;
    overflow:hidden;
}
.swing-tile:hover { border-color:#2f3645; }
.swing-tile .strip { position:absolute;left:0;top:0;bottom:0;width:3px; }
.swing-tile .symbol { color:#e8ecf1;font-size:1.0rem;font-weight:700; }
.swing-tile .company { color:#7a8294;font-size:0.76rem;margin-top:2px;min-height:18px; }
.swing-tile .score { color:#e8ecf1;font-size:1.25rem;font-weight:700;margin-top:10px; }
.swing-tile .setup { font-size:0.76rem;font-weight:600;margin-top:4px; }
.swing-tile .meta { color:#c9cfd9;font-size:0.78rem;line-height:1.45;margin-top:8px; }
.swing-tile .muted { color:#7a8294; }
@media (max-width: 900px) {
    .swing-tile-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
    .swing-tile-grid { grid-template-columns: 1fr; }
}
</style>
"""


def render_swing_ideas_tiles(limit: int = 6):
    """Render compact swing setup cards inside Market Overview."""
    st.markdown(_OVERVIEW_SWING_CSS, unsafe_allow_html=True)
    st.subheader("Swing Ideas")

    load_key = "overview_swing_ideas_loaded"
    if not st.session_state.get(load_key, False):
        st.markdown(
            '<div class="overview-load-card">'
            '<div style="color:#e8ecf1;font-weight:600;margin-bottom:4px;">Technical setup tiles</div>'
            '<div style="color:#7a8294;font-size:0.82rem;">Loads on demand so Market Overview stays fast.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("Load swing tiles", key="load_swing_tiles"):
            st.session_state[load_key] = True
            st.rerun()
        return

    universe = "Nifty 50"
    stocks = _stocks_for_universe(universe)
    with st.spinner("Scanning swing setups..."):
        histories, index_close = _load_swing_history(universe, _market_minute_bucket())

    metadata = {symbol: (meta[1], meta[2]) for symbol, meta in stocks.items()}
    ideas = build_swing_ideas(histories, metadata, index_close=index_close, min_score=60.0)

    if ideas.empty:
        st.info("No swing setups passed the default quality bar right now.")
        return

    tiles = []
    for _, row in ideas.head(limit).iterrows():
        setup = str(row["setup"])
        meta = _SETUP_META.get(setup, {"color": "#7a8294", "icon": "", "tagline": ""})
        color = meta["color"]
        tiles.append(
            "<div class='swing-tile'>"
            f"<div class='strip' style='background:{color};'></div>"
            f"<div class='symbol'>{escape(str(row['symbol']))}</div>"
            f"<div class='company'>{escape(str(row['company']))} · {escape(str(row['sector']))}</div>"
            f"<div class='score' style='color:{color};'>{float(row['score']):.0f}<span style='font-size:0.78rem;color:#7a8294;'>/100</span></div>"
            f"<div class='setup' style='color:{color};'>{escape(meta['icon'])} {escape(setup)}</div>"
            f"<div class='meta'>Entry ₹{float(row['entry']):,.2f} · Stop ₹{float(row['stop']):,.2f}</div>"
            f"<div class='meta muted'>T1 ₹{float(row['target_1']):,.2f} · R:R {float(row['reward_risk']):.1f}x · {escape(str(row['horizon']))}</div>"
            "</div>"
        )
    st.markdown(f"<div class='swing-tile-grid'>{''.join(tiles)}</div>", unsafe_allow_html=True)
    st.caption("Default scan: Nifty 50, minimum setup quality 60/100, top ranked ideas.")


def render_swing_ideas():
    """Render a daily action board for recurring swing logic."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.title("Swing Ideas")
    st.caption("Daily action board — repeatable setups with clear entry, stop, and targets.")

    _render_hero_intro()

    st.markdown('<div class="lens-strip">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.0])
    with c1:
        universe = st.selectbox(
            "Universe",
            list(_UNIVERSES.keys()) + ["All"],
            key="swing_universe",
            help="The stock list to scan.",
        )
    with c2:
        min_score = st.slider(
            "Minimum setup quality",
            40, 90, 60, 5,
            key="swing_min_score",
            help="Filter to only show setups that meet a minimum score (out of 100).",
        )
    with c3:
        setup_filter = st.multiselect(
            "Setup types",
            ["Momentum breakout", "Trend pullback", "Mean-reversion bounce"],
            default=["Momentum breakout", "Trend pullback", "Mean-reversion bounce"],
            key="swing_setup_filter",
        )
    with c4:
        limit = st.number_input(
            "Show top",
            min_value=5, max_value=50, value=20, step=5,
            key="swing_limit",
            help="How many ideas to display.",
        )
    st.markdown('</div>', unsafe_allow_html=True)

    stocks = _stocks_for_universe(universe)
    with st.spinner(f"Scanning {len(stocks)} stocks for swing setups..."):
        histories, index_close = _load_swing_history(universe, _market_minute_bucket())

    metadata = {symbol: (meta[1], meta[2]) for symbol, meta in stocks.items()}
    ideas = build_swing_ideas(histories, metadata, index_close=index_close, min_score=float(min_score))
    if setup_filter and not ideas.empty:
        ideas = ideas[ideas["setup"].isin(setup_filter)].copy()

    if ideas.empty:
        st.info(
            "No setups match your filters right now. Try lowering the minimum score, "
            "switching to a broader universe, or selecting more setup types."
        )
        _render_glossary_chips()
        _render_rulebook()
        return

    ideas = ideas.head(int(limit))
    _render_summary(ideas)
    _render_glossary_chips()

    tab_board, tab_table, tab_rulebook = st.tabs(["Action board", "Table", "How to read this"])
    with tab_board:
        _render_setup_legend(ideas)
        _render_cards(ideas)
    with tab_table:
        _render_table(ideas)
    with tab_rulebook:
        _render_rulebook()


def _render_hero_intro():
    st.markdown(
        """
        <div class="swing-hero">
            <div class="icon">▲</div>
            <div>
                <div class="title">What is a swing idea?</div>
                <div class="body">
                    A <b>swing trade</b> is a short-term position (a few days to a few weeks) that aims to capture
                    one clean move in a stock. We scan the market every day for three time-tested setups, score each one,
                    and give you a clear plan: <b>where to enter</b>, <b>where to stop out</b>, and <b>where to take profit</b>.
                    No black-box calls — every idea shows the reasoning and what would invalidate it.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    avg_score = float(ideas["score"].mean())
    top_setup = str(ideas.iloc[0]["setup"])
    top_symbol = str(ideas.iloc[0]["symbol"])
    top_sector = ideas["sector"].value_counts().idxmax() if not ideas.empty else "-"
    top_sector_count = int(ideas["sector"].value_counts().max()) if not ideas.empty else 0
    avg_rr = float(ideas["reward_risk"].mean()) if "reward_risk" in ideas.columns else 0.0

    score_label = (
        "High quality" if avg_score >= 75 else
        "Solid" if avg_score >= 60 else "Mixed"
    )
    score_color = "#00d09c" if avg_score >= 75 else "#5eb8ff" if avg_score >= 60 else "#f0b429"

    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat-tile">
                <div class="label">Ideas found</div>
                <div class="value">{len(ideas)}</div>
                <div class="sub">Across {ideas['sector'].nunique()} sectors</div>
            </div>
            <div class="stat-tile">
                <div class="label">Avg quality score</div>
                <div class="value" style="color:{score_color};">{avg_score:.0f}<span style="font-size:0.95rem; color:#7a8294;">/100</span></div>
                <div class="sub"><b style="color:{score_color};">{score_label}</b></div>
            </div>
            <div class="stat-tile">
                <div class="label">Best idea</div>
                <div class="value" style="font-size:1.15rem;">{top_symbol}</div>
                <div class="sub">{top_setup}</div>
            </div>
            <div class="stat-tile">
                <div class="label">Hottest sector</div>
                <div class="value" style="font-size:1.05rem;">{top_sector}</div>
                <div class="sub"><b>{top_sector_count} ideas</b> · Avg R:R {avg_rr:.1f}x</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_glossary_chips():
    st.markdown(
        """
        <div class="glossary-row">
            <div class="gl-chip"><b>Entry</b> — Price level to buy at</div>
            <div class="gl-chip"><b>Stop</b> — Price level to cut loss if wrong</div>
            <div class="gl-chip"><b>T1 / T2</b> — Profit targets to scale out</div>
            <div class="gl-chip"><b>R:R</b> — Reward-to-risk. 2x means you'd make ₹2 for every ₹1 risked</div>
            <div class="gl-chip"><b>Score</b> — How many strength checks the stock passed (out of 100)</div>
            <div class="gl-chip"><b>Horizon</b> — Typical time the move takes to play out</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_setup_legend(ideas: pd.DataFrame):
    counts = ideas["setup"].value_counts().to_dict()
    chips = []
    for setup, meta in _SETUP_META.items():
        n = counts.get(setup, 0)
        chips.append(
            f"<div class='setup-badge' style='color:{meta['color']}; border-color:{meta['color']}55;'>"
            f"{meta['icon']} <b>{setup}</b> · {n} ideas"
            f"</div>"
        )
    st.markdown(f"<div class='setup-legend'>{''.join(chips)}</div>", unsafe_allow_html=True)


def _render_cards(ideas: pd.DataFrame):
    for _, row in ideas.iterrows():
        setup = str(row["setup"])
        meta = _SETUP_META.get(setup, {"color": "#7a8294", "icon": "·", "tagline": ""})
        color = meta["color"]
        score = float(row["score"])
        score_pct = max(0.0, min(100.0, score))
        score_color = (
            "#00d09c" if score >= 75 else
            "#5eb8ff" if score >= 60 else
            "#f0b429" if score >= 50 else "#eb5757"
        )

        close = float(row["close"])
        entry = float(row["entry"])
        stop = float(row["stop"])
        target_1 = float(row["target_1"])
        target_2 = float(row["target_2"])
        rr = float(row["reward_risk"])
        rr_class = "good" if rr >= 2.0 else "ok" if rr >= 1.3 else ""

        # Distance helpers
        upside_t1 = (target_1 / close - 1) * 100 if close else 0
        upside_t2 = (target_2 / close - 1) * 100 if close else 0
        downside_stop = (stop / close - 1) * 100 if close else 0
        entry_diff = (entry / close - 1) * 100 if close else 0

        st.markdown(
            f"""
            <div class="swing-card">
              <div class="accent-strip" style="background:{color};"></div>
              <div class="head">
                <div style="flex:1;">
                  <div>
                    <span class="symbol">{row['symbol']}</span>
                    <span class="setup-pill" style="color:{color}; background:{color}1f; border:1px solid {color}55;">
                        {meta['icon']} {setup}
                    </span>
                    <span class="horizon-pill">{row['horizon']}</span>
                  </div>
                  <div class="company">{row['company']} · {row['sector']}</div>
                </div>
                <div class="score-wrap">
                  <div class="score-num" style="color:{score_color};">{score:.0f}<span style="font-size:0.8rem; color:#7a8294;">/100</span></div>
                  <div class="score-label">Quality</div>
                  <div class="score-bar"><div style="width:{score_pct:.0f}%; background:{score_color};"></div></div>
                </div>
              </div>

              <div class="ladder">
                <div class="lvl lvl-close">
                  <div class="lvl-label">Last close</div>
                  <div class="lvl-value">₹{close:,.2f}</div>
                  <div class="lvl-help">Current market price</div>
                </div>
                <div class="lvl lvl-entry">
                  <div class="lvl-label"><span style="color:#00d09c;">Entry</span></div>
                  <div class="lvl-value">₹{entry:,.2f}</div>
                  <div class="lvl-help">{entry_diff:+.2f}% vs close</div>
                </div>
                <div class="lvl lvl-stop">
                  <div class="lvl-label"><span style="color:#eb5757;">Stop</span></div>
                  <div class="lvl-value">₹{stop:,.2f}</div>
                  <div class="lvl-help">Risk {downside_stop:+.2f}%</div>
                </div>
                <div class="lvl lvl-t1">
                  <div class="lvl-label"><span style="color:#5eb8ff;">Target 1</span></div>
                  <div class="lvl-value">₹{target_1:,.2f}</div>
                  <div class="lvl-help">Upside {upside_t1:+.2f}%</div>
                </div>
                <div class="lvl lvl-t2">
                  <div class="lvl-label"><span style="color:#5eb8ff;">Target 2</span></div>
                  <div class="lvl-value">₹{target_2:,.2f}</div>
                  <div class="lvl-help">Upside {upside_t2:+.2f}%</div>
                </div>
              </div>

              <div style="margin-top:6px;">
                <span class="rr-pill {rr_class}">Reward : Risk = {rr:.1f}x</span>
              </div>

              <div class="why"><b>Why this trade:</b> {row['rationale']}</div>
              <div class="invalid"><b>Invalidates if:</b> {row['invalidation']}</div>
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
    ]].rename(columns={
        "symbol": "Symbol",
        "company": "Company",
        "sector": "Sector",
        "setup": "Setup",
        "score": "Score",
        "close": "Close",
        "entry": "Entry",
        "stop": "Stop",
        "target_1": "Target 1",
        "target_2": "Target 2",
        "reward_risk": "R:R",
        "horizon": "Horizon",
        "rationale": "Why",
        "invalidation": "Invalidates if",
    }).copy()

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score",
                help="Setup quality, 0-100",
                min_value=0,
                max_value=100,
                format="%.0f",
            ),
            "Close": st.column_config.NumberColumn(format="₹%.2f"),
            "Entry": st.column_config.NumberColumn(format="₹%.2f"),
            "Stop": st.column_config.NumberColumn(format="₹%.2f"),
            "Target 1": st.column_config.NumberColumn(format="₹%.2f"),
            "Target 2": st.column_config.NumberColumn(format="₹%.2f"),
            "R:R": st.column_config.NumberColumn(format="%.2fx"),
        },
    )


def _render_rulebook():
    st.markdown(
        "<div style='color:#9aa3b2; font-size:0.86rem; margin:6px 0 12px 0;'>"
        "Three repeatable setups we scan for, in plain English.</div>",
        unsafe_allow_html=True,
    )
    items = [
        {
            "setup": "Momentum breakout",
            "color": "#00d09c",
            "icon": "▲",
            "what": "Stock closes above its 20-day high while trend, volume, RSI, and relative strength all agree.",
            "buy": "On the day of breakout or on a small retest of the breakout level.",
            "stop": "Below recent swing low or 1.5× ATR — whichever is tighter.",
            "horizon": "2 to 4 weeks",
        },
        {
            "setup": "Trend pullback",
            "color": "#5eb8ff",
            "icon": "↗",
            "what": "Stock is in an uptrend but has paused and pulled back near its 20-day EMA with a healthy RSI.",
            "buy": "When price stabilises near EMA-20 and starts to turn up.",
            "stop": "Below the recent low or EMA-50, whichever is more conservative.",
            "horizon": "1 to 3 weeks",
        },
        {
            "setup": "Mean-reversion bounce",
            "color": "#f0b429",
            "icon": "⤴",
            "what": "Stock is oversold near its 20-day low — but the long-term trend isn't broken.",
            "buy": "On signs of intraday reversal or a higher-low forming.",
            "stop": "About 1.2× ATR below current close.",
            "horizon": "3 to 10 sessions",
        },
    ]
    for it in items:
        st.markdown(
            f"""
            <div style="background:#11151d; border:1px solid #232834; border-left:3px solid {it['color']};
                        border-radius:10px; padding:14px 16px; margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <div style="color:{it['color']}; font-weight:700; font-size:0.95rem;">
                        {it['icon']} {it['setup']}
                    </div>
                    <div style="color:#7a8294; font-size:0.75rem;">Horizon: <b style="color:#cfd6e1;">{it['horizon']}</b></div>
                </div>
                <div style="color:#cfd6e1; font-size:0.82rem; line-height:1.55; margin-bottom:6px;">
                    <b>What it is:</b> {it['what']}
                </div>
                <div style="color:#9aa3b2; font-size:0.78rem; line-height:1.5;">
                    <b style="color:#cfd6e1;">When to buy:</b> {it['buy']} &nbsp;·&nbsp;
                    <b style="color:#eb5757;">Stop:</b> {it['stop']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
