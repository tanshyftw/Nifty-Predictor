"""FII/DII quarterly ownership concentration dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data.shareholding_tracker import (
    INCOMING_DIR,
    OWNERSHIP_STORE,
    compute_ownership_deltas,
    ensure_shareholding_store,
    ingest_quarterly_file,
    ingest_quarterly_records,
    latest_ownership_snapshot,
    load_ownership_history,
    sector_ownership_summary,
)
from data.sources.nse_shareholding_fetcher import NSEShareholdingFetcher
from dashboard.universe import get_universe_symbols


_CSS = """
<style>
/* Hero explainer card */
.own-hero {
    background: linear-gradient(135deg, #11151d 0%, #131826 60%, #0f1a25 100%);
    border: 1px solid #232834;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 18px;
    display: flex;
    gap: 18px;
    align-items: flex-start;
}
.own-hero .icon {
    width: 44px; height: 44px;
    border-radius: 12px;
    background: rgba(0,208,156,0.10);
    border: 1px solid rgba(0,208,156,0.25);
    display:flex; align-items:center; justify-content:center;
    color:#00d09c; font-weight:700; font-size:1.05rem;
    flex-shrink:0;
}
.own-hero .title { color:#e8ecf1; font-size:1.0rem; font-weight:600; margin-bottom:4px; }
.own-hero .body  { color:#9aa3b2; font-size:0.84rem; line-height:1.55; }
.own-hero .body b { color:#cfd6e1; }
.own-hero .body .pill {
    display:inline-block; padding:1px 8px; border-radius:999px;
    background:#0d1119; border:1px solid #232834;
    color:#cfd6e1; font-size:0.72rem; margin:0 4px;
}

/* Headline stat tiles */
.stat-row { display:flex; gap:12px; margin-bottom:14px; flex-wrap:wrap; }
.stat-tile {
    flex: 1 1 220px;
    background:#11151d;
    border:1px solid #232834;
    border-radius:12px;
    padding:14px 16px;
    position: relative;
    overflow: hidden;
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
    font-size:1.45rem;
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
.stat-tile .accent-fii { color:#00d09c; }
.stat-tile .accent-dii { color:#5eb8ff; }
.stat-tile .breadth-bar {
    margin-top:10px;
    height:6px;
    background:#1a1f2b;
    border-radius:4px;
    overflow:hidden;
}
.stat-tile .breadth-fill {
    height:100%;
    border-radius:4px;
    transition: width 0.3s ease;
}

/* Glossary chips row */
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
.gl-chip .dot-fii { display:inline-block; width:8px; height:8px; border-radius:50%; background:#00d09c; margin-right:6px; vertical-align:middle; }
.gl-chip .dot-dii { display:inline-block; width:8px; height:8px; border-radius:50%; background:#5eb8ff; margin-right:6px; vertical-align:middle; }

/* Lens / control strip */
.lens-strip {
    background:#11151d;
    border:1px solid #232834;
    border-radius:12px;
    padding:12px 14px;
    margin-bottom:14px;
}

/* Quadrant legend chips (sector lens) */
.quad-legend { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; margin-bottom:10px; }
.quad-chip {
    padding:5px 10px;
    border-radius:8px;
    font-size:0.74rem;
    border:1px solid #232834;
    background:#11151d;
    color:#cfd6e1;
}
.quad-chip .sw { display:inline-block; width:8px; height:8px; border-radius:2px; margin-right:6px; vertical-align:middle; }
</style>
"""


def render_ownership_flows():
    """Render quarterly institutional ownership analytics."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.title("Ownership Radar")
    st.caption("See where the big money is moving — quarter after quarter.")

    ensure_shareholding_store()

    history = _load_history_cached(_store_mtime())
    if history.empty:
        _render_hero_intro(empty=True)
        st.info(
            "No quarterly ownership data is loaded yet. Use **Sync from NSE** below "
            "or import a CSV/XLSX with quarter, symbol, fii_pct, and dii_pct columns."
        )
        _render_importer()
        _render_schema_template()
        return

    deltas = compute_ownership_deltas(history)
    latest = latest_ownership_snapshot(history)
    if latest.empty:
        st.info("Ownership store exists, but no valid rows were found.")
        return

    quarters = sorted(deltas["quarter"].dropna().unique().tolist())
    latest_quarter = quarters[-1] if quarters else "Unknown"
    previous_quarter = quarters[-2] if len(quarters) >= 2 else "None"

    _render_hero_intro(
        latest_quarter=latest_quarter,
        previous_quarter=previous_quarter,
    )

    fii_buyers = int((latest["fii_delta_pp"].fillna(0) > 0).sum())
    dii_buyers = int((latest["dii_delta_pp"].fillna(0) > 0).sum())
    total_stocks = max(len(latest), 1)
    avg_fii = latest["fii_delta_pp"].mean(skipna=True)
    avg_dii = latest["dii_delta_pp"].mean(skipna=True)

    _render_headline_tiles(
        latest_quarter=latest_quarter,
        previous_quarter=previous_quarter,
        fii_buyers=fii_buyers,
        dii_buyers=dii_buyers,
        total=total_stocks,
        avg_fii=avg_fii if pd.notna(avg_fii) else 0.0,
        avg_dii=avg_dii if pd.notna(avg_dii) else 0.0,
        history_rows=len(history),
        quarters_count=len(quarters),
    )

    _render_glossary_strip()

    # Lens controls
    st.markdown('<div class="lens-strip">', unsafe_allow_html=True)
    lc1, lc2 = st.columns([1, 2])
    with lc1:
        investor = st.radio(
            "Whose moves do you want to see?",
            ["Foreign (FII)", "Domestic (DII)"],
            horizontal=True,
            key="ownership_investor_lens",
            help="FII = Foreign Institutional Investors (e.g., overseas funds). "
                 "DII = Domestic Institutional Investors (Indian mutual funds, insurers).",
        )
        investor_key = "fii" if investor.startswith("Foreign") else "dii"
    with lc2:
        all_sectors = sorted(latest["sector"].dropna().unique().tolist())
        sector_filter = st.multiselect(
            "Filter sectors (leave all selected to see the full market)",
            all_sectors,
            default=all_sectors,
            key="ownership_sector_filter",
        )
    st.markdown('</div>', unsafe_allow_html=True)

    if sector_filter:
        latest_view = latest[latest["sector"].isin(sector_filter)].copy()
    else:
        latest_view = latest.copy()

    with st.expander("Sync or import quarterly data", expanded=False):
        _render_importer_inner()

    tab_stocks, tab_sectors, tab_trends = st.tabs([
        "Top movers",
        "Sector map",
        "Stock history",
    ])

    with tab_stocks:
        _render_stock_concentration(latest_view, investor_key)
    with tab_sectors:
        _render_sector_lens(history, latest_quarter)
    with tab_trends:
        _render_ticker_trail(deltas, latest_view)


def _render_hero_intro(latest_quarter: str | None = None, previous_quarter: str | None = None, empty: bool = False):
    if empty:
        body = (
            "Every quarter, big funds disclose what they bought and sold. "
            "This radar surfaces the stocks they crowded into — and the ones they walked away from."
        )
    else:
        body = (
            "Every quarter, big funds disclose what they bought and sold. "
            f"You're looking at <b>{latest_quarter}</b> versus <b>{previous_quarter}</b>. "
            "Green moves = funds added to their position. Red moves = funds trimmed it. "
            "The bigger the number, the stronger the conviction."
        )
    st.markdown(
        f"""
        <div class="own-hero">
            <div class="icon">●</div>
            <div>
                <div class="title">What is the Ownership Radar?</div>
                <div class="body">{body}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_headline_tiles(
    *,
    latest_quarter: str,
    previous_quarter: str,
    fii_buyers: int,
    dii_buyers: int,
    total: int,
    avg_fii: float,
    avg_dii: float,
    history_rows: int,
    quarters_count: int,
):
    fii_pct = (fii_buyers / total) * 100 if total else 0
    dii_pct = (dii_buyers / total) * 100 if total else 0

    fii_label = "Strong buying" if fii_pct >= 60 else "Mixed" if fii_pct >= 40 else "Net selling"
    dii_label = "Strong buying" if dii_pct >= 60 else "Mixed" if dii_pct >= 40 else "Net selling"

    fii_color = "#00d09c" if fii_pct >= 60 else "#f0b429" if fii_pct >= 40 else "#eb5757"
    dii_color = "#00d09c" if dii_pct >= 60 else "#f0b429" if dii_pct >= 40 else "#eb5757"

    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat-tile">
                <div class="label">Latest filing</div>
                <div class="value">{latest_quarter}</div>
                <div class="sub">Compared with <b>{previous_quarter}</b></div>
            </div>
            <div class="stat-tile">
                <div class="label"><span class="accent-fii">●</span> Foreign funds (FII)</div>
                <div class="value">{fii_buyers}<span style="font-size:1.0rem; color:#7a8294;"> / {total} stocks</span></div>
                <div class="sub"><b style="color:{fii_color};">{fii_label}</b> · avg move {avg_fii:+.2f} pp</div>
                <div class="breadth-bar"><div class="breadth-fill" style="width:{fii_pct:.0f}%; background:{fii_color};"></div></div>
            </div>
            <div class="stat-tile">
                <div class="label"><span class="accent-dii">●</span> Domestic funds (DII)</div>
                <div class="value">{dii_buyers}<span style="font-size:1.0rem; color:#7a8294;"> / {total} stocks</span></div>
                <div class="sub"><b style="color:{dii_color};">{dii_label}</b> · avg move {avg_dii:+.2f} pp</div>
                <div class="breadth-bar"><div class="breadth-fill" style="width:{dii_pct:.0f}%; background:{dii_color};"></div></div>
            </div>
            <div class="stat-tile">
                <div class="label">Coverage</div>
                <div class="value">{history_rows:,}<span style="font-size:1.0rem; color:#7a8294;"> rows</span></div>
                <div class="sub">{quarters_count} quarters of history</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_glossary_strip():
    st.markdown(
        """
        <div class="glossary-row">
            <div class="gl-chip"><span class="dot-fii"></span><b>FII</b> — Foreign funds (overseas institutional money)</div>
            <div class="gl-chip"><span class="dot-dii"></span><b>DII</b> — Domestic funds (Indian mutual funds & insurers)</div>
            <div class="gl-chip"><b>pp</b> — Percentage points. If FII goes from 19% → 21%, that's <b>+2 pp</b></div>
            <div class="gl-chip"><b>Relative increase</b> — How much the holding grew vs the previous quarter</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=300, show_spinner=False)
def _load_history_cached(_mtime: float) -> pd.DataFrame:
    del _mtime
    return load_ownership_history()


def _store_mtime() -> float:
    try:
        return OWNERSHIP_STORE.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _render_importer():
    with st.expander("Sync or import quarterly data", expanded=False):
        _render_importer_inner()


def _render_importer_inner():
    st.markdown("**Auto-sync from NSE**")
    c0, c1, c2 = st.columns([2.5, 1, 1])
    with c0:
        raw_symbols = st.text_input(
            "Symbols",
            placeholder="RELIANCE,TCS,HDFCBANK or blank for full tracked universe",
            key="ownership_nse_symbols",
        )
    with c1:
        quarters = st.number_input("Quarters", min_value=1, max_value=12, value=4, step=1, key="ownership_nse_quarters")
    with c2:
        delay = st.number_input("Delay/sec", min_value=0.1, max_value=2.0, value=0.35, step=0.05, key="ownership_nse_delay")

    if st.button("Sync from NSE", type="primary", key="ownership_nse_sync"):
        symbols = (
            [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]
            if raw_symbols.strip()
            else get_universe_symbols()
        )
        with st.spinner(f"Fetching NSE shareholding filings for {len(symbols)} symbols..."):
            try:
                fetcher = NSEShareholdingFetcher(request_delay=float(delay))
                frame, stats = fetcher.fetch_symbols(symbols, quarters=int(quarters))
                result = ingest_quarterly_records(frame, source="NSE corporate shareholding API")
                st.cache_data.clear()
                st.success(
                    f"Fetched {stats.records_fetched} rows for "
                    f"{stats.symbols_succeeded}/{stats.symbols_requested} symbols; "
                    f"imported {result.rows_ingested} rows."
                )
                if stats.failures:
                    with st.expander("NSE fetch misses", expanded=False):
                        st.dataframe(
                            pd.DataFrame([
                                {"symbol": k, "reason": v}
                                for k, v in stats.failures.items()
                            ]),
                            hide_index=True,
                            use_container_width=True,
                        )
            except Exception as exc:
                st.error(f"NSE sync failed: {exc}")

    st.divider()
    st.markdown("**Manual fallback**")
    c1, c2 = st.columns([2, 1])
    with c1:
        upload = st.file_uploader(
            "Quarterly shareholding file",
            type=["csv", "xlsx", "xls"],
            key="ownership_upload",
        )
    with c2:
        quarter = st.text_input("Quarter override", placeholder="2026Q1", key="ownership_quarter")

    if upload is not None and st.button("Import file", type="primary", key="ownership_import_btn"):
        INCOMING_DIR.mkdir(parents=True, exist_ok=True)
        dest = INCOMING_DIR / upload.name
        dest.write_bytes(upload.getbuffer())
        try:
            result = ingest_quarterly_file(dest, quarter=quarter or None)
            st.cache_data.clear()
            st.success(
                f"Imported {result.rows_ingested} rows for {', '.join(result.quarters)}."
            )
        except Exception as exc:
            st.error(f"Import failed: {exc}")


def _render_stock_concentration(df: pd.DataFrame, investor: str):
    if df.empty:
        st.info("No stocks match the current filters. Try selecting more sectors above.")
        return
    investor_name = "Foreign funds (FII)" if investor == "fii" else "Domestic funds (DII)"
    accent = "#00d09c" if investor == "fii" else "#5eb8ff"

    delta_col = f"{investor}_delta_pp"
    pct_col = f"{investor}_pct"
    rel_col = f"{investor}_increase_pct"

    sorted_df = df.sort_values(delta_col, ascending=False)
    buyers = sorted_df.head(12).copy()
    sellers = sorted_df.tail(12).sort_values(delta_col).copy()

    st.markdown(
        f"<div style='color:#9aa3b2; font-size:0.86rem; margin:6px 0 12px 0;'>"
        f"Top stocks <b style='color:#e8ecf1;'>{investor_name}</b> bought and sold last quarter. "
        f"Each bar is the change in their ownership stake.</div>",
        unsafe_allow_html=True,
    )

    col_buy, col_sell = st.columns(2)
    with col_buy:
        st.markdown(
            "<div style='color:#00d09c; font-size:0.86rem; font-weight:600; margin-bottom:6px;'>"
            "▲ Top accumulations (funds adding)</div>",
            unsafe_allow_html=True,
        )
        _render_movers_bar(buyers, delta_col, pct_col, rel_col, color="#00d09c", positive=True)

    with col_sell:
        st.markdown(
            "<div style='color:#eb5757; font-size:0.86rem; font-weight:600; margin-bottom:6px;'>"
            "▼ Top reductions (funds trimming)</div>",
            unsafe_allow_html=True,
        )
        _render_movers_bar(sellers, delta_col, pct_col, rel_col, color="#eb5757", positive=False)

    st.markdown(
        f"<div style='color:#9aa3b2; font-size:0.82rem; margin:14px 0 6px 0;'>"
        f"Full breakdown — sorted by largest QoQ ownership change</div>",
        unsafe_allow_html=True,
    )

    table = sorted_df[[
        "quarter",
        "symbol",
        "company",
        "sector",
        pct_col,
        f"prev_{investor}_pct",
        delta_col,
        rel_col,
        "ownership_signal",
    ]].rename(columns={
        "quarter": "Quarter",
        "symbol": "Symbol",
        "company": "Company",
        "sector": "Sector",
        pct_col: f"{investor.upper()} %",
        f"prev_{investor}_pct": "Previous %",
        delta_col: "Δ (pp)",
        rel_col: "Rel. increase %",
        "ownership_signal": "Signal",
    })
    st.dataframe(table, use_container_width=True, hide_index=True)


def _render_movers_bar(df: pd.DataFrame, delta_col: str, pct_col: str, rel_col: str, color: str, positive: bool):
    if df.empty:
        st.info("No data.")
        return
    fig = go.Figure(go.Bar(
        x=df[delta_col],
        y=df["symbol"],
        orientation="h",
        marker=dict(
            color=color,
            line=dict(color=color, width=0),
        ),
        opacity=0.92,
        customdata=df[["company", "sector", pct_col, rel_col]].fillna("").values,
        hovertemplate=(
            "<b>%{y}</b> · %{customdata[0]}<br>"
            "<span style='color:#9aa3b2;'>%{customdata[1]}</span><br>"
            "Now: %{customdata[2]:.2f}%<br>"
            "Change: %{x:+.2f} pp (%{customdata[3]:+.1f}% relative)<extra></extra>"
        ),
        text=[f"{v:+.2f} pp" for v in df[delta_col].fillna(0)],
        textposition="outside",
        textfont=dict(color="#c9cfd9", size=11),
    ))
    height = max(360, 28 * len(df))
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=40, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(
            title=None,
            showgrid=True,
            gridcolor="#1a1f2b",
            zerolinecolor="#2f3645",
            tickfont=dict(size=10, color="#7a8294"),
        ),
        yaxis=dict(
            title=None,
            autorange="reversed",
            tickfont=dict(size=11, color="#c9cfd9"),
            showgrid=False,
        ),
        bargap=0.32,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "displayModeBar": False})


def _render_sector_lens(history: pd.DataFrame, latest_quarter: str):
    summary = sector_ownership_summary(history, quarter=latest_quarter)
    if summary.empty:
        st.info("Sector summary unavailable.")
        return

    st.markdown(
        "<div style='color:#9aa3b2; font-size:0.86rem; margin:6px 0 8px 0;'>"
        "Each bubble is a sector. Where it sits tells you the story — "
        "<b style='color:#e8ecf1;'>top-right</b> means both foreign and domestic funds are buying. "
        "<b style='color:#e8ecf1;'>Bottom-left</b> means both are selling.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="quad-legend">
            <div class="quad-chip"><span class="sw" style="background:#00d09c;"></span><b>Top-right</b> — Both buying (high conviction)</div>
            <div class="quad-chip"><span class="sw" style="background:#5eb8ff;"></span><b>Top-left</b> — DII buying, FII selling (domestic rotation)</div>
            <div class="quad-chip"><span class="sw" style="background:#f0b429;"></span><b>Bottom-right</b> — FII buying, DII selling (foreign rotation)</div>
            <div class="quad-chip"><span class="sw" style="background:#eb5757;"></span><b>Bottom-left</b> — Both selling (distribution)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = px.scatter(
        summary,
        x="avg_fii_delta_pp",
        y="avg_dii_delta_pp",
        size="stock_count",
        color="sector",
        text="sector",
        hover_data={
            "stock_count": True,
            "avg_fii_pct": ":.2f",
            "avg_dii_pct": ":.2f",
            "fii_accumulation_count": True,
            "dii_accumulation_count": True,
            "sector": False,
        },
        labels={
            "avg_fii_delta_pp": "Average FII change (pp) →",
            "avg_dii_delta_pp": "Average DII change (pp) →",
        },
        size_max=46,
    )
    fig.update_traces(
        textposition="top center",
        textfont=dict(size=10, color="#cfd6e1"),
        marker=dict(line=dict(width=0)),
        opacity=0.85,
    )
    fig.add_hline(y=0, line_color="#2f3645", line_width=1, line_dash="dot")
    fig.add_vline(x=0, line_color="#2f3645", line_width=1, line_dash="dot")

    # Quadrant background shading
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=0.5, x1=1, y0=0.5, y1=1,
                  fillcolor="rgba(0,208,156,0.04)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, x1=0.5, y0=0.5, y1=1,
                  fillcolor="rgba(94,184,255,0.04)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=0.5, x1=1, y0=0, y1=0.5,
                  fillcolor="rgba(240,180,41,0.04)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, x1=0.5, y0=0, y1=0.5,
                  fillcolor="rgba(235,87,87,0.04)", line=dict(width=0), layer="below")

    fig.update_layout(
        template="plotly_dark",
        height=540,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=20),
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="#1a1f2b", zerolinecolor="#2f3645"),
        yaxis=dict(showgrid=True, gridcolor="#1a1f2b", zerolinecolor="#2f3645"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "displayModeBar": False})

    pretty = summary.rename(columns={
        "sector": "Sector",
        "stock_count": "Stocks",
        "avg_fii_pct": "Avg FII %",
        "avg_dii_pct": "Avg DII %",
        "avg_fii_delta_pp": "Avg FII Δ (pp)",
        "avg_dii_delta_pp": "Avg DII Δ (pp)",
        "fii_accumulation_count": "FII buying",
        "dii_accumulation_count": "DII buying",
    })
    st.dataframe(pretty, use_container_width=True, hide_index=True)


def _render_ticker_trail(deltas: pd.DataFrame, latest: pd.DataFrame):
    if latest.empty:
        st.info("No ticker data available.")
        return

    st.markdown(
        "<div style='color:#9aa3b2; font-size:0.86rem; margin:6px 0 12px 0;'>"
        "Pick a stock to see how foreign and domestic fund ownership has evolved over time.</div>",
        unsafe_allow_html=True,
    )

    default_symbol = str(latest.sort_values("fii_delta_pp", ascending=False).iloc[0]["symbol"])
    options = sorted(latest["symbol"].dropna().unique().tolist())
    symbol = st.selectbox(
        "Stock",
        options,
        index=options.index(default_symbol) if default_symbol in options else 0,
        key="ownership_ticker_trail",
    )
    trail = deltas[deltas["symbol"] == symbol].sort_values("quarter").copy()
    if trail.empty:
        st.info("No history for this ticker.")
        return

    company = ""
    sector = ""
    row = latest[latest["symbol"] == symbol]
    if not row.empty:
        company = str(row.iloc[0].get("company") or "")
        sector = str(row.iloc[0].get("sector") or "")

    latest_fii = float(trail.iloc[-1].get("fii_pct") or 0)
    latest_dii = float(trail.iloc[-1].get("dii_pct") or 0)
    delta_fii = float(trail.iloc[-1].get("fii_delta_pp") or 0)
    delta_dii = float(trail.iloc[-1].get("dii_delta_pp") or 0)

    st.markdown(
        f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin-bottom:12px;">
            <div class="stat-tile" style="flex:1 1 200px;">
                <div class="label">Company</div>
                <div class="value" style="font-size:1.05rem;">{symbol}</div>
                <div class="sub">{company} · {sector}</div>
            </div>
            <div class="stat-tile" style="flex:1 1 200px;">
                <div class="label"><span class="accent-fii">●</span> FII ownership (latest)</div>
                <div class="value accent-fii">{latest_fii:.2f}%</div>
                <div class="sub">Q-o-Q change <b style="color:{'#00d09c' if delta_fii >= 0 else '#eb5757'};">{delta_fii:+.2f} pp</b></div>
            </div>
            <div class="stat-tile" style="flex:1 1 200px;">
                <div class="label"><span class="accent-dii">●</span> DII ownership (latest)</div>
                <div class="value accent-dii">{latest_dii:.2f}%</div>
                <div class="sub">Q-o-Q change <b style="color:{'#00d09c' if delta_dii >= 0 else '#eb5757'};">{delta_dii:+.2f} pp</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trail["quarter"],
        y=trail["fii_pct"],
        mode="lines+markers",
        name="FII %",
        line=dict(color="#00d09c", width=3, shape="spline", smoothing=0.6),
        marker=dict(size=7, color="#00d09c", line=dict(color="#0b0e14", width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(0,208,156,0.10)",
        hovertemplate="<b>%{x}</b><br>FII: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=trail["quarter"],
        y=trail["dii_pct"],
        mode="lines+markers",
        name="DII %",
        line=dict(color="#5eb8ff", width=3, shape="spline", smoothing=0.6),
        marker=dict(size=7, color="#5eb8ff", line=dict(color="#0b0e14", width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(94,184,255,0.10)",
        hovertemplate="<b>%{x}</b><br>DII: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=20),
        yaxis=dict(
            title="Ownership %",
            showgrid=True,
            gridcolor="#1a1f2b",
            zerolinecolor="#2f3645",
            tickfont=dict(color="#7a8294"),
        ),
        xaxis=dict(
            title="Quarter",
            showgrid=False,
            tickfont=dict(color="#7a8294"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "displayModeBar": False})

    pretty = trail.rename(columns={
        "quarter": "Quarter",
        "symbol": "Symbol",
        "company": "Company",
        "sector": "Sector",
        "fii_pct": "FII %",
        "dii_pct": "DII %",
        "prev_fii_pct": "Prev FII %",
        "prev_dii_pct": "Prev DII %",
        "fii_delta_pp": "FII Δ (pp)",
        "dii_delta_pp": "DII Δ (pp)",
        "fii_increase_pct": "FII rel. %",
        "dii_increase_pct": "DII rel. %",
        "institutional_delta_pp": "Total Δ (pp)",
        "ownership_signal": "Signal",
    })
    st.dataframe(pretty, use_container_width=True, hide_index=True)


def _render_schema_template():
    template = pd.DataFrame([
        {
            "quarter": "2026Q1",
            "symbol": "RELIANCE",
            "fii_pct": 19.25,
            "dii_pct": 16.80,
            "source": "NSE shareholding pattern",
        }
    ])
    st.dataframe(template, use_container_width=True, hide_index=True)
