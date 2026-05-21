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
    latest_ownership_snapshot,
    load_ownership_history,
    sector_ownership_summary,
)


_CSS = """
<style>
.ownership-note {
    background:#0f131c;
    border:1px solid #232834;
    border-radius:10px;
    padding:12px 14px;
    color:#9aa3b2;
    font-size:0.82rem;
    line-height:1.55;
    margin-bottom:14px;
}
.ownership-note b { color:#e8ecf1; }
.flow-chip {
    display:inline-block;
    padding:2px 8px;
    border-radius:999px;
    background:#151922;
    border:1px solid #232834;
    color:#c9cfd9;
    font-size:0.72rem;
    margin-right:6px;
}
</style>
"""


def render_ownership_flows():
    """Render quarterly institutional ownership analytics."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.title("Institutional Ownership Radar")
    st.caption("Quarterly FII/DII ownership concentration across the tracked stock universe.")

    ensure_shareholding_store()
    _render_importer()

    history = _load_history_cached(_store_mtime())
    if history.empty:
        st.info(
            "No quarterly ownership data is loaded yet. Import a CSV/XLSX with "
            "quarter, symbol, fii_pct, and dii_pct columns to activate the radar."
        )
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

    fii_buyers = int((latest["fii_delta_pp"].fillna(0) > 0).sum())
    dii_buyers = int((latest["dii_delta_pp"].fillna(0) > 0).sum())
    avg_fii = latest["fii_delta_pp"].mean(skipna=True)
    avg_dii = latest["dii_delta_pp"].mean(skipna=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest quarter", latest_quarter, f"vs {previous_quarter}")
    m2.metric("FII accumulation breadth", f"{fii_buyers}/{len(latest)}", f"{avg_fii:+.2f} pp avg")
    m3.metric("DII accumulation breadth", f"{dii_buyers}/{len(latest)}", f"{avg_dii:+.2f} pp avg")
    m4.metric("Tracked rows", f"{len(history):,}", f"{len(quarters)} quarters")

    st.markdown(
        "<span class='flow-chip'>pp = percentage-point ownership change QoQ</span>"
        "<span class='flow-chip'>relative increase = delta / previous ownership</span>",
        unsafe_allow_html=True,
    )

    investor = st.radio(
        "Investor lens",
        ["FII", "DII"],
        horizontal=True,
        key="ownership_investor_lens",
    ).lower()

    sector_filter = st.multiselect(
        "Sectors",
        sorted(latest["sector"].dropna().unique().tolist()),
        default=sorted(latest["sector"].dropna().unique().tolist()),
        key="ownership_sector_filter",
    )
    if sector_filter:
        latest_view = latest[latest["sector"].isin(sector_filter)].copy()
    else:
        latest_view = latest.copy()

    tab_stocks, tab_sectors, tab_trends = st.tabs(["Stock concentration", "Sector lens", "Ticker trail"])

    with tab_stocks:
        _render_stock_concentration(latest_view, investor)
    with tab_sectors:
        _render_sector_lens(history, latest_quarter)
    with tab_trends:
        _render_ticker_trail(deltas, latest_view)


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
    with st.expander("Import quarterly data", expanded=False):
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
        st.info("No stocks match the current filters.")
        return
    delta_col = f"{investor}_delta_pp"
    pct_col = f"{investor}_pct"
    rel_col = f"{investor}_increase_pct"

    leaders = df.sort_values(delta_col, ascending=False).head(25)
    colors = ["#00d09c" if v >= 0 else "#eb5757" for v in leaders[delta_col].fillna(0)]
    fig = go.Figure(go.Bar(
        x=leaders[delta_col],
        y=leaders["symbol"],
        orientation="h",
        marker_color=colors,
        customdata=leaders[["company", "sector", pct_col, rel_col]].fillna("").values,
        hovertemplate=(
            "<b>%{y}</b> %{customdata[0]}<br>"
            "Sector: %{customdata[1]}<br>"
            "Current ownership: %{customdata[2]:.2f}%<br>"
            "QoQ relative increase: %{customdata[3]:+.1f}%<br>"
            "QoQ delta: %{x:+.2f} pp<extra></extra>"
        ),
        text=[f"{v:+.2f} pp" for v in leaders[delta_col].fillna(0)],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=max(420, 22 * len(leaders)),
        margin=dict(l=10, r=30, t=20, b=20),
        paper_bgcolor="#0b0e14",
        plot_bgcolor="#0b0e14",
        xaxis_title=f"{investor.upper()} ownership QoQ change (pp)",
        yaxis=dict(autorange="reversed", title=""),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    table = leaders[[
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
        pct_col: f"{investor.upper()} %",
        f"prev_{investor}_pct": "Previous %",
        delta_col: "Delta pp",
        rel_col: "Relative increase %",
        "ownership_signal": "Signal",
    })
    st.dataframe(table, use_container_width=True, hide_index=True)


def _render_sector_lens(history: pd.DataFrame, latest_quarter: str):
    summary = sector_ownership_summary(history, quarter=latest_quarter)
    if summary.empty:
        st.info("Sector summary unavailable.")
        return

    fig = px.scatter(
        summary,
        x="avg_fii_delta_pp",
        y="avg_dii_delta_pp",
        size="stock_count",
        color="sector",
        hover_data=[
            "stock_count",
            "avg_fii_pct",
            "avg_dii_pct",
            "fii_accumulation_count",
            "dii_accumulation_count",
        ],
        labels={
            "avg_fii_delta_pp": "Avg FII QoQ delta (pp)",
            "avg_dii_delta_pp": "Avg DII QoQ delta (pp)",
        },
    )
    fig.add_hline(y=0, line_color="#2f3645", line_width=1)
    fig.add_vline(x=0, line_color="#2f3645", line_width=1)
    fig.update_layout(
        template="plotly_dark",
        height=520,
        paper_bgcolor="#0b0e14",
        plot_bgcolor="#0b0e14",
        margin=dict(l=10, r=10, t=20, b=20),
        legend_title_text="Sector",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.dataframe(summary, use_container_width=True, hide_index=True)


def _render_ticker_trail(deltas: pd.DataFrame, latest: pd.DataFrame):
    if latest.empty:
        st.info("No ticker data available.")
        return
    default_symbol = str(latest.sort_values("fii_delta_pp", ascending=False).iloc[0]["symbol"])
    symbol = st.selectbox(
        "Ticker",
        sorted(latest["symbol"].dropna().unique().tolist()),
        index=sorted(latest["symbol"].dropna().unique().tolist()).index(default_symbol),
        key="ownership_ticker_trail",
    )
    trail = deltas[deltas["symbol"] == symbol].sort_values("quarter").copy()
    if trail.empty:
        st.info("No history for this ticker.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trail["quarter"],
        y=trail["fii_pct"],
        mode="lines+markers",
        name="FII %",
        line=dict(color="#00d09c", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=trail["quarter"],
        y=trail["dii_pct"],
        mode="lines+markers",
        name="DII %",
        line=dict(color="#5eb8ff", width=3),
    ))
    fig.update_layout(
        template="plotly_dark",
        height=420,
        paper_bgcolor="#0b0e14",
        plot_bgcolor="#0b0e14",
        margin=dict(l=10, r=10, t=20, b=20),
        yaxis_title="Ownership %",
        xaxis_title="Quarter",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.dataframe(trail, use_container_width=True, hide_index=True)


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
