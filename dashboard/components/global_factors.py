"""Global indices and supply chain factors — value-chain visualization."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from dashboard.data_loader import MarketSnapshot
from dashboard.config import SECTOR_SUPPLY_CHAIN, SECTOR_INDEX_TO_SECTOR


def render_global_indices(snapshot: MarketSnapshot):
    """Render global index cards — price, DoD %, WoW %, and a sparkline.

    Each card is a self-contained HTML block so the grid stays tight and the
    UI keeps working even when a card is missing. Sparklines are inline SVG
    polylines drawn from the last ~10 closes of each index.
    """
    st.markdown("#### Global Indices")

    if not snapshot.global_indices:
        st.info("Global index data unavailable")
        return

    st.markdown(_INDICES_CSS, unsafe_allow_html=True)

    cards_html = []
    for name, data in snapshot.global_indices.items():
        close = data.get("close")
        dod = data.get("ret_pct")
        wow = data.get("wow_pct")
        spark = data.get("spark") or []

        if close is None or dod is None:
            continue

        cards_html.append(_index_card_html(name, close, dod, wow, spark))

    if not cards_html:
        st.info("Global index data unavailable")
        return

    st.markdown(
        f"<div class='idx-grid'>{''.join(cards_html)}</div>",
        unsafe_allow_html=True,
    )


_INDICES_CSS = """
<style>
.idx-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px;
    margin-top: 6px;
}
.idx-card {
    background: #11151d;
    border: 1px solid #1f2633;
    border-radius: 12px;
    padding: 0.85rem 1rem 0.9rem;
    transition: border-color 0.15s ease, transform 0.15s ease;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
}
.idx-card:hover { border-color: #2f3a4d; transform: translateY(-1px); }
.idx-card .name {
    color: #7a8294;
    font-size: 0.66rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
}
.idx-card .row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.4rem;
}
.idx-card .price {
    color: #e8ecf1;
    font-size: 1.15rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.01em;
}
.idx-card .dod {
    font-size: 0.82rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}
.idx-card .dod.pos { color: #00d09c; }
.idx-card .dod.neg { color: #ef5350; }
.idx-card .dod.flat { color: #8a92a0; }
.idx-card .foot {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
}
.idx-card .wow {
    color: #7a8294;
    font-size: 0.72rem;
    font-variant-numeric: tabular-nums;
}
.idx-card .wow .val.pos { color: #00d09c; }
.idx-card .wow .val.neg { color: #ef5350; }
.idx-card svg.spark {
    display: block;
    overflow: visible;
}
</style>
"""


def _index_card_html(name: str, close: float, dod: float, wow: float | None,
                     spark: list[float]) -> str:
    """Render one global-index card as a self-contained HTML block."""
    if dod > 0.02:
        dod_cls, dod_colour = "pos", "#00d09c"
    elif dod < -0.02:
        dod_cls, dod_colour = "neg", "#ef5350"
    else:
        dod_cls, dod_colour = "flat", "#8a92a0"

    spark_svg = _sparkline_svg(spark, stroke=dod_colour) if spark else ""

    wow_html = ""
    if wow is not None:
        w_cls = "pos" if wow > 0.02 else "neg" if wow < -0.02 else ""
        w_sign = "+" if wow > 0 else ""
        wow_html = (
            f"<span class='wow'>WoW "
            f"<span class='val {w_cls}'>{w_sign}{wow:.1f}%</span></span>"
        )

    dod_sign = "+" if dod > 0 else ""
    return (
        f"<div class='idx-card'>"
        f"<div class='name'>{name}</div>"
        f"<div class='row'>"
        f"<span class='price'>{close:,.2f}</span>"
        f"<span class='dod {dod_cls}'>{dod_sign}{dod:.1f}%</span>"
        f"</div>"
        f"<div class='foot'>{wow_html}{spark_svg}</div>"
        f"</div>"
    )


def _sparkline_svg(points: list[float], width: int = 72, height: int = 22,
                   stroke: str = "#00d09c") -> str:
    """Inline SVG polyline sparkline from a series of closes."""
    if len(points) < 2:
        return ""
    lo = min(points)
    hi = max(points)
    span = hi - lo or 1.0
    step = width / (len(points) - 1)
    coords = []
    for i, v in enumerate(points):
        x = i * step
        # Higher value = lower y in SVG coords; pad 2px top/bottom.
        y = height - 2 - ((v - lo) / span) * (height - 4)
        coords.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(coords)
    return (
        f"<svg class='spark' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>"
        f"<polyline fill='none' stroke='{stroke}' stroke-width='1.5' "
        f"stroke-linecap='round' stroke-linejoin='round' points='{poly}'/>"
        f"</svg>"
    )


def render_supply_chain(snapshot: MarketSnapshot):
    """Render the supply chain as a series of value-chain visualizations.

    Each row = international factor(s) → sector → top affected stock,
    with chain connectors that visually 'break' under shock conditions.
    """
    st.markdown("#### Supply Chain & International Factors")

    if snapshot.supply_chain.empty:
        st.info("Supply chain data unavailable")
        return

    df = snapshot.supply_chain.copy()
    factor_data = {
        row["Factor"]: {"price": row["Price"], "dod": row["DoD %"], "wow": row["WoW %"]}
        for _, row in df.iterrows()
    }

    # Sector → sectoral_index row lookup
    sector_to_idx = {}
    if not snapshot.sectoral_data.empty:
        for idx_name, sect_name in SECTOR_INDEX_TO_SECTOR.items():
            if sect_name in sector_to_idx:
                continue
            row = snapshot.sectoral_data[snapshot.sectoral_data["Index"] == idx_name]
            if not row.empty:
                sector_to_idx[sect_name] = row.iloc[0]

    # Top mover per sector (largest absolute move)
    sector_top_stock = {}
    if not snapshot.stock_changes.empty:
        for sect, group in snapshot.stock_changes.groupby("sector"):
            top = group.iloc[group["dod_pct"].abs().argmax()]
            sector_top_stock[sect] = {"symbol": top["symbol"], "change": float(top["dod_pct"])}

    # Factors-at-a-glance strip
    _render_factor_strip(factor_data)

    # Legend
    st.markdown(
        '<div style="font-size:0.72rem;color:#7a8294;margin:18px 0 14px 0;">'
        'Each row traces an international factor &rarr; Indian sector &rarr; top affected stock. '
        'Chain link state: '
        '<span style="color:#eb5757;font-weight:600;">&#9889; Shock</span> (factor &gt;5%) '
        '&middot; <span style="color:#f5a623;font-weight:600;">&#9888; Pressure</span> (1.5–5%) '
        '&middot; <span style="color:#00d09c;font-weight:600;">&#9679; Stable</span> (&lt;1.5%)'
        '</div>',
        unsafe_allow_html=True,
    )

    # Build chain rows, sorted by max factor severity (most stressed first)
    chains = []
    for sector, info in SECTOR_SUPPLY_CHAIN.items():
        factors = [(f, factor_data[f]) for f in info["factors"] if f in factor_data]
        if not factors:
            continue
        max_severity = max(abs(fd["dod"]) for _, fd in factors)
        chains.append({
            "sector": sector,
            "factors": factors,
            "note": info["note"],
            "severity": max_severity,
            "sector_row": sector_to_idx.get(sector),
            "top_stock": sector_top_stock.get(sector),
        })
    chains.sort(key=lambda c: c["severity"], reverse=True)

    if not chains:
        st.caption("No supply-chain factor data to map.")
        return

    for chain in chains:
        st.markdown(_render_chain_row(chain), unsafe_allow_html=True)


def _render_factor_strip(factor_data: dict):
    """Compact horizontal chips showing all 8 factors at a glance."""
    if not factor_data:
        return

    chips = []
    for name, fd in factor_data.items():
        dod = fd["dod"]
        if dod > 0:
            color, bg, border = "#00d09c", "rgba(0,208,156,0.10)", "rgba(0,208,156,0.25)"
        elif dod < 0:
            color, bg, border = "#eb5757", "rgba(235,87,87,0.10)", "rgba(235,87,87,0.25)"
        else:
            color, bg, border = "#c9cfd9", "rgba(201,207,217,0.06)", "rgba(201,207,217,0.18)"
        chips.append(
            f'<div style="background:{bg};border:1px solid {border};border-radius:8px;'
            f'padding:8px 12px;min-width:130px;flex:1 1 130px;">'
            f'<div style="font-size:0.66rem;color:#7a8294;font-weight:500;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name}</div>'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
            f'margin-top:3px;gap:8px;">'
            f'<span style="font-size:0.82rem;font-weight:600;color:#e8ecf1;">'
            f'{fd["price"]:,.2f}</span>'
            f'<span style="font-size:0.78rem;font-weight:600;color:{color};">'
            f'{dod:+.1f}%</span>'
            f'</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:6px;">'
        f'{"".join(chips)}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _stress_style(severity: float) -> dict:
    """Return color/icon/label for a chain-link based on factor severity."""
    if severity >= 5:
        return {
            "color": "#eb5757",
            "bg": "rgba(235,87,87,0.14)",
            "icon": "&#9889;",  # ⚡
            "label": "SHOCK",
            "line_style": "dashed",
        }
    if severity >= 1.5:
        return {
            "color": "#f5a623",
            "bg": "rgba(245,166,35,0.14)",
            "icon": "&#9888;",  # ⚠
            "label": "PRESSURE",
            "line_style": "dotted",
        }
    return {
        "color": "#00d09c",
        "bg": "rgba(0,208,156,0.14)",
        "icon": "&#9679;",  # ●
        "label": "STABLE",
        "line_style": "solid",
    }


def _node_html(label: str, name: str, value_str: str, value_color: str,
               border: str = "#2a3142", min_width: int = 140) -> str:
    """Render a chain node (factor / sector / stock) as a card."""
    return (
        f'<div style="background:#151922;border:1px solid {border};border-radius:10px;'
        f'padding:10px 14px;min-width:{min_width}px;flex:0 0 auto;">'
        f'<div style="font-size:0.6rem;color:#7a8294;text-transform:uppercase;'
        f'letter-spacing:0.06em;font-weight:600;">{label}</div>'
        f'<div style="font-size:0.84rem;font-weight:600;color:#e8ecf1;margin-top:3px;'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:200px;">'
        f'{name}</div>'
        f'<div style="font-size:0.78rem;font-weight:600;color:{value_color};margin-top:4px;">'
        f'{value_str}</div>'
        f'</div>'
    )


def _connector_html(severity: float) -> str:
    """Horizontal chain-link connector with stress-aware styling."""
    s = _stress_style(severity)
    return (
        f'<div style="display:flex;flex-direction:column;align-items:center;'
        f'min-width:100px;flex:1 1 80px;padding:0 4px;">'
        f'<div style="display:flex;align-items:center;width:100%;">'
        f'<div style="flex:1;height:0;border-top:2px {s["line_style"]} {s["color"]};opacity:0.7;">'
        f'</div>'
        f'<div style="background:{s["bg"]};color:{s["color"]};padding:4px 10px;'
        f'border:1px solid {s["color"]};border-radius:14px;margin:0 6px;'
        f'font-size:0.85rem;font-weight:700;line-height:1;">{s["icon"]}</div>'
        f'<div style="flex:1;height:0;border-top:2px {s["line_style"]} {s["color"]};opacity:0.7;">'
        f'</div>'
        f'</div>'
        f'<div style="font-size:0.6rem;color:{s["color"]};font-weight:700;'
        f'letter-spacing:0.08em;margin-top:6px;">{s["label"]}</div>'
        f'</div>'
    )


def _render_chain_row(c: dict) -> str:
    """Render one full chain: factor(s) → connector → sector → connector → stock."""
    sector = c["sector"]
    sector_row = c["sector_row"]
    top_stock = c["top_stock"]
    factors = c["factors"]
    severity = c["severity"]

    # Factor block(s) — stacked vertically if multiple
    factor_nodes = []
    for fname, fd in factors:
        dod = fd["dod"]
        f_color = "#00d09c" if dod > 0 else "#eb5757" if dod < 0 else "#c9cfd9"
        f_border = ("rgba(235,87,87,0.4)" if abs(dod) >= 5
                    else "rgba(245,166,35,0.4)" if abs(dod) >= 1.5
                    else "#2a3142")
        factor_nodes.append(_node_html(
            "Global Factor", fname, f"{dod:+.1f}%  &middot;  {fd['price']:,.2f}",
            f_color, border=f_border, min_width=170,
        ))
    factor_block = (
        f'<div style="display:flex;flex-direction:column;gap:6px;flex:0 0 auto;">'
        f'{"".join(factor_nodes)}</div>'
    )

    # Sector node
    if sector_row is not None:
        sect_chg = float(sector_row["DoD %"])
        sect_color = "#00d09c" if sect_chg > 0 else "#eb5757" if sect_chg < 0 else "#c9cfd9"
        sect_idx_name = sector_row["Index"]
        sector_node = _node_html(
            "Indian Sector",
            f"{sector} &middot; {sect_idx_name}",
            f"{sect_chg:+.1f}%  &middot;  {float(sector_row['Close']):,.0f}",
            sect_color, min_width=200,
        )
    else:
        sector_node = _node_html(
            "Indian Sector", sector, "no NSE index", "#7a8294", min_width=160,
        )

    # Stock node
    if top_stock:
        stk_chg = top_stock["change"]
        stk_color = "#00d09c" if stk_chg > 0 else "#eb5757" if stk_chg < 0 else "#c9cfd9"
        stock_node = _node_html(
            "Top Mover", top_stock["symbol"], f"{stk_chg:+.1f}%",
            stk_color, min_width=130,
        )
    else:
        stock_node = _node_html(
            "Top Mover", "—", "no Nifty 50 stock", "#7a8294", min_width=130,
        )

    # Sector-level severity for the second connector (sector vs stock —
    # use sector move magnitude if available, else fall back to factor severity)
    sector_severity = (abs(float(sector_row["DoD %"])) if sector_row is not None
                       else severity * 0.5)

    return (
        f'<div style="background:#0f131c;border:1px solid #232834;border-radius:12px;'
        f'padding:14px 16px;margin-bottom:10px;">'
        f'<div style="display:flex;align-items:stretch;flex-wrap:wrap;gap:0;">'
        f'{factor_block}'
        f'{_connector_html(severity)}'
        f'{sector_node}'
        f'{_connector_html(sector_severity)}'
        f'{stock_node}'
        f'</div>'
        f'<div style="font-size:0.7rem;color:#7a8294;margin-top:12px;'
        f'padding-top:10px;border-top:1px solid #232834;line-height:1.4;">'
        f'{c["note"]}</div>'
        f'</div>'
    )
