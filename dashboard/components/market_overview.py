"""Market overview: Nifty 50 KPIs and sectoral chip cards."""

from __future__ import annotations

import streamlit as st
import pandas as pd
from loguru import logger

from dashboard.data_loader import MarketSnapshot, _market_minute_bucket
from dashboard.config import SECTOR_INDEX_TO_SECTOR
from dashboard.sector_signals import compute_sector_signals


def render_key_metrics(snapshot: MarketSnapshot):
    """Render the headline KPI metric cards.

    Every chip with a sensible weekly comparison (Nifty, VIX, USD/INR,
    FII/DII) shows DoD and WoW together with the prior-week anchor labelled
    in plain text — no view-toggle gymnastics, no DoD-mistaken-for-WoW.
    """
    cols = st.columns(5)

    with cols[0]:
        _render_nifty_chip(snapshot)

    with cols[1]:
        _render_price_chip(
            title="India VIX",
            value=f"{snapshot.india_vix:.1f}",
            dod=snapshot.india_vix_change,
            wow=snapshot.india_vix_wow_pct,
            anchor_close=snapshot.india_vix_wow_anchor_close,
            anchor_date=snapshot.india_vix_wow_anchor_date,
            anchor_close_format="{:.1f}",
            inverse=True,
        )

    with cols[2]:
        _render_price_chip(
            title="USD/INR",
            value=f"{snapshot.usdinr:.2f}",
            dod=snapshot.usdinr_change,
            wow=snapshot.usdinr_wow_pct,
            anchor_close=snapshot.usdinr_wow_anchor_close,
            anchor_date=snapshot.usdinr_wow_anchor_date,
            anchor_close_format="{:.2f}",
            inverse=True,
        )

    with cols[3]:
        _render_flow_chip(snapshot)

    with cols[4]:
        _render_breadth_chip(snapshot)


def _delta_color(val: float, inverse: bool = False) -> str:
    """Green/red/grey for a percent delta. `inverse=True` flips for VIX/USDINR
    where rising is bad."""
    if inverse:
        return "#eb5757" if val > 0 else "#00d09c" if val < 0 else "#c9cfd9"
    return "#00d09c" if val > 0 else "#eb5757" if val < 0 else "#c9cfd9"


def _format_anchor_caption(anchor_date: str, anchor_close: float, fmt: str) -> str:
    if not (anchor_date and anchor_close):
        return ""
    try:
        label = pd.Timestamp(anchor_date).strftime("%a %d-%b")
    except Exception:
        label = anchor_date
    return f"WoW vs {label} close ({fmt.format(anchor_close)})"


def _kpi_chip_html(
    title: str,
    headline: str,
    left_label: str,
    left_value: str,
    left_color: str,
    right_label: str,
    right_value: str,
    right_color: str,
    caption: str = "",
) -> str:
    """Shared markup for every headline KPI chip on the dashboard."""
    caption_html = (
        f'<div style="font-size:0.66rem;color:#6b7587;margin-top:8px;">{caption}</div>'
        if caption else ""
    )
    return f"""
        <div style="background:#151922;border:1px solid #232834;border-radius:10px;
                    padding:14px 18px;">
          <div style="font-size:0.68rem;color:#7a8294;font-weight:500;
                      text-transform:uppercase;letter-spacing:0.06em;">{title}</div>
          <div style="font-size:1.4rem;font-weight:600;color:#e8ecf1;
                      letter-spacing:-0.01em;margin-top:2px;">{headline}</div>
          <div style="display:flex;gap:14px;margin-top:8px;font-size:0.78rem;">
            <div>
              <span style="color:#7a8294;font-size:0.62rem;text-transform:uppercase;
                          letter-spacing:0.04em;">{left_label}</span>
              <div style="color:{left_color};font-weight:600;">{left_value}</div>
            </div>
            <div style="border-left:1px solid #232834;padding-left:14px;">
              <span style="color:#7a8294;font-size:0.62rem;text-transform:uppercase;
                          letter-spacing:0.04em;">{right_label}</span>
              <div style="color:{right_color};font-weight:600;">{right_value}</div>
            </div>
          </div>
          {caption_html}
        </div>
    """


def _render_nifty_chip(snapshot: MarketSnapshot):
    """Nifty 50 chip — DoD + WoW with prior-Friday anchor."""
    dod = snapshot.nifty50_change_pct
    wow = snapshot.nifty50_wow_pct
    caption = _format_anchor_caption(
        snapshot.nifty50_wow_anchor_date,
        snapshot.nifty50_wow_anchor_close,
        "{:,.0f}",
    )
    st.markdown(
        _kpi_chip_html(
            title="Nifty 50",
            headline=f"{snapshot.nifty50_close:,.0f}",
            left_label="DoD", left_value=f"{dod:+.1f}%", left_color=_delta_color(dod),
            right_label="WoW", right_value=f"{wow:+.1f}%", right_color=_delta_color(wow),
            caption=caption,
        ),
        unsafe_allow_html=True,
    )


def _render_price_chip(
    title: str,
    value: str,
    dod: float,
    wow: float,
    anchor_close: float,
    anchor_date: str,
    anchor_close_format: str,
    inverse: bool = False,
):
    """Generic price-style KPI chip (VIX, USD/INR)."""
    caption = _format_anchor_caption(anchor_date, anchor_close, anchor_close_format)
    st.markdown(
        _kpi_chip_html(
            title=title,
            headline=value,
            left_label="DoD",
            left_value=f"{dod:+.1f}%",
            left_color=_delta_color(dod, inverse=inverse),
            right_label="WoW",
            right_value=f"{wow:+.1f}%",
            right_color=_delta_color(wow, inverse=inverse),
            caption=caption,
        ),
        unsafe_allow_html=True,
    )


def _render_flow_chip(snapshot: MarketSnapshot):
    """FII/DII flow chip — today's net + week-to-date with prior-week anchor.

    The flow analogue of WoW is "this week's cumulative net buy" vs "last
    week's cumulative net buy" — same Mon-anchored window as every other
    WoW number on the dashboard.
    """
    today_net = snapshot.fii_net_buy + snapshot.dii_net_buy
    wtd_net = snapshot.fii_wtd + snapshot.dii_wtd
    prev_net = snapshot.fii_prev_week + snapshot.dii_prev_week

    caption = ""
    if snapshot.flow_prev_week_label:
        sign = "+" if prev_net > 0 else ""
        caption = f"WoW vs prior wk ({snapshot.flow_prev_week_label}): {sign}{prev_net:,.0f}"

    st.markdown(
        _kpi_chip_html(
            title="FII + DII Net (₹ Cr)",
            headline=f"{today_net:+,.0f}",
            left_label="FII WTD",
            left_value=f"{snapshot.fii_wtd:+,.0f}",
            left_color=_delta_color(snapshot.fii_wtd),
            right_label="DII WTD",
            right_value=f"{snapshot.dii_wtd:+,.0f}",
            right_color=_delta_color(snapshot.dii_wtd),
            caption=caption,
        ),
        unsafe_allow_html=True,
    )


def _render_breadth_chip(snapshot: MarketSnapshot):
    """Advance/Decline breadth chip — kept as a count, no WoW concept."""
    total = snapshot.advance_count + snapshot.decline_count + snapshot.unchanged_count
    if total > 0:
        adv_pct = snapshot.advance_count / total * 100
        dec_pct = snapshot.decline_count / total * 100
    else:
        adv_pct = dec_pct = 0.0

    headline = f"{snapshot.advance_count} / {snapshot.decline_count}"
    st.markdown(
        _kpi_chip_html(
            title="Advance / Decline",
            headline=headline,
            left_label="Up", left_value=f"{adv_pct:.0f}%", left_color="#00d09c",
            right_label="Down", right_value=f"{dec_pct:.0f}%", right_color="#eb5757",
            caption=f"of {total} largecap+midcap stocks" if total else "",
        ),
        unsafe_allow_html=True,
    )


def render_sectoral_heatmap(snapshot: MarketSnapshot):
    """Render sectoral indices as colored chip cards with per-sector context."""
    st.markdown("#### Sectors")

    if snapshot.sectoral_data.empty:
        st.info("Sectoral index data unavailable")
        return

    df = snapshot.sectoral_data.copy()
    sort_col = "DoD %" if snapshot.view == "daily" else "WoW %"
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    # Per-sector stock-level stats
    sector_stats = _compute_sector_stats(snapshot)

    # Compute S/R + bias overlay lazily here rather than during snapshot
    # construction. `_fetch_sector_history` is already called from
    # `render_sector_deep_dive` in the same tab, so by the time the user
    # sees the chips this is a cache hit; doing it inside
    # `load_market_snapshot` made the whole page block on a 2y × 18-sector
    # fetch even when the user was on a different tab.
    signals = _compute_sector_signals_cached()

    change_label = "Day" if snapshot.view == "daily" else "Week"
    st.markdown(
        f'<div style="font-size:0.72rem;color:#7a8294;margin-bottom:6px;">'
        f'Sorted by {change_label} change &middot; '
        f'Green = positive momentum &middot; '
        f'<span style="color:#00d09c;">&#9650;</span> advancing / '
        f'<span style="color:#eb5757;">&#9660;</span> declining stocks in sector &middot; '
        f'<b>S/R</b> = 20-day support/resistance, <b>Bias</b> = next-session lean '
        f'(RSI + EMA-21 + S/R proximity)</div>',
        unsafe_allow_html=True,
    )

    chips = []
    for _, row in df.iterrows():
        idx_name = row["Index"]
        change = row[sort_col]
        close = row["Close"]
        wow = row["WoW %"]
        mom = row["1M %"]
        stats = sector_stats.get(idx_name, {})

        # Gradient + border based on performance intensity
        if change > 1.5:
            bg = "linear-gradient(135deg, #0a2e1c 0%, #151922 100%)"
            border = "rgba(0,208,156,0.4)"
        elif change > 0:
            bg = "linear-gradient(135deg, #0f2018 0%, #151922 100%)"
            border = "rgba(0,208,156,0.2)"
        elif change > -1.5:
            bg = "linear-gradient(135deg, #2a1515 0%, #151922 100%)"
            border = "rgba(235,87,87,0.2)"
        else:
            bg = "linear-gradient(135deg, #3a1515 0%, #151922 100%)"
            border = "rgba(235,87,87,0.4)"

        chg_color = "#00d09c" if change > 0 else "#eb5757" if change < 0 else "#c9cfd9"
        wow_color = "#00d09c" if wow > 0 else "#eb5757" if wow < 0 else "#c9cfd9"
        mom_color = "#00d09c" if mom > 0 else "#eb5757" if mom < 0 else "#c9cfd9"

        # Advance / decline row
        ad_html = ""
        if stats:
            top_sym = stats["top_symbol"]
            top_chg = stats["top_change"]
            top_color = "#00d09c" if top_chg > 0 else "#eb5757"
            ad_html = (
                f'<div style="margin-top:8px;font-size:0.66rem;color:#7a8294;">'
                f'<span style="color:#00d09c;">&#9650;{stats["adv"]}</span>'
                f' &middot; <span style="color:#eb5757;">&#9660;{stats["dec"]}</span>'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;'
                f'<span style="color:#e8ecf1;">{top_sym}</span> '
                f'<span style="color:{top_color};">{top_chg:+.1f}%</span>'
                f'</div>'
            )

        # Support / resistance + next-session bias overlay
        sr_html = _sector_sr_html(close, signals.get(idx_name))

        # Momentum verdict
        verdict = _sector_verdict(change, wow, mom)
        verdict_html = ""
        if verdict:
            verdict_html = (
                f'<div style="margin-top:7px;padding-top:6px;border-top:1px solid #232834;'
                f'font-size:0.64rem;color:#7a8294;line-height:1.3;">{verdict}</div>'
            )

        chip = (
            f'<div style="background:{bg};border:1px solid {border};border-radius:12px;'
            f'padding:14px 16px;min-width:230px;flex:1 1 230px;max-width:300px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="font-weight:700;font-size:0.88rem;color:#e8ecf1;">{idx_name}</span>'
            f'<span style="color:{chg_color};font-weight:600;font-size:0.88rem;">{change:+.1f}%</span>'
            f'</div>'
            f'<div style="font-size:1.0rem;font-weight:500;color:#c9cfd9;margin-top:4px;">'
            f'{close:,.0f}</div>'
            f'<div style="display:flex;gap:18px;margin-top:10px;font-size:0.72rem;">'
            f'<div><span style="color:#7a8294;font-size:0.62rem;text-transform:uppercase;'
            f'letter-spacing:0.04em;">Week</span>'
            f'<br><span style="color:{wow_color};font-weight:500;">{wow:+.1f}%</span></div>'
            f'<div><span style="color:#7a8294;font-size:0.62rem;text-transform:uppercase;'
            f'letter-spacing:0.04em;">Month</span>'
            f'<br><span style="color:{mom_color};font-weight:500;">{mom:+.1f}%</span></div>'
            f'</div>'
            f'{ad_html}{sr_html}{verdict_html}'
            f'</div>'
        )
        chips.append(chip)

    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:8px;">{"".join(chips)}</div>',
        unsafe_allow_html=True,
    )


def _compute_sector_stats(snapshot: MarketSnapshot) -> dict:
    """Advance/decline counts and top mover per sector index."""
    if snapshot.stock_changes.empty:
        return {}

    change_col = "dod_pct" if snapshot.view == "daily" else "wow_pct"
    result = {}

    for idx_name, sector_name in SECTOR_INDEX_TO_SECTOR.items():
        stocks = snapshot.stock_changes[snapshot.stock_changes["sector"] == sector_name]
        if stocks.empty:
            continue
        adv = int((stocks[change_col] > 0.1).sum())
        dec = int((stocks[change_col] < -0.1).sum())
        top = stocks.sort_values(change_col, ascending=False).iloc[0]
        result[idx_name] = {
            "adv": adv,
            "dec": dec,
            "top_symbol": top["symbol"],
            "top_change": top[change_col],
        }
    return result


def _compute_sector_signals_cached() -> dict[str, dict]:
    """Return `{sector_index: signal_dict}` using the shared 2y close cache.

    Wrapped here (not at the data-loader layer) so the cost is only paid on
    the Sectors tab. Failures degrade gracefully — the chips just lose the
    S/R overlay rather than blanking out.
    """
    # Local import keeps `dashboard.components.sector_deep_dive` out of the
    # module-load import cycle.
    from dashboard.components.sector_deep_dive import _fetch_sector_history

    try:
        histories = _fetch_sector_history(_bucket=_market_minute_bucket())
    except Exception as e:
        logger.warning(f"sector signals: history fetch failed: {e}")
        return {}
    signals = compute_sector_signals(histories)
    return {k: v.as_dict() for k, v in signals.items()}


def _bias_style(bias: str) -> tuple[str, str]:
    """Color + glyph for a directional bias label."""
    if bias == "Bullish":
        return "#00d09c", "▲"
    if bias == "Bearish":
        return "#eb5757", "▼"
    return "#c9cfd9", "→"


def _sector_sr_html(close: float, signal: dict | None) -> str:
    """Compact Support / Resistance / Bias mini-block for a sector chip.

    Returns "" when no signal is available so the chip still renders for
    sectors with insufficient history.
    """
    if not signal:
        return ""

    support = signal.get("support") or 0.0
    resistance = signal.get("resistance") or 0.0
    bias = signal.get("bias") or "Neutral"
    rsi = signal.get("rsi") or 0.0
    bias_color, bias_glyph = _bias_style(bias)

    # Position-in-range bar: how far close is between S and R (0-100%)
    if resistance > support:
        pos_pct = max(0.0, min(100.0, (close - support) / (resistance - support) * 100))
    else:
        pos_pct = 50.0

    proximity_note = ""
    if signal.get("near_resistance"):
        proximity_note = " &middot; <span style=\"color:#eb5757;\">at R</span>"
    elif signal.get("near_support"):
        proximity_note = " &middot; <span style=\"color:#00d09c;\">at S</span>"

    rationale = signal.get("rationale", "") or ""

    return (
        f'<div style="margin-top:8px;padding-top:7px;border-top:1px solid #232834;'
        f'font-size:0.66rem;color:#7a8294;line-height:1.4;" title="{rationale}">'
        # S/R values row
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span><span style="color:#00d09c;font-weight:600;">S</span> {support:,.0f}'
        f' &middot; <span style="color:#eb5757;font-weight:600;">R</span> {resistance:,.0f}</span>'
        f'<span style="color:{bias_color};font-weight:600;">{bias_glyph} {bias}</span>'
        f'</div>'
        # Position-in-range bar
        f'<div style="margin-top:5px;height:4px;background:rgba(255,255,255,0.06);'
        f'border-radius:2px;position:relative;">'
        f'<div style="position:absolute;left:{pos_pct:.0f}%;top:-3px;width:2px;height:10px;'
        f'background:#e8ecf1;border-radius:1px;transform:translateX(-1px);"></div>'
        f'</div>'
        f'<div style="margin-top:5px;font-size:0.62rem;color:#6b7587;">'
        f'RSI {rsi:.0f}{proximity_note}</div>'
        f'</div>'
    )


def _sector_verdict(day_chg, wow, mom):
    """One-line momentum verdict for a sector chip."""
    if wow > 3 and mom > 5:
        return "Strong multi-week rally — momentum intact"
    if wow > 2 and day_chg > 1:
        return "Accelerating momentum this week"
    if mom > 5 and day_chg < -1:
        return "Monthly gainer pulling back — dip opportunity?"
    if wow < -3 and mom < -3:
        return "Sustained weakness — avoid or wait for reversal"
    if wow < -2 and day_chg > 1:
        return "Bouncing from weekly low — watch for follow-through"
    if mom > 3 and wow < 0:
        return "Monthly uptrend, short-term pause"
    if mom < -3 and wow > 0:
        return "Short-term bounce in a downtrend"
    return ""
