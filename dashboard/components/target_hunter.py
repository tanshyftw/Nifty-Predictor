"""
Target Hunter view: surface stocks where analyst price targets are materially
above current price, validated by a transparent technical score, with a
conviction band, an ETA-to-target bucket, and a distance-from-ATH chip filter.

Universe switch covers Nifty 50, a curated Midcap subset, or both.
"""

import sys
import os
from datetime import datetime
from html import escape

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.nifty50_tickers import NIFTY50_STOCKS
from config.midcap_tickers import MIDCAP_STOCKS
from config.largecap_extra_tickers import LARGECAP_NEXT50_STOCKS
from dashboard.analyst_data import AnalystSnapshot, load_analyst_snapshot
from dashboard.config import (
    ANALYST_CACHE_TTL_SECONDS,
    SECTOR_SUPPLY_CHAIN,
    TARGET_HUNTER_MIN_ANALYSTS,
    TARGET_HUNTER_MIN_UPSIDE,
    TARGET_HUNTER_TECH_SCORE_THRESHOLD,
)


# Universe → stock dict
_UNIVERSE_MAP = {
    "Nifty 50": NIFTY50_STOCKS,
    "Largecap+": LARGECAP_NEXT50_STOCKS,
    "Midcap": MIDCAP_STOCKS,
}

def _segmented(label, options, *, default=None, key=None, selection_mode="single"):
    """st.segmented_control if available (Streamlit ≥1.40), else radio/multiselect."""
    fn = getattr(st, "segmented_control", None)
    if fn is not None:
        try:
            return fn(
                label, options, default=default, key=key,
                selection_mode=selection_mode, label_visibility="visible",
            )
        except TypeError:
            return fn(label, options, default=default, key=key)
    if selection_mode == "multi":
        return st.multiselect(label, options, default=default or [], key=key)
    idx = options.index(default) if default in options else 0
    return st.radio(label, options, index=idx, horizontal=True, key=key)


HORIZON_MONTHS = {"1M": 1, "3M": 3, "6M": 6, "12M": 12}
DRAWDOWN_CHIPS = {"≥10%": 10.0, "≥20%": 20.0, "≥30%": 30.0}
CONVICTION_COLORS = {
    "High":   {"bg": "#003d2e", "fg": "#00d09c", "border": "#00a67c"},
    "Medium": {"bg": "#3d2f00", "fg": "#ffc247", "border": "#b88a2e"},
    "Low":    {"bg": "#3d1a1a", "fg": "#ff6b6b", "border": "#a83838"},
}
HORIZON_COLORS = {
    "1M":  {"bg": "#1a2d3d", "fg": "#5eb8ff", "border": "#2f6fa8"},
    "3M":  {"bg": "#1a2d3d", "fg": "#5eb8ff", "border": "#2f6fa8"},
    "6M":  {"bg": "#22203d", "fg": "#9d8cff", "border": "#5a4fb8"},
    "12M": {"bg": "#2d2038", "fg": "#c792e3", "border": "#7a4fa8"},
}


# ---------------------------------------------------------------------------
# Technical scoring
# ---------------------------------------------------------------------------

def _universe_stocks(universe: str) -> dict:
    """Return the combined stock dict for the selected universe label.

    "Both" spans Nifty 50 + Nifty Next 50 + Midcap so it stays consistent
    with the rest of the dashboard's expanded universe — otherwise names
    classified as Nifty Next 50 (DMART, HAL, BEL, ...) would silently
    disappear from this view.
    """
    if universe == "Both":
        merged = dict(NIFTY50_STOCKS)
        for sym, meta in LARGECAP_NEXT50_STOCKS.items():
            merged.setdefault(sym, meta)
        for sym, meta in MIDCAP_STOCKS.items():
            merged.setdefault(sym, meta)
        return merged
    return _UNIVERSE_MAP.get(universe, NIFTY50_STOCKS)


@st.cache_data(ttl=ANALYST_CACHE_TTL_SECONDS, show_spinner="Loading 5-year history for technicals & ATH...")
def _load_technical_history(universe: str = "Nifty 50") -> dict:
    """Fetch 5-year daily OHLCV for the universe. Returns dict[symbol] -> DataFrame.

    5y (vs the old 1y) lets us compute distance-from-ATH without a second fetch.
    """
    stocks = _universe_stocks(universe)
    tickers = " ".join(meta[0] for meta in stocks.values())
    try:
        data = yf.download(
            tickers,
            period="5y",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        logger.error(f"Technical history fetch failed: {e}")
        return {}

    if data.empty:
        return {}

    result = {}
    for symbol, (yahoo_ticker, _, _) in stocks.items():
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if yahoo_ticker in data.columns.get_level_values(0):
                    sub = data[yahoo_ticker].dropna(how="all")
                else:
                    continue
            else:
                sub = data.dropna(how="all")
            if sub.empty or "Close" not in sub.columns:
                continue
            result[symbol] = sub
        except Exception:
            continue
    return result


def _rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else 50.0


def _macd(close: pd.Series) -> tuple[float, float]:
    """Returns (macd_line_last, signal_line_last)."""
    if len(close) < 35:
        return 0.0, 0.0
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(signal.iloc[-1])


def _technical_breakdown(df: pd.DataFrame) -> dict:
    """Compute indicators + a 0–100 score. Returns dict of metrics."""
    if df is None or df.empty or "Close" not in df.columns:
        return {}
    close = df["Close"].dropna()
    if len(close) < 30:
        return {}

    last = float(close.iloc[-1])
    sma50 = float(close.tail(50).mean()) if len(close) >= 50 else last
    sma200 = float(close.tail(200).mean()) if len(close) >= 200 else last
    rsi = _rsi(close)
    macd_line, signal_line = _macd(close)

    vol_ratio = 1.0
    if "Volume" in df.columns:
        vol = df["Volume"].dropna()
        if len(vol) >= 20:
            vol_ratio = float(vol.iloc[-1] / max(vol.tail(20).mean(), 1))

    high_52w = float(close.tail(252).max()) if len(close) >= 20 else last
    low_52w = float(close.tail(252).min()) if len(close) >= 20 else last
    pct_of_52w = (last - low_52w) / (high_52w - low_52w) * 100 if high_52w > low_52w else 50.0

    signals = {
        "above_sma50": last > sma50,
        "above_sma200": last > sma200,
        "rsi_healthy": 40 <= rsi <= 65,
        "rsi_not_overbought": rsi < 70,
        "macd_bullish": macd_line > signal_line,
        "volume_support": vol_ratio > 0.8,
    }
    score = sum(signals.values()) * 100 / 6

    return {
        "close": last,
        "sma50": sma50,
        "sma200": sma200,
        "rsi": rsi,
        "macd": macd_line,
        "macd_signal": signal_line,
        "vol_ratio": vol_ratio,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "pct_of_52w": pct_of_52w,
        "score": score,
        "signals": signals,
    }


# ---------------------------------------------------------------------------
# Fair exit + timeline POV
# ---------------------------------------------------------------------------

def _compute_fair_exit(current: float, target_mean: float, target_high: float, tech_score: float) -> float:
    """Blend analyst mean with technical confidence. Capped at analyst high."""
    if current <= 0 or target_mean <= 0:
        return 0.0
    upside = (target_mean - current) / current
    tech_factor = tech_score / 100.0
    fair = current * (1 + upside * (0.5 + 0.5 * tech_factor))
    if target_high > 0:
        fair = min(fair, target_high)
    return fair


def _compute_timeline(beta: float, tech_score: float, macd_bullish: bool, rsi: float) -> str:
    """~6 / ~12 / ~18+ months based on momentum."""
    if tech_score < TARGET_HUNTER_TECH_SCORE_THRESHOLD:
        return "~18+ months"
    if beta >= 1.3 and macd_bullish and rsi < 70:
        return "~6 months"
    return "~12 months"


# ---------------------------------------------------------------------------
# Conviction band + ETA bucket
# ---------------------------------------------------------------------------

def _conviction_band(
    spread_pct: float,
    num_analysts: int,
    tech_score: float,
    upside_pct: float,
    macd_bullish: bool,
) -> tuple[str, float]:
    """Blend spread tightness, coverage, and tech alignment into High/Med/Low.

    Returns (label, score_0_1). Score is used to rank Top Picks.
    """
    # Spread tightness: 0% spread → 1.0, 40%+ → 0.0
    spread_score = max(0.0, 1.0 - (spread_pct / 40.0))

    # Coverage: 0 analysts → 0.0, 20+ → 1.0
    coverage_score = min(1.0, num_analysts / 20.0) if num_analysts else 0.0

    # Technical alignment: tech_score on 0..100 and direction match
    alignment = tech_score / 100.0
    if upside_pct >= 0 and macd_bullish:
        alignment = min(1.0, alignment + 0.15)
    elif upside_pct >= 0 and not macd_bullish:
        alignment = max(0.0, alignment - 0.15)

    score = 0.45 * spread_score + 0.30 * coverage_score + 0.25 * alignment

    if score >= 0.65:
        band = "High"
    elif score >= 0.40:
        band = "Medium"
    else:
        band = "Low"
    return band, round(score, 3)


def _eta_months(upside_pct: float, beta: float, macd_bullish: bool, rsi: float) -> tuple[float, str]:
    """Estimate months until target is reached, bucketed to 1M/3M/6M/12M.

    Larger required move → longer. Bullish momentum + higher beta shortens.
    Overbought (RSI > 70) stretches the ETA.
    """
    magnitude = abs(upside_pct)
    base = float(np.clip(magnitude / 3.0, 1.0, 12.0))  # 3% → 1M, 36%+ → 12M

    base *= 0.75 if macd_bullish else 1.25
    base /= float(np.clip(beta, 0.5, 2.0))
    if not np.isnan(rsi):
        if upside_pct >= 0 and rsi > 70:
            base *= 1.15
        elif upside_pct >= 0 and rsi < 35:
            base *= 0.85

    eta = float(np.clip(base, 0.5, 24.0))

    if eta <= 1.5:
        bucket = "1M"
    elif eta <= 3.5:
        bucket = "3M"
    elif eta <= 6.5:
        bucket = "6M"
    else:
        bucket = "12M"
    return round(eta, 1), bucket


def _scale_to_horizon(current: float, target_mean: float, eta_m: float, horizon: str) -> float:
    """Pro-rate the target for a user-selected horizon.

    If full target ETA is 9 months and user picks a 3M horizon, show the
    interim price the stock should reach at 3/9 of the way there.
    """
    if current <= 0 or target_mean <= 0 or eta_m <= 0:
        return 0.0
    horizon_m = HORIZON_MONTHS.get(horizon, 12)
    ratio = min(1.0, horizon_m / eta_m)
    return current + (target_mean - current) * ratio


# ---------------------------------------------------------------------------
# Chip / badge HTML helpers
# ---------------------------------------------------------------------------

def _badge_html(label: str, palette: dict) -> str:
    """Inline colored chip for use inside st.markdown (unsafe_allow_html=True)."""
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;'
        f'font-size:0.72rem;font-weight:600;letter-spacing:0.02em;'
        f'background:{palette["bg"]};color:{palette["fg"]};'
        f'border:1px solid {palette["border"]};">{label}</span>'
    )


def _conviction_badge(band: str) -> str:
    return _badge_html(band, CONVICTION_COLORS[band])


def _horizon_badge(bucket: str) -> str:
    return _badge_html(bucket, HORIZON_COLORS.get(bucket, HORIZON_COLORS["12M"]))


_CHIP_BAR_CSS = """
<style>
.chip-bar {
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
    padding: 10px 12px; border: 1px solid #232834; border-radius: 10px;
    background: #0f131c; margin: 8px 0 14px 0;
}
.chip-bar .chip-label {
    font-size: 0.72rem; color: #7a8294; text-transform: uppercase;
    letter-spacing: 0.06em; margin-right: 2px; font-weight: 600;
}
.chip-bar [data-testid="stSegmentedControl"] button {
    font-size: 0.8rem !important; padding: 4px 12px !important;
}
.top-picks-grid {
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px; margin: 10px 0 18px 0;
}
.top-pick-card {
    background: linear-gradient(180deg, #141924 0%, #0f131c 100%);
    border: 1px solid #232834; border-radius: 12px; padding: 14px 16px;
}
.top-pick-card:hover { border-color: #00a67c; }
.top-pick-card .tp-symbol { font-size: 1.05rem; font-weight: 700; color: #e8ecf1; }
.top-pick-card .tp-company { font-size: 0.78rem; color: #7a8294; margin-bottom: 8px; }
.top-pick-card .tp-upside { font-size: 1.4rem; font-weight: 700; color: #00d09c; margin: 4px 0; }
.top-pick-card .tp-meta { font-size: 0.78rem; color: #c9cfd9; margin-top: 6px; }
.active-filters { font-size: 0.78rem; color: #7a8294; margin: 4px 0 8px 0; }
.active-filters .pill {
    display: inline-block; padding: 2px 8px; border-radius: 6px;
    background: #1a1f2b; color: #c9cfd9; margin-right: 6px; font-size: 0.72rem;
}
</style>
"""


def _risk_flags(num_analysts: int, rsi: float, high: float, low: float, mean: float) -> list:
    flags = []
    if num_analysts < TARGET_HUNTER_MIN_ANALYSTS:
        flags.append("thin coverage")
    if rsi > 70:
        flags.append("overbought")
    if mean > 0 and (high - low) / mean > 0.30:
        flags.append("wide target range")
    return flags


# ---------------------------------------------------------------------------
# Table renderer with colored badges
# ---------------------------------------------------------------------------

_TABLE_CSS = """
<style>
.th-table {
    width: 100%; border-collapse: collapse; font-size: 0.82rem;
    background: #0f131c; border: 1px solid #232834; border-radius: 10px;
    overflow: hidden; margin: 8px 0 16px 0;
}
.th-table th {
    text-align: left; padding: 10px 12px; font-weight: 600;
    color: #c9cfd9; background: #151922; border-bottom: 1px solid #232834;
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em;
}
.th-table td {
    padding: 10px 12px; border-bottom: 1px solid #1a1f2b; color: #c9cfd9;
}
.th-table tr:hover td { background: #141924; }
.th-table .num { text-align: right; font-variant-numeric: tabular-nums; }
.th-table .pos { color: #00d09c; font-weight: 600; }
.th-table .neg { color: #ff6b6b; font-weight: 600; }
</style>
"""


def _build_table_html(df: pd.DataFrame, horizon: str) -> str:
    """Render the candidates table with inline colored conviction + ETA badges."""
    rows_html = []
    for _, r in df.iterrows():
        upside_cls = "pos" if r["Upside %"] >= 0 else "neg"
        dd_cls = "neg" if r["Drawdown %"] >= 20 else ""
        rows_html.append(
            f'<tr>'
            f'<td><b>{r["Symbol"]}</b><div style="font-size:0.72rem;color:#7a8294;">{r["Company"]}</div></td>'
            f'<td>{r["Sector"]}</td>'
            f'<td style="font-size:0.72rem;color:#7a8294;">{r["Universe"]}</td>'
            f'<td class="num">₹{r["Current"]:,.0f}</td>'
            f'<td class="num">₹{r["Mean Target"]:,.0f}</td>'
            f'<td class="num">₹{r["Horizon Target"]:,.0f}</td>'
            f'<td class="num {upside_cls}">{r["Upside %"]:+.1f}%</td>'
            f'<td>{_conviction_badge(r["Conviction"])}</td>'
            f'<td>{_horizon_badge(r["ETA"])}</td>'
            f'<td class="num {dd_cls}">{r["Drawdown %"]:.1f}%</td>'
            f'<td class="num">{int(r["Analysts"])}</td>'
            f'<td class="num">{int(r["Tech Score"])}</td>'
            f'</tr>'
        )
    return (
        _TABLE_CSS
        + '<table class="th-table"><thead><tr>'
        + '<th>Stock</th><th>Sector</th><th>Set</th>'
        + '<th class="num">Now</th><th class="num">Target</th>'
        + f'<th class="num">{horizon} Target</th>'
        + '<th class="num">Upside</th>'
        + '<th>Conviction</th><th>ETA</th>'
        + '<th class="num">Off ATH</th><th class="num">Analysts</th><th class="num">Tech</th>'
        + '</tr></thead><tbody>'
        + "".join(rows_html)
        + "</tbody></table>"
    )


_OVERVIEW_TILE_CSS = """
<style>
.target-tile-grid {
    display:grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap:12px;
    margin:10px 0 4px 0;
}
.target-tile {
    background:linear-gradient(180deg, #141924 0%, #0f131c 100%);
    border:1px solid #232834;
    border-radius:10px;
    padding:14px 16px;
}
.target-tile:hover { border-color:#00a67c; }
.target-tile .symbol { color:#e8ecf1; font-size:1.0rem; font-weight:700; }
.target-tile .company { color:#7a8294; font-size:0.76rem; margin-top:2px; min-height:18px; }
.target-tile .upside { color:#00d09c; font-size:1.35rem; font-weight:700; margin:9px 0 2px; }
.target-tile .meta { color:#c9cfd9; font-size:0.78rem; line-height:1.45; }
.target-tile .muted { color:#7a8294; }
.overview-load-card {
    background:#11151d;
    border:1px solid #232834;
    border-radius:10px;
    padding:14px 16px;
    margin-top:8px;
}
@media (max-width: 900px) {
    .target-tile-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
    .target-tile-grid { grid-template-columns: 1fr; }
}
</style>
"""


def _build_target_candidates(
    *,
    universe: str = "Nifty 50",
    horizon: str = "12M",
    min_upside: float = TARGET_HUNTER_MIN_UPSIDE,
    min_analysts: int = TARGET_HUNTER_MIN_ANALYSTS,
    min_tech_score: int = TARGET_HUNTER_TECH_SCORE_THRESHOLD,
) -> tuple[pd.DataFrame, AnalystSnapshot]:
    """Build the Target Hunter candidate frame without rendering filters."""
    analyst = load_analyst_snapshot(universe=universe)
    tech_history = _load_technical_history(universe=universe)
    stocks_meta = _universe_stocks(universe)

    rows = []
    for symbol, t in analyst.targets.items():
        current = t.get("currentPrice") or 0.0
        target_mean = t.get("targetMeanPrice") or 0.0
        target_high = t.get("targetHighPrice") or 0.0
        target_low = t.get("targetLowPrice") or 0.0
        num_analysts = t.get("numberOfAnalystOpinions") or 0

        if current <= 0 or target_mean <= 0:
            continue

        upside = (target_mean - current) / current
        spread_pct = (
            (target_high - target_low) / target_mean * 100
            if target_mean and target_high and target_low else 100.0
        )
        tech = _technical_breakdown(tech_history.get(symbol))
        tech_score = tech.get("score", 0.0)
        rsi = tech.get("rsi", 50.0)
        macd_bullish = tech.get("signals", {}).get("macd_bullish", False)
        beta = t.get("beta") or 1.0
        conviction, c_score = _conviction_band(
            spread_pct, num_analysts, tech_score, upside * 100, macd_bullish,
        )
        eta_m, eta_bucket = _eta_months(upside * 100, beta, macd_bullish, rsi)
        horizon_target = _scale_to_horizon(current, target_mean, eta_m, horizon)
        fair_exit = _compute_fair_exit(current, target_mean, target_high, tech_score)

        meta = stocks_meta.get(symbol, (None, symbol, "Unknown"))
        rows.append({
            "Symbol": symbol,
            "Company": meta[1],
            "Sector": meta[2],
            "Current": current,
            "Mean Target": target_mean,
            "Horizon Target": round(horizon_target, 2),
            "Upside %": round(upside * 100, 2),
            "Analysts": num_analysts,
            "Tech Score": round(tech_score, 0),
            "Fair Exit": round(fair_exit, 2),
            "Conviction": conviction,
            "Conviction Score": c_score,
            "ETA": eta_bucket,
            "ETA (months)": eta_m,
            "Drawdown %": round(t.get("drawdown_pct") or 0.0, 2),
        })

    if not rows:
        return pd.DataFrame(), analyst

    df = pd.DataFrame(rows)
    df = df[
        (df["Upside %"] >= min_upside * 100)
        & (df["Analysts"] >= min_analysts)
        & (df["Tech Score"] >= min_tech_score)
    ]
    return df.sort_values("Conviction Score", ascending=False), analyst


def render_target_hunter_tiles(limit: int = 6):
    """Render compact Target Hunter cards inside Market Overview."""
    st.markdown(_OVERVIEW_TILE_CSS, unsafe_allow_html=True)
    st.subheader("Target Hunter")

    load_key = "overview_target_hunter_loaded"
    if not st.session_state.get(load_key, False):
        st.markdown(
            '<div class="overview-load-card">'
            '<div style="color:#e8ecf1;font-weight:600;margin-bottom:4px;">Analyst upside tiles</div>'
            '<div style="color:#7a8294;font-size:0.82rem;">Loads on demand so Market Overview stays fast.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("Load target tiles", key="load_target_tiles"):
            st.session_state[load_key] = True
            st.rerun()
        return

    with st.spinner("Finding analyst-backed upside ideas..."):
        df, analyst = _build_target_candidates(universe="Nifty 50", horizon="12M")

    if df.empty:
        st.info("No Target Hunter candidates passed the default quality bar.")
        return

    tiles = []
    for _, r in df.head(limit).iterrows():
        upside_color = "#00d09c" if r["Upside %"] >= 0 else "#ff6b6b"
        tiles.append(
            "<div class='target-tile'>"
            f"<div class='symbol'>{escape(str(r['Symbol']))}</div>"
            f"<div class='company'>{escape(str(r['Company']))} · {escape(str(r['Sector']))}</div>"
            f"<div class='upside' style='color:{upside_color};'>{r['Upside %']:+.1f}%</div>"
            f"<div class='meta'>Now ₹{r['Current']:,.0f} → Target ₹{r['Mean Target']:,.0f}</div>"
            f"<div class='meta muted'>{r['Conviction']} conviction · ETA {r['ETA']} · "
            f"{int(r['Analysts'])} analysts · Tech {int(r['Tech Score'])}/100</div>"
            "</div>"
        )
    st.markdown(f"<div class='target-tile-grid'>{''.join(tiles)}</div>", unsafe_allow_html=True)
    st.caption(
        f"Default scan: Nifty 50, {TARGET_HUNTER_MIN_UPSIDE:.0%}+ upside, "
        f"{TARGET_HUNTER_MIN_ANALYSTS}+ analysts, {TARGET_HUNTER_TECH_SCORE_THRESHOLD}+ tech score. "
        f"Snapshot {analyst.timestamp.strftime('%I:%M %p IST')}."
    )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_target_hunter():
    """Main entry: render the Target Hunter view."""
    st.markdown(_CHIP_BAR_CSS, unsafe_allow_html=True)
    st.title("Target Hunter — Analyst Upside Finder")
    st.caption(
        "Stocks where the analyst mean target sits materially above the current price, "
        "validated against a transparent 6-factor technical score. Toggle the chips to shard "
        "the basis by conviction, expected horizon, and distance from all-time high."
    )

    # --- Chip bar: universe / conviction / horizon / ATH drawdown / clear ---
    st.markdown('<div class="chip-label">Filters</div>', unsafe_allow_html=True)
    chip_cols = st.columns([1.2, 1.4, 1.4, 1.4, 0.8])
    with chip_cols[0]:
        universe = _segmented(
            "Universe", ["Nifty 50", "Largecap+", "Midcap", "Both"],
            default=st.session_state.get("th_universe", "Nifty 50"),
            key="th_universe",
        ) or "Nifty 50"
    with chip_cols[1]:
        conviction_pick = _segmented(
            "Conviction", ["High", "Medium", "Low"],
            default=st.session_state.get("th_conviction"),
            selection_mode="multi",
            key="th_conviction",
        ) or []
    with chip_cols[2]:
        horizon = _segmented(
            "Horizon", ["1M", "3M", "6M", "12M"],
            default=st.session_state.get("th_horizon", "12M"),
            key="th_horizon",
        ) or "12M"
    with chip_cols[3]:
        drawdown_pick = _segmented(
            "Down from ATH", list(DRAWDOWN_CHIPS.keys()),
            default=st.session_state.get("th_drawdown"),
            key="th_drawdown",
        )
    with chip_cols[4]:
        if st.button("Clear all", use_container_width=True, key="th_clear"):
            for k in ("th_conviction", "th_drawdown", "th_universe", "th_horizon"):
                st.session_state.pop(k, None)
            st.rerun()

    # --- Sidebar min-bar filters (kept from original, still useful) ---
    st.sidebar.subheader("Target Hunter filters")
    min_upside = st.sidebar.slider(
        "Min upside (%)", 0, 50,
        int(TARGET_HUNTER_MIN_UPSIDE * 100), step=5,
    ) / 100
    min_analysts = st.sidebar.slider(
        "Min analyst count", 1, 30, TARGET_HUNTER_MIN_ANALYSTS,
    )
    min_tech_score = st.sidebar.slider(
        "Min technical score", 0, 100, TARGET_HUNTER_TECH_SCORE_THRESHOLD, step=10,
    )

    # --- Load data for the selected universe ---
    analyst = load_analyst_snapshot(universe=universe)
    tech_history = _load_technical_history(universe=universe)

    if not analyst.targets:
        st.warning(
            "No analyst coverage returned from Yahoo Finance. "
            "This can happen on weekends, during Yahoo API hiccups, or if the cache is empty. "
            "Try Refresh Data in the sidebar."
        )
        return

    stocks_meta = _universe_stocks(universe)

    # --- Build rows ---
    rows = []
    detail_cache = {}

    for symbol, t in analyst.targets.items():
        current = t.get("currentPrice") or 0.0
        target_mean = t.get("targetMeanPrice") or 0.0
        target_high = t.get("targetHighPrice") or 0.0
        target_low = t.get("targetLowPrice") or 0.0
        num_analysts = t.get("numberOfAnalystOpinions") or 0

        if current <= 0 or target_mean <= 0:
            continue

        upside = (target_mean - current) / current
        spread_pct = (
            (target_high - target_low) / target_mean * 100
            if target_mean and target_high and target_low else 100.0
        )

        tech = _technical_breakdown(tech_history.get(symbol))
        tech_score = tech.get("score", 0.0)
        rsi = tech.get("rsi", 50.0)
        macd_bullish = tech.get("signals", {}).get("macd_bullish", False)
        beta = t.get("beta") or 1.0

        conviction, c_score = _conviction_band(
            spread_pct, num_analysts, tech_score, upside * 100, macd_bullish,
        )
        eta_m, eta_bucket = _eta_months(upside * 100, beta, macd_bullish, rsi)
        horizon_target = _scale_to_horizon(current, target_mean, eta_m, horizon)

        ath = t.get("ath") or 0.0
        drawdown_pct = t.get("drawdown_pct") or 0.0

        fair_exit = _compute_fair_exit(current, target_mean, target_high, tech_score)
        timeline_str = _compute_timeline(beta, tech_score, macd_bullish, rsi)
        flags = _risk_flags(num_analysts, rsi, target_high, target_low, target_mean)

        meta = stocks_meta.get(symbol, (None, symbol, "Unknown"))
        company, sector = meta[1], meta[2]
        if t.get("universe"):
            uni_label = t["universe"]
        elif symbol in NIFTY50_STOCKS:
            uni_label = "Nifty 50"
        elif symbol in LARGECAP_NEXT50_STOCKS:
            uni_label = "Largecap+"
        elif symbol in MIDCAP_STOCKS:
            uni_label = "Midcap"
        else:
            uni_label = "Nifty 50"

        row = {
            "Symbol": symbol,
            "Company": company,
            "Sector": sector,
            "Universe": uni_label,
            "Current": current,
            "Mean Target": target_mean,
            "High Target": target_high,
            "Upside %": round(upside * 100, 2),
            "Horizon Target": round(horizon_target, 2),
            "Analysts": num_analysts,
            "Rec": (t.get("recommendationKey") or "").replace("_", " ").title() or "—",
            "Tech Score": round(tech_score, 0),
            "Fair Exit": round(fair_exit, 2),
            "Conviction": conviction,
            "Conviction Score": c_score,
            "ETA": eta_bucket,
            "ETA (months)": eta_m,
            "Timeline": timeline_str,
            "ATH": round(ath, 2),
            "Drawdown %": round(drawdown_pct, 2),
            "Flags": ", ".join(flags) if flags else "",
        }
        rows.append(row)
        detail_cache[symbol] = {"tech": tech, "targets": t}

    if not rows:
        st.info("No stocks passed the minimum analyst-coverage bar.")
        return

    df = pd.DataFrame(rows)

    # --- Apply chip filters + sidebar min-bars ---
    filtered = df.copy()
    filtered = filtered[
        (filtered["Upside %"] >= min_upside * 100)
        & (filtered["Analysts"] >= min_analysts)
        & (filtered["Tech Score"] >= min_tech_score)
    ]
    if conviction_pick:
        filtered = filtered[filtered["Conviction"].isin(conviction_pick)]
    # Horizon chip: only keep stocks whose ETA ≤ selected horizon
    horizon_m_cap = HORIZON_MONTHS.get(horizon, 12)
    filtered = filtered[filtered["ETA (months)"] <= horizon_m_cap + 0.01]
    if drawdown_pick:
        filtered = filtered[filtered["Drawdown %"] >= DRAWDOWN_CHIPS[drawdown_pick]]

    filtered = filtered.sort_values("Conviction Score", ascending=False)

    # --- Active filter summary ---
    active_pills = [f'<span class="pill">Universe: {universe}</span>',
                    f'<span class="pill">Horizon ≤ {horizon}</span>']
    if conviction_pick:
        active_pills.append(f'<span class="pill">Conviction: {", ".join(conviction_pick)}</span>')
    if drawdown_pick:
        active_pills.append(f'<span class="pill">ATH drop {drawdown_pick}</span>')
    st.markdown(
        f'<div class="active-filters">{" ".join(active_pills)}'
        f'<span style="margin-left:8px;">· {len(filtered)} of {len(df)} names match</span></div>',
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.info("No candidates match these chips — relax the universe, horizon, or ATH drop.")
        return

    # --- Top Picks card grid (3 highest conviction scores) ---
    st.subheader("Top picks")
    top3 = filtered.head(3)
    cards_html = '<div class="top-picks-grid">'
    for _, r in top3.iterrows():
        conv_badge = _conviction_badge(r["Conviction"])
        eta_badge = _horizon_badge(r["ETA"])
        upside_color = "#00d09c" if r["Upside %"] >= 0 else "#ff6b6b"
        cards_html += (
            f'<div class="top-pick-card">'
            f'  <div style="display:flex;justify-content:space-between;align-items:center;">'
            f'    <div>'
            f'      <div class="tp-symbol">{r["Symbol"]}</div>'
            f'      <div class="tp-company">{r["Company"]} · {r["Sector"]}</div>'
            f'    </div>'
            f'    <div style="display:flex;gap:4px;">{conv_badge}{eta_badge}</div>'
            f'  </div>'
            f'  <div class="tp-upside" style="color:{upside_color};">{r["Upside %"]:+.1f}%</div>'
            f'  <div class="tp-meta">Now ₹{r["Current"]:,.0f} → Target ₹{r["Mean Target"]:,.0f}</div>'
            f'  <div class="tp-meta">{horizon} view: ₹{r["Horizon Target"]:,.0f} · '
            f'    {int(r["Analysts"])} analysts · {r["Drawdown %"]:.0f}% off ATH</div>'
            f'</div>'
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    # --- Main table with colored conviction + ETA badges (HTML render) ---
    st.subheader("All candidates")
    table_html = _build_table_html(filtered, horizon)
    st.markdown(table_html, unsafe_allow_html=True)

    st.divider()
    st.subheader("Deep dive")

    for symbol in filtered["Symbol"].tolist():
        d = detail_cache[symbol]
        t = d["targets"]
        tech = d["tech"]
        row = filtered[filtered["Symbol"] == symbol].iloc[0]

        header = (
            f"**{symbol}** — {row['Company']} ・ {row['Conviction']} conviction ・ ETA {row['ETA']} ・ "
            f"Upside {row['Upside %']:+.1f}% ・ {row['Drawdown %']:.0f}% off ATH ・ Fair Exit ₹{row['Fair Exit']:,.2f}"
        )
        with st.expander(header):
            c1, c2 = st.columns([3, 2])

            with c1:
                # Recent broker revisions
                st.markdown("**Recent broker revisions (last 90 days)**")
                revisions = analyst.revisions.get(symbol)
                if revisions is not None and not revisions.empty:
                    rev_df = revisions.reset_index()
                    rev_df.columns = [str(c).replace("_", " ").title() for c in rev_df.columns]
                    st.dataframe(rev_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No recent revisions in the last 90 days.")

                # Analyst target spread + ATH context
                st.markdown("**Analyst target spread**")
                spread_df = pd.DataFrame({
                    "Metric": [
                        "5y ATH", "Current", "Low", "Mean", "Median", "High",
                        f"{st.session_state.get('th_horizon', '12M')} Interim", "Fair Exit (ours)",
                    ],
                    "Price": [
                        t.get("ath") or 0,
                        t.get("currentPrice") or 0,
                        t.get("targetLowPrice") or 0,
                        t.get("targetMeanPrice") or 0,
                        t.get("targetMedianPrice") or 0,
                        t.get("targetHighPrice") or 0,
                        row["Horizon Target"],
                        row["Fair Exit"],
                    ],
                })
                st.dataframe(spread_df, use_container_width=True, hide_index=True,
                             column_config={"Price": st.column_config.NumberColumn(format="%.2f")})

            with c2:
                st.markdown("**Technical breakdown**")
                if tech:
                    st.metric("RSI (14)", f"{tech['rsi']:.1f}")
                    st.metric("Price vs SMA50",
                              f"{(tech['close']/tech['sma50']-1)*100:+.1f}%" if tech['sma50'] else "—")
                    st.metric("Price vs SMA200",
                              f"{(tech['close']/tech['sma200']-1)*100:+.1f}%" if tech['sma200'] else "—")
                    st.metric("% of 52W range", f"{tech['pct_of_52w']:.0f}%")
                    st.caption(
                        f"MACD {'bullish' if tech['signals']['macd_bullish'] else 'bearish'} ・ "
                        f"Volume {tech['vol_ratio']:.2f}× 20d avg"
                    )
                else:
                    st.caption("Technical data unavailable.")

                st.markdown("**Timeline & flags**")
                st.caption(f"Expected: **{row['Timeline']}**")
                if row["Flags"]:
                    st.caption(f":orange[Risk: {row['Flags']}]")

                # Sector supply chain context
                sc = SECTOR_SUPPLY_CHAIN.get(row["Sector"])
                if sc:
                    st.markdown("**Macro drivers**")
                    st.caption(f"Linked to: {', '.join(sc['factors'])}")
                    st.caption(sc["note"])

    # Coverage footer
    st.caption(
        f"Analyst snapshot from {analyst.timestamp.strftime('%I:%M %p IST')} ・ "
        f"{len(analyst.targets)} stocks covered, {len(analyst.failed)} without analyst data."
    )
