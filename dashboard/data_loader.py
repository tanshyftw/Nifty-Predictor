"""
Dashboard data fetching layer.
Wraps existing fetchers (YahooFetcher, GlobalFetcher, NSEFetcher) and adds
new fetching for sectoral indices and supply chain factors.
All functions use @st.cache_data for smart caching.
"""

from __future__ import annotations

import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from loguru import logger

# Add project root to path so we can import existing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.nifty50_tickers import NIFTY50_STOCKS, GLOBAL_INDICES, get_yahoo_tickers
from config.holidays import is_trading_day, prev_trading_day, days_to_expiry, next_fno_expiry
from data.sources.yahoo_fetcher import YahooFetcher
from data.sources.nse_fetcher import NSEFetcher
from dashboard.universe import (
    EXPANDED_UNIVERSE,
    get_universe_yahoo_tickers,
)
from dashboard.config import (
    SECTORAL_INDICES,
    SUPPLY_CHAIN_TICKERS,
    CACHE_TTL_SECONDS,
    TOP_MOVERS_COUNT,
    PERIOD_DAILY,
    PERIOD_WEEKLY,
)
from dashboard.disk_cache import disk_cached
from dashboard.precomputed_cache import (
    load_fii_dii as _load_precomputed_fii_dii,
    load_market_caps as _load_precomputed_market_caps,
    merge_today_into_fii_dii,
)

IST = timezone(timedelta(hours=5, minutes=30))

# NSE regular session (IST)
_MARKET_OPEN = (9, 15)
_MARKET_CLOSE = (15, 30)
MARKET_OVERVIEW_UNIVERSE = NIFTY50_STOCKS
YFINANCE_TIMEOUT_SECONDS = 8
SNAPSHOT_FETCH_DEADLINE_SECONDS = 12
LIVE_QUOTE_DEADLINE_SECONDS = 4


def is_market_open_now(now: datetime | None = None) -> bool:
    """True iff `now` (IST) falls inside the NSE regular trading session."""
    now = now or datetime.now(IST)
    if not is_trading_day(now.date()):
        return False
    hm = (now.hour, now.minute)
    return _MARKET_OPEN <= hm <= _MARKET_CLOSE


@dataclass
class MarketSnapshot:
    """All data needed to render the dashboard."""

    timestamp: datetime
    view: str  # "daily" or "weekly"

    # Nifty 50 index
    nifty50_close: float = 0.0
    nifty50_change_pct: float = 0.0
    nifty50_wow_pct: float = 0.0
    # Anchor used for the WoW number: the close (and its date) from the last
    # trading day strictly before the current calendar week's Monday.
    # Surfaced so the dashboard can render "WoW: +0.4% (vs Fri 25-Apr 23,892)"
    # and users don't have to guess what the comparison is.
    nifty50_wow_anchor_close: float = 0.0
    nifty50_wow_anchor_date: str = ""

    # Sectoral indices DataFrame: columns=[close, dod_pct, wow_pct, mom_pct]
    sectoral_data: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Stock-level data
    stock_changes: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_gainers: list = field(default_factory=list)
    top_losers: list = field(default_factory=list)

    # Macro
    india_vix: float = 0.0
    india_vix_change: float = 0.0
    india_vix_wow_pct: float = 0.0
    india_vix_wow_anchor_close: float = 0.0
    india_vix_wow_anchor_date: str = ""
    usdinr: float = 0.0
    usdinr_change: float = 0.0
    usdinr_wow_pct: float = 0.0
    usdinr_wow_anchor_close: float = 0.0
    usdinr_wow_anchor_date: str = ""
    fii_net_buy: float = 0.0
    dii_net_buy: float = 0.0
    # Week-cycle flow aggregates: cumulative ₹ cr from Monday-of-this-week to
    # the latest available print, plus the same metric for the prior calendar
    # week (used as the WoW anchor on the headline chip).
    fii_wtd: float = 0.0
    dii_wtd: float = 0.0
    fii_prev_week: float = 0.0
    dii_prev_week: float = 0.0
    flow_prev_week_label: str = ""

    # Macro time series for charts
    vix_series: pd.DataFrame = field(default_factory=pd.DataFrame)
    usdinr_series: pd.DataFrame = field(default_factory=pd.DataFrame)
    fii_dii_series: list = field(default_factory=list)

    # Global indices: {name: {close, ret_pct}}
    global_indices: dict = field(default_factory=dict)

    # Supply chain factors DataFrame: columns=[close, dod_pct, wow_pct]
    supply_chain: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Market status
    is_market_open: bool = False
    last_trading_date: str = ""

    # Breadth
    advance_count: int = 0
    decline_count: int = 0
    unchanged_count: int = 0


def _week_pct_change_full(closes: pd.Series, dates=None) -> tuple[float, float, str]:
    """Return (wow_pct, anchor_close, anchor_date_iso) for the current week.

    Anchor = close on the last trading day strictly before this calendar week's
    Monday. On a Friday close this is the prior Friday. Mid-week it's still the
    prior Friday — which is exactly the "last Friday vs latest" semantics the
    dashboard surfaces as WoW.

    `dates` may be supplied when `closes.index` isn't a DatetimeIndex (e.g. the
    YahooFetcher reset_index() flow). If no calendar information is available,
    falls back to a 5-trading-day rolling anchor so callers never raise; in
    that fallback `anchor_date_iso` is "".
    """
    if len(closes) < 2:
        return 0.0, 0.0, ""
    latest = float(closes.iloc[-1])
    if not latest:
        return 0.0, 0.0, ""

    if dates is None:
        if isinstance(closes.index, pd.DatetimeIndex):
            dates_arr = list(closes.index)
        else:
            anchor_close, pct = _rolling_week_anchor(closes)
            return pct, anchor_close, ""
    else:
        try:
            dates_arr = pd.to_datetime(pd.Series(dates).reset_index(drop=True)).tolist()
        except Exception:
            anchor_close, pct = _rolling_week_anchor(closes)
            return pct, anchor_close, ""

    try:
        latest_dt = pd.Timestamp(dates_arr[-1])
        monday = latest_dt.normalize() - pd.Timedelta(days=latest_dt.weekday())
    except Exception:
        anchor_close, pct = _rolling_week_anchor(closes)
        return pct, anchor_close, ""

    anchor_pos = -1
    for i in range(len(dates_arr) - 1, -1, -1):
        if pd.Timestamp(dates_arr[i]) < monday:
            anchor_pos = i
            break

    if anchor_pos < 0:
        anchor_close, pct = _rolling_week_anchor(closes)
        return pct, anchor_close, ""

    anchor = float(closes.iloc[anchor_pos])
    if not anchor:
        return 0.0, anchor, ""
    pct = ((latest - anchor) / anchor) * 100
    anchor_date = pd.Timestamp(dates_arr[anchor_pos]).date().isoformat()
    return pct, anchor, anchor_date


def _week_pct_change(closes: pd.Series, dates=None) -> float:
    """Percent change vs the prior week's last close. See _week_pct_change_full."""
    pct, _, _ = _week_pct_change_full(closes, dates)
    return pct


def _rolling_week_anchor(closes: pd.Series) -> tuple[float, float]:
    """Fallback: (anchor_close, pct) using a 5-trading-day rolling delta."""
    if len(closes) < 6:
        return 0.0, 0.0
    anchor = float(closes.iloc[-6])
    latest = float(closes.iloc[-1])
    pct = ((latest - anchor) / anchor) * 100 if anchor else 0.0
    return anchor, pct


def _market_minute_bucket() -> str:
    """Cache-key bucket that drives cache invalidation across the dashboard.

    - During NSE market hours (9:15–15:30 IST on trading days): rotates every
      minute, so cached snapshots stay <=60s behind the live tape.
    - Outside market hours (after-hours, weekends, holidays): keyed on the
      most recent trading day plus an AM/PM marker. This guarantees a fresh
      fetch the moment a new session starts, so a viewer on Saturday cannot
      see Friday's last intraday tick masquerading as the close — the bucket
      transitions from `live-...` to `closed-2026-04-24-pm` immediately at
      15:30 IST and forces a re-fetch that picks up the settled close.
    """
    now = datetime.now(IST)
    if is_market_open_now(now):
        return f"live-{now:%Y%m%d-%H%M}"
    today = now.date()
    last_session = today if is_trading_day(today) else prev_trading_day(today)
    half = "am" if now.hour < 12 else "pm"
    return f"closed-{last_session.isoformat()}-{half}"


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_live_quotes_cached(_bucket: str = "") -> dict[str, dict]:
    """Live-quote overlay used to compute DoD across the dashboard.

    Yahoo's daily candle for Indian tickers commonly lags by a full session
    (i.e. `yf.download(period="1mo", interval="1d")` may not include today's
    close until well after market close). That makes `iloc[-1] / iloc[-2]`
    surface yesterday's DoD as if it were today's. fast_info hits the live
    quote endpoint which stays current through the session, so we use it as
    the source of truth for "latest price" and "previous-day close".

    Universe: Nifty 50 + Nifty 50 index + India VIX + USD/INR. Cached on the
    same minute bucket as load_market_snapshot. Broader scans are loaded only
    on demand so the default overview stays responsive.
    """
    del _bucket

    fetcher = YahooFetcher(max_retries=1, retry_delay=0.5)

    yt_list: list[str] = []
    yt_to_sym: dict[str, str] = {}
    for sym, (yt, _, _) in MARKET_OVERVIEW_UNIVERSE.items():
        if yt not in yt_to_sym:
            yt_list.append(yt)
            yt_to_sym[yt] = sym
    # Indices the chips care about
    for yt in ("^NSEI", "^INDIAVIX", "INR=X"):
        yt_list.append(yt)
        yt_to_sym[yt] = yt

    raw = fetcher.fetch_live_quotes(yt_list)

    # Re-key by NSE symbol where possible (and keep raw yt keys for indices)
    out: dict[str, dict] = {}
    for yt, q in raw.items():
        out[yt] = q
        sym = yt_to_sym.get(yt)
        if sym and sym != yt:
            out[sym] = q
    return out


def _apply_live_quote(
    closes: pd.Series, quote: dict | None
) -> tuple[float, float, bool]:
    """Return (latest, prev_close, used_live).

    If a live quote is available, use it; otherwise fall back to the last
    two non-null entries in the historical close series.
    """
    if quote and quote.get("last_price") and quote.get("previous_close"):
        return float(quote["last_price"]), float(quote["previous_close"]), True
    if len(closes) >= 2:
        return float(closes.iloc[-1]), float(closes.iloc[-2]), False
    return 0.0, 0.0, False


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Fetching live market data...")
def load_market_snapshot(view: str = "daily", _bucket: str = "") -> MarketSnapshot:
    """Master function: fetches all data and returns a MarketSnapshot.

    The `_bucket` argument is a cache-key only — pass `_market_minute_bucket()`
    so the cache rotates every minute during market hours and forces a fresh
    fetch on every session boundary (so Friday's last intraday tick can never
    leak into Saturday's view).
    """
    now = datetime.now(IST)
    today = date.today()
    trading_day = today if is_trading_day(today) else prev_trading_day(today)

    # Always fetch 1mo to have enough data for WoW/MoM computations
    period = "1mo"

    snapshot = MarketSnapshot(
        timestamp=now,
        view=view,
        is_market_open=is_market_open_now(now),
        last_trading_date=trading_day.isoformat(),
    )

    # The live quote overlay is only worth the extra per-symbol fast_info fanout
    # during market hours. Off-hours/weekends should use daily candles so cold
    # starts do not wait on 50 quote endpoints before rendering.
    live_quotes = (
        _fetch_live_quotes_bounded(_bucket)
        if snapshot.is_market_open
        else {}
    )

    # Critical path: universe (top movers, breadth, heatmap) + macro (key
    # metrics chips, FII/DII). Both are independent network calls; run them
    # in parallel. yfinance's internal threads=True is left on inside each
    # batch — the two outer threads are a tiny multiplier compared with the
    # per-ticker fan-out yfinance already does, and in testing Yahoo did not
    # rate-limit at this concurrency.
    _populate_snapshot_bounded(snapshot, period, live_quotes)

    # Sectoral and supply chain are NOT loaded here — they're only needed by
    # the Sectors and Macro & Global tabs. Lazy-loading them via
    # `attach_sectoral_data` / `attach_supply_chain_data` in app.py shaves
    # ~1s off Market Overview cold start.
    return snapshot


def _fetch_live_quotes_bounded(bucket: str) -> dict[str, dict]:
    """Fetch live quotes without letting fast_info block first paint."""
    ex = ThreadPoolExecutor(max_workers=1)
    future = ex.submit(_fetch_live_quotes_cached, _bucket=bucket)
    try:
        return future.result(timeout=LIVE_QUOTE_DEADLINE_SECONDS)
    except TimeoutError:
        logger.warning("Live quote overlay timed out; using historical closes")
        return {}
    except Exception as e:
        logger.warning(f"Live quote overlay failed: {e}")
        return {}
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _populate_snapshot_bounded(
    snapshot: MarketSnapshot,
    period: str,
    live_quotes: dict[str, dict],
):
    """Populate critical snapshot data under a fixed first-paint deadline."""
    ex = ThreadPoolExecutor(max_workers=2)
    futures = {
        "stocks": ex.submit(_populate_stock_data, snapshot, period, live_quotes),
        "macro": ex.submit(_populate_macro_data, snapshot, period, live_quotes),
    }
    deadline = time.monotonic() + SNAPSHOT_FETCH_DEADLINE_SECONDS
    try:
        for name, future in futures.items():
            remaining = max(0.1, deadline - time.monotonic())
            try:
                future.result(timeout=remaining)
            except TimeoutError:
                logger.warning(f"{name} fetch timed out; rendering partial snapshot")
            except Exception as e:
                logger.warning(f"{name} fetch failed: {e}")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def attach_sectoral_data(snapshot: MarketSnapshot, period: str = "1mo") -> MarketSnapshot:
    """Lazy-load sectoral indices and attach onto an existing snapshot.

    Called from app.py only when the user opens a tab that actually renders
    sectoral data (Sectors tab, plus the cross-sector callouts inside the
    Macro & Global tab). Cached per minute-bucket so repeated tab opens are
    free.
    """
    if not snapshot.sectoral_data.empty:
        return snapshot
    df = _fetch_sectoral_data_cached(_bucket=_market_minute_bucket(), period=period)
    if df is not None and not df.empty:
        snapshot.sectoral_data = df
    return snapshot


def attach_supply_chain_data(snapshot: MarketSnapshot, period: str = "1mo") -> MarketSnapshot:
    """Lazy-load supply chain / commodity factors. See attach_sectoral_data."""
    if not snapshot.supply_chain.empty:
        return snapshot
    df = _fetch_supply_chain_cached(_bucket=_market_minute_bucket(), period=period)
    if df is not None and not df.empty:
        snapshot.supply_chain = df
    return snapshot


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_sectoral_data_cached(_bucket: str = "", period: str = "1mo") -> pd.DataFrame:
    """Cached wrapper around the existing sectoral populate logic."""
    del _bucket
    sink = MarketSnapshot(timestamp=datetime.now(IST), view="daily")
    _populate_sectoral_data(sink, period)
    return sink.sectoral_data


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_supply_chain_cached(_bucket: str = "", period: str = "1mo") -> pd.DataFrame:
    """Cached wrapper around the existing supply chain populate logic."""
    del _bucket
    sink = MarketSnapshot(timestamp=datetime.now(IST), view="daily")
    _populate_supply_chain(sink, period)
    return sink.supply_chain


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_macro_batch_cached(_bucket: str = "") -> dict[str, pd.Series]:
    """Single batched download for every macro/index ticker.

    Replaces the previous 9-call sequential storm (Nifty, VIX, USD/INR,
    S&P 500, NASDAQ, Dow, FTSE, Nikkei, Hang Seng — each its own yfinance
    request). One batched call collapses that to roughly one HTTP round
    trip, which is the single biggest contributor to dashboard cold-start
    latency.

    Returns `{ticker: close_series}` keyed by Yahoo ticker. Tickers that
    yfinance silently drops on the batch (rare for indices, but possible)
    fall back to a one-off `_fetch_single_ticker` so the dashboard never
    loses a headline KPI to a flaky batch response.
    """
    del _bucket
    tickers = ["^NSEI", "^INDIAVIX", "INR=X"]
    for key in ("SP500", "NASDAQ", "DOW", "FTSE", "NIKKEI", "HANGSENG"):
        t = GLOBAL_INDICES.get(key)
        if t:
            tickers.append(t)

    out: dict[str, pd.Series] = {}
    try:
        data = yf.download(
            " ".join(tickers),
            period="1mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
            timeout=YFINANCE_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logger.warning(f"macro batch fetch failed: {e}")
        data = pd.DataFrame()

    if not data.empty:
        for tkr in tickers:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    level_0 = data.columns.get_level_values(0).unique().tolist()
                    level_1 = data.columns.get_level_values(1).unique().tolist()
                    if tkr in level_0:
                        close = data[tkr]["Close"]
                    elif tkr in level_1:
                        sub = data.xs(tkr, level=1, axis=1)
                        close = sub["Close"] if "Close" in sub.columns else sub.iloc[:, 0]
                    else:
                        continue
                else:
                    close = data["Close"]
                close = close.dropna()
                if len(close) >= 2:
                    out[tkr] = close
            except Exception:
                continue

    # Per-ticker fallback for anything the batch dropped
    for tkr in tickers:
        if tkr not in out:
            series = _fetch_single_ticker(tkr, "1mo")
            if len(series) >= 2:
                out[tkr] = series

    return out


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_universe_ohlcv(period: str = "1mo") -> dict[str, pd.DataFrame]:
    """Batch-download daily OHLCV for the default Market Overview universe.

    Keep this to the Nifty 50. The broader Largecap+/Midcap scans are heavier
    and now live behind on-demand tiles or secondary views.
    """
    yt_to_sym = {info[0]: sym for sym, info in MARKET_OVERVIEW_UNIVERSE.items()}
    tickers_str = " ".join(yt_to_sym.keys())

    try:
        data = yf.download(
            tickers_str,
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
            timeout=YFINANCE_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logger.warning(f"Universe batch fetch failed: {e}")
        return {}

    if data is None or data.empty:
        return {}

    out: dict[str, pd.DataFrame] = {}
    for yt, sym in yt_to_sym.items():
        try:
            if isinstance(data.columns, pd.MultiIndex):
                level_0 = data.columns.get_level_values(0).unique().tolist()
                level_1 = data.columns.get_level_values(1).unique().tolist()
                if yt in level_0:
                    sub = data[yt].copy()
                elif yt in level_1:
                    sub = data.xs(yt, level=1, axis=1).copy()
                else:
                    continue
            else:
                sub = data.copy()

            sub.columns = [str(c).lower() for c in sub.columns]
            if "close" not in sub.columns:
                continue

            sub = sub.reset_index()
            sub.columns = [str(c).lower() for c in sub.columns]
            sub = sub.dropna(subset=["close"])
            if sub.empty:
                continue
            out[sym] = sub
        except Exception:
            continue

    return out


def _populate_stock_data(snapshot: MarketSnapshot, period: str, live_quotes: dict | None = None):
    """Fetch default Market Overview stock prices and compute movers."""
    live_quotes = live_quotes or {}
    try:
        stock_data = _fetch_universe_ohlcv(period=period)

        if not stock_data:
            logger.warning("No stock data fetched")
            return

        records = []
        for symbol, df in stock_data.items():
            if df is None or df.empty:
                continue

            close_col = "close" if "close" in df.columns else None
            vol_col = "volume" if "volume" in df.columns else None
            if close_col is None:
                continue

            closes = df[close_col].dropna()
            if len(closes) < 2:
                continue

            latest_close, prev_close, _ = _apply_live_quote(closes, live_quotes.get(symbol))
            dod_pct = ((latest_close - prev_close) / prev_close) * 100 if prev_close else 0

            # WoW + MoM: anchor pulled from historical, latest taken from the
            # live quote (above). Splitting the two keeps WoW correct even
            # when Yahoo's day-of candle hasn't rolled forward.
            date_col = "date" if "date" in df.columns else None
            dates_for_close = df.loc[closes.index, date_col] if date_col else None
            _, wow_anchor, _ = _week_pct_change_full(closes, dates_for_close)
            wow_pct = ((latest_close - wow_anchor) / wow_anchor) * 100 if wow_anchor else 0.0

            mom_pct = 0.0
            if len(closes) >= 15:
                month_ago_close = float(closes.iloc[0])
                mom_pct = ((latest_close - month_ago_close) / month_ago_close) * 100 if month_ago_close else 0

            volume = int(df[vol_col].iloc[-1]) if vol_col and pd.notna(df[vol_col].iloc[-1]) else 0

            meta = MARKET_OVERVIEW_UNIVERSE.get(symbol, (None, symbol, "Unknown"))
            company = meta[1]
            sector = meta[2]

            records.append({
                "symbol": symbol,
                "company": company,
                "sector": sector,
                "close": latest_close,
                "dod_pct": round(dod_pct, 2),
                "wow_pct": round(wow_pct, 2),
                "mom_pct": round(mom_pct, 2),
                "volume": volume,
            })

        if not records:
            return

        df_all = pd.DataFrame(records)
        snapshot.stock_changes = df_all

        # Determine sort column based on view
        sort_col = "dod_pct" if snapshot.view == "daily" else "wow_pct"

        sorted_df = df_all.sort_values(sort_col, ascending=False)
        snapshot.top_gainers = sorted_df.head(TOP_MOVERS_COUNT).to_dict("records")
        snapshot.top_losers = sorted_df.tail(TOP_MOVERS_COUNT).sort_values(sort_col).to_dict("records")

        # Breadth
        change_col = sort_col
        snapshot.advance_count = int((df_all[change_col] > 0.1).sum())
        snapshot.decline_count = int((df_all[change_col] < -0.1).sum())
        snapshot.unchanged_count = len(df_all) - snapshot.advance_count - snapshot.decline_count

    except Exception as e:
        logger.error(f"Stock data fetch failed: {e}")


def _populate_macro_data(snapshot: MarketSnapshot, period: str, live_quotes: dict | None = None):
    """Fetch global indices, VIX, USD/INR, FII/DII.

    All index closes are pulled from a single batched yfinance call
    (`_fetch_macro_batch_cached`) rather than the historical pattern of
    one yfinance request per ticker. The per-ticker fallback inside the
    batch helper still kicks in if Yahoo silently drops an Indian index
    on a given batch, so headline KPIs degrade gracefully rather than
    going blank.
    """
    live_quotes = live_quotes or {}
    closes_by_ticker = _fetch_macro_batch_cached(_bucket=_market_minute_bucket())

    _apply_nifty50(snapshot, closes_by_ticker.get("^NSEI"), live_quotes.get("^NSEI"))
    _apply_india_vix(snapshot, closes_by_ticker.get("^INDIAVIX"), live_quotes.get("^INDIAVIX"))
    _apply_usdinr(snapshot, closes_by_ticker.get("INR=X"), live_quotes.get("INR=X"))

    # --- Global indices ---
    from dashboard.config import GLOBAL_INDEX_DISPLAY
    for key, display_name in GLOBAL_INDEX_DISPLAY.items():
        ticker = GLOBAL_INDICES.get(key)
        if not ticker:
            continue
        closes = closes_by_ticker.get(ticker)
        if closes is None or len(closes) < 2:
            continue
        latest = float(closes.iloc[-1])
        prev = float(closes.iloc[-2])
        dod_pct = ((latest - prev) / prev) * 100 if prev else 0.0
        wow_pct = _week_pct_change(closes)
        spark = closes.tail(10).reset_index(drop=True).tolist()
        snapshot.global_indices[display_name] = {
            "close": round(latest, 2),
            "ret_pct": round(dod_pct, 2),
            "wow_pct": round(wow_pct, 2),
            "spark": [float(v) for v in spark],
        }

    # --- FII/DII --- need ~30 days for WoW and MoM aggregates.
    # Fast path: nightly GitHub Action precomputes the full 30-day window and
    # commits it to storage/precomputed/fii_dii.parquet. At runtime we read
    # that file and only fetch today's record live (1 nselib call instead of
    # the 30-day loop). Slow path: parquet missing or too stale → full live
    # fetch, same as before.
    fii_dii = _load_fii_dii_with_today()
    if fii_dii:
        snapshot.fii_dii_series = fii_dii
        latest_fii = fii_dii[-1]
        snapshot.fii_net_buy = latest_fii.get("fii_net_buy", 0.0)
        snapshot.dii_net_buy = latest_fii.get("dii_net_buy", 0.0)
        _populate_flow_week_aggregates(snapshot, fii_dii)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _load_fii_dii_with_today() -> list[dict]:
    """Read the precomputed FII/DII baseline and fold in today's print.

    On cold start in production this costs ~50ms (parquet read) + at most one
    nselib round-trip for the current day. The old path made up to 30 nselib
    calls with retry sleeps, accounting for 2–5s of dashboard latency.

    If the parquet is missing or stale (CI hasn't refreshed it), we fall back
    to a live 30-day lookback so the macro panel still shows flows, chart, and
    WoW/MoM aggregates — cached separately so the lookback only runs once per
    hour rather than on every 60-second page refresh.
    """
    precomputed = _load_precomputed_fii_dii(max_age_days=3)
    today = date.today()
    today_record: dict | None = None

    if is_trading_day(today):
        try:
            nf = NSEFetcher(max_retries=2, retry_delay=1.0)
            today_only = nf.fetch_fii_dii(today.isoformat(), today.isoformat())
            if today_only:
                today_record = today_only[-1]
        except Exception as e:
            logger.debug(f"Today's FII/DII fetch failed (using baseline only): {e}")

    if precomputed:
        return merge_today_into_fii_dii(precomputed, today_record)

    # Cold fallback: no parquet on disk (CI never ran or has been failing).
    # Do the full live lookback so WoW/MoM/chart still work; the result is
    # cached for an hour so this 30-call loop doesn't repeat on every reload.
    baseline = _live_fii_dii_lookback()
    if baseline:
        return merge_today_into_fii_dii(baseline, today_record)
    return [today_record] if today_record else []


@st.cache_data(ttl=3600, show_spinner=False)
def _live_fii_dii_lookback() -> list[dict]:
    """30-day live FII/DII fetch used when the precomputed parquet is absent."""
    try:
        nf = NSEFetcher(max_retries=2, retry_delay=1.0)
        return nf.fetch_recent_fii_dii(lookback_days=30) or []
    except Exception as e:
        logger.warning(f"FII/DII live fallback failed: {e}")
        return []


def _populate_flow_week_aggregates(snapshot: MarketSnapshot, fii_dii: list[dict]):
    """Compute week-to-date and prior-week net flow sums for FII and DII.

    Anchor weeks on calendar Mondays so the WoW comparison on the FII chip
    matches every other WoW number on the dashboard:
      - WTD = sum of net flows from this week's Monday up to the latest print
      - prior week = sum across the prior calendar week (Mon..Sun)
    """
    if not fii_dii:
        return
    rows = []
    for r in fii_dii:
        d = r.get("date")
        try:
            ts = pd.Timestamp(d)
        except Exception:
            continue
        rows.append((
            ts.normalize(),
            float(r.get("fii_net_buy") or 0.0),
            float(r.get("dii_net_buy") or 0.0),
        ))
    if not rows:
        return

    rows.sort(key=lambda x: x[0])
    latest_dt = rows[-1][0]
    monday_this = latest_dt - pd.Timedelta(days=latest_dt.weekday())
    monday_prev = monday_this - pd.Timedelta(days=7)

    fii_wtd = sum(f for d, f, _ in rows if d >= monday_this)
    dii_wtd = sum(di for d, _, di in rows if d >= monday_this)
    fii_prev = sum(f for d, f, _ in rows if monday_prev <= d < monday_this)
    dii_prev = sum(di for d, _, di in rows if monday_prev <= d < monday_this)

    snapshot.fii_wtd = fii_wtd
    snapshot.dii_wtd = dii_wtd
    snapshot.fii_prev_week = fii_prev
    snapshot.dii_prev_week = dii_prev
    snapshot.flow_prev_week_label = (
        f"{monday_prev.strftime('%d-%b')}–{(monday_this - pd.Timedelta(days=1)).strftime('%d-%b')}"
    )


def _fetch_single_ticker(ticker: str, period: str) -> pd.Series:
    """Fetch close series for a single ticker. Returns empty Series on failure."""
    try:
        data = yf.download(
            ticker, period=period, interval="1d",
            auto_adjust=True, progress=False, threads=False,
            timeout=YFINANCE_TIMEOUT_SECONDS,
        )
        if data.empty:
            return pd.Series(dtype=float)
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.dropna()
    except Exception as e:
        logger.warning(f"Single-ticker fetch failed for {ticker}: {e}")
        return pd.Series(dtype=float)


def _apply_nifty50(
    snapshot: MarketSnapshot,
    closes: pd.Series | None,
    live_quote: dict | None = None,
):
    """Set Nifty 50 fields on the snapshot from pre-fetched close history."""
    if closes is None or len(closes) < 2:
        return

    latest, prev, _ = _apply_live_quote(closes, live_quote)
    snapshot.nifty50_close = latest
    snapshot.nifty50_change_pct = ((latest - prev) / prev) * 100 if prev else 0

    _, anchor_close, anchor_date = _week_pct_change_full(closes)
    snapshot.nifty50_wow_pct = ((latest - anchor_close) / anchor_close) * 100 if anchor_close else 0.0
    snapshot.nifty50_wow_anchor_close = anchor_close
    snapshot.nifty50_wow_anchor_date = anchor_date


def _apply_india_vix(
    snapshot: MarketSnapshot,
    closes: pd.Series | None,
    live_quote: dict | None = None,
):
    """Set India VIX fields on the snapshot from pre-fetched close history."""
    if closes is None or len(closes) < 2:
        return

    latest, prev, _ = _apply_live_quote(closes, live_quote)
    snapshot.india_vix = latest
    if prev:
        snapshot.india_vix_change = ((latest - prev) / prev) * 100

    _, anchor_close, anchor_date = _week_pct_change_full(closes)
    snapshot.india_vix_wow_pct = ((latest - anchor_close) / anchor_close) * 100 if anchor_close else 0.0
    snapshot.india_vix_wow_anchor_close = anchor_close
    snapshot.india_vix_wow_anchor_date = anchor_date

    vix_data = [{"date": str(d.date()) if hasattr(d, "date") else str(d)[:10],
                 "India VIX": float(v)} for d, v in closes.items()]
    snapshot.vix_series = pd.DataFrame(vix_data).set_index("date") if vix_data else pd.DataFrame()


def _apply_usdinr(
    snapshot: MarketSnapshot,
    closes: pd.Series | None,
    live_quote: dict | None = None,
):
    """Set USD/INR fields on the snapshot from pre-fetched close history."""
    if closes is None or len(closes) < 2:
        return

    latest, prev, _ = _apply_live_quote(closes, live_quote)
    snapshot.usdinr = latest
    if prev:
        snapshot.usdinr_change = ((latest - prev) / prev) * 100

    _, anchor_close, anchor_date = _week_pct_change_full(closes)
    snapshot.usdinr_wow_pct = ((latest - anchor_close) / anchor_close) * 100 if anchor_close else 0.0
    snapshot.usdinr_wow_anchor_close = anchor_close
    snapshot.usdinr_wow_anchor_date = anchor_date

    usdinr_data = [{"date": str(d.date()) if hasattr(d, "date") else str(d)[:10],
                    "USD/INR": float(v)} for d, v in closes.items()]
    snapshot.usdinr_series = pd.DataFrame(usdinr_data).set_index("date") if usdinr_data else pd.DataFrame()


def _populate_sectoral_data(snapshot: MarketSnapshot, period: str):
    """Fetch sectoral index data."""
    try:
        tickers = list(SECTORAL_INDICES.values())
        ticker_str = " ".join(tickers)

        data = yf.download(
            ticker_str,
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
            timeout=YFINANCE_TIMEOUT_SECONDS,
        )

        if data.empty:
            logger.warning("No sectoral index data fetched")
            return

        records = []
        for name, ticker in SECTORAL_INDICES.items():
            try:
                if len(SECTORAL_INDICES) == 1:
                    close_series = data["Close"]
                else:
                    # Handle both MultiIndex formats
                    if isinstance(data.columns, pd.MultiIndex):
                        level_0 = data.columns.get_level_values(0).unique().tolist()
                        level_1 = data.columns.get_level_values(1).unique().tolist()
                        if ticker in level_0:
                            close_series = data[ticker]["Close"]
                        elif ticker in level_1:
                            close_series = data.xs(ticker, level=1, axis=1)["Close"] if "Close" in data.xs(ticker, level=1, axis=1).columns else data.xs(ticker, level=1, axis=1).iloc[:, 0]
                        else:
                            continue
                    else:
                        close_series = data["Close"]

                closes = close_series.dropna()
                if len(closes) < 2:
                    continue

                latest = float(closes.iloc[-1])
                prev = float(closes.iloc[-2])
                dod = ((latest - prev) / prev) * 100 if prev else 0

                wow = _week_pct_change(closes)

                mom = 0.0
                if len(closes) >= 15:
                    m = float(closes.iloc[0])
                    mom = ((latest - m) / m) * 100 if m else 0

                records.append({
                    "Index": name,
                    "Close": round(latest, 2),
                    "DoD %": round(dod, 2),
                    "WoW %": round(wow, 2),
                    "1M %": round(mom, 2),
                })
            except Exception as e:
                logger.debug(f"Sectoral index {name} ({ticker}) failed: {e}")
                continue

        if records:
            snapshot.sectoral_data = pd.DataFrame(records)

    except Exception as e:
        logger.error(f"Sectoral data fetch failed: {e}")


def _populate_supply_chain(snapshot: MarketSnapshot, period: str):
    """Fetch supply chain / international factor data."""
    try:
        tickers = list(SUPPLY_CHAIN_TICKERS.values())
        ticker_str = " ".join(tickers)

        data = yf.download(
            ticker_str,
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
            timeout=YFINANCE_TIMEOUT_SECONDS,
        )

        if data.empty:
            logger.warning("No supply chain data fetched")
            return

        records = []
        for name, ticker in SUPPLY_CHAIN_TICKERS.items():
            try:
                if len(SUPPLY_CHAIN_TICKERS) == 1:
                    close_series = data["Close"]
                else:
                    if isinstance(data.columns, pd.MultiIndex):
                        level_0 = data.columns.get_level_values(0).unique().tolist()
                        level_1 = data.columns.get_level_values(1).unique().tolist()
                        if ticker in level_0:
                            close_series = data[ticker]["Close"]
                        elif ticker in level_1:
                            close_series = data.xs(ticker, level=1, axis=1)["Close"] if "Close" in data.xs(ticker, level=1, axis=1).columns else data.xs(ticker, level=1, axis=1).iloc[:, 0]
                        else:
                            continue
                    else:
                        close_series = data["Close"]

                closes = close_series.dropna()
                if len(closes) < 2:
                    continue

                latest = float(closes.iloc[-1])
                prev = float(closes.iloc[-2])
                dod = ((latest - prev) / prev) * 100 if prev else 0

                wow = _week_pct_change(closes)

                records.append({
                    "Factor": name,
                    "Price": round(latest, 2),
                    "DoD %": round(dod, 2),
                    "WoW %": round(wow, 2),
                })
            except Exception as e:
                logger.debug(f"Supply chain {name} ({ticker}) failed: {e}")
                continue

        if records:
            snapshot.supply_chain = pd.DataFrame(records)

    except Exception as e:
        logger.error(f"Supply chain data fetch failed: {e}")


# ---------------------------------------------------------------------------
# Heatmap universe loader (Nifty 50 + Midcap, sized by market cap)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
@disk_cached(name="market_caps", ttl_hours=24)
def _fetch_market_caps(_day_bucket: str = "") -> dict[str, float]:
    """Fetch market caps for the heatmap universe. Cached for 24h on both
    the in-process Streamlit cache and a pickle-on-disk cache so the value
    survives server restarts (cold start of the dashboard is otherwise
    dominated by 250+ `fast_info` round-trips for sizing the heatmap tiles).

    Market cap is structurally stable intraday — refreshing daily is plenty.
    `fast_info` is preferred (orders of magnitude faster than `.info`); we
    fall back to `.info` only when fast_info doesn't surface a cap.

    Fast path: the nightly GitHub Action commits a parquet of market caps to
    `storage/precomputed/market_caps.parquet`. If present and fresh we use it
    directly and skip the 250-ticker `fast_info` sweep entirely.
    """
    del _day_bucket

    precomputed = _load_precomputed_market_caps(max_age_days=7)
    if precomputed:
        return precomputed

    universe: dict[str, str] = {sym: info[0] for sym, info in EXPANDED_UNIVERSE.items()}

    def _one(item):
        sym, tkr = item
        try:
            t = yf.Ticker(tkr)
            mc = None
            try:
                fi = t.fast_info
                mc = getattr(fi, "market_cap", None) or (fi.get("market_cap") if hasattr(fi, "get") else None)
            except Exception:
                mc = None
            if not mc:
                try:
                    mc = t.info.get("marketCap")
                except Exception:
                    mc = None
            return sym, float(mc) if mc else None
        except Exception:
            return sym, None

    out: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for sym, mc in ex.map(_one, universe.items()):
            if mc and mc > 0:
                out[sym] = mc
    return out


def load_heatmap_data(view: str = "daily", snapshot: "MarketSnapshot | None" = None) -> pd.DataFrame:
    """Return a DataFrame for the heatmap: full expanded universe (Nifty 50 +
    Next 50 + Midcap).

    All numbers come straight from `snapshot.stock_changes` — which now spans
    the expanded universe — so DoD/WoW match the rest of the dashboard exactly.

    Columns: symbol, company, sector, close, dod_pct, wow_pct, volume,
    market_cap, size.
    """
    del view  # only used for display; computation covers both DoD + WoW

    if snapshot is None or snapshot.stock_changes.empty:
        return pd.DataFrame()

    cols = ["symbol", "company", "sector", "close", "dod_pct", "wow_pct", "volume"]
    df = snapshot.stock_changes[cols].copy()
    df["sector"] = df["sector"].fillna("Other")

    today = date.today()
    day_bucket = (today if is_trading_day(today) else prev_trading_day(today)).isoformat()
    mcaps = _fetch_market_caps(_day_bucket=day_bucket)
    df["market_cap"] = df["symbol"].map(mcaps).fillna(0.0)

    # Liquidity fallback so a stock without a fetched cap still gets a tile.
    liquidity = (df["close"].astype(float) * df["volume"].astype(float)).replace(0, 1.0)
    df["size"] = df["market_cap"].where(df["market_cap"] > 0, liquidity)

    return df
