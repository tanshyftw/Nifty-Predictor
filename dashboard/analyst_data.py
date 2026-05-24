"""
Analyst data fetcher for the Target Hunter view.

Pulls analyst price targets, recent broker revisions, and 5-year all-time-high
context from yfinance for every stock in the chosen universe (Nifty 50,
curated Midcap subset, or both). Cached separately from the market snapshot
since analyst data moves much more slowly (hours/days, not minutes).
"""

import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

import pandas as pd
import yfinance as yf
import streamlit as st
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.nifty50_tickers import NIFTY50_STOCKS
from config.midcap_tickers import MIDCAP_STOCKS
from config.largecap_extra_tickers import LARGECAP_NEXT50_STOCKS
from dashboard.config import (
    ANALYST_CACHE_TTL_SECONDS,
    TARGET_REVISION_WINDOW_DAYS,
)

IST = timezone(timedelta(hours=5, minutes=30))

Universe = Literal["Nifty 50", "Largecap+", "Midcap", "Both"]

# yfinance Ticker.info keys we care about
_INFO_KEYS = (
    "currentPrice",
    "targetMeanPrice",
    "targetHighPrice",
    "targetLowPrice",
    "targetMedianPrice",
    "recommendationKey",
    "recommendationMean",
    "numberOfAnalystOpinions",
    "beta",
    "marketCap",
)

_ANALYST_FETCH_WORKERS = 8


@dataclass
class AnalystSnapshot:
    """Per-symbol analyst coverage: targets, recommendations, revisions, ATH."""

    timestamp: datetime
    universe: str = "Nifty 50"
    # per-symbol dict of scalar analyst fields (+ universe label, ATH fields)
    targets: dict = field(default_factory=dict)
    # per-symbol DataFrame of recent revisions
    revisions: dict = field(default_factory=dict)
    # symbols where fetch failed
    failed: list = field(default_factory=list)


def _safe_float(value) -> float:
    try:
        if value is None:
            return 0.0
        f = float(value)
        if f != f:  # NaN guard
            return 0.0
        return f
    except (TypeError, ValueError):
        return 0.0


def _fetch_analyst_targets(ticker: yf.Ticker) -> dict:
    """Pull the analyst-target fields from Ticker.info."""
    try:
        info = ticker.info or {}
    except Exception as e:
        logger.debug(f"Ticker.info failed: {e}")
        return {}

    out = {k: info.get(k) for k in _INFO_KEYS}
    for k in (
        "currentPrice",
        "targetMeanPrice",
        "targetHighPrice",
        "targetLowPrice",
        "targetMedianPrice",
        "recommendationMean",
        "beta",
    ):
        out[k] = _safe_float(out.get(k))
    out["numberOfAnalystOpinions"] = int(out.get("numberOfAnalystOpinions") or 0)
    out["marketCap"] = _safe_float(out.get("marketCap"))
    out["recommendationKey"] = (out.get("recommendationKey") or "").lower()
    return out


def _fetch_recent_revisions(ticker: yf.Ticker, window_days: int) -> pd.DataFrame:
    """Pull recent upgrades/downgrades table, filter to the last window_days."""
    try:
        df = ticker.upgrades_downgrades
    except Exception as e:
        logger.debug(f"upgrades_downgrades failed: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col:
            df.index = pd.to_datetime(df[date_col], errors="coerce")
            df = df.drop(columns=[date_col])

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    if df.empty:
        return pd.DataFrame()

    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=window_days)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df = df[df.index >= cutoff]
    return df.sort_index(ascending=False)


def _fetch_ath(ticker: yf.Ticker) -> dict:
    """Return 5y all-time-high context: ath, ath_date, drawdown_pct."""
    try:
        hist = ticker.history(period="5y", interval="1d", auto_adjust=False)
    except Exception as e:
        logger.debug(f"5y history failed: {e}")
        return {}

    if hist is None or hist.empty or "Close" not in hist.columns:
        return {}

    close = hist["Close"].dropna()
    if close.empty:
        return {}

    last = float(close.iloc[-1])
    ath_idx = close.idxmax()
    ath = float(close.loc[ath_idx])
    ath_date = (
        ath_idx.strftime("%Y-%m-%d")
        if hasattr(ath_idx, "strftime") else str(ath_idx)
    )
    drawdown_pct = (ath - last) / ath * 100 if ath > 0 else 0.0
    return {
        "ath": round(ath, 2),
        "ath_date": ath_date,
        "drawdown_pct": round(drawdown_pct, 2),
    }


def _universe_items(universe: Universe) -> list[tuple[str, tuple[str, str, str], str]]:
    """Return [(symbol, (ticker, company, sector), universe_label)]."""
    items: list[tuple[str, tuple[str, str, str], str]] = []
    seen: set[str] = set()
    if universe in ("Nifty 50", "Both"):
        for sym, meta in NIFTY50_STOCKS.items():
            items.append((sym, meta, "Nifty 50"))
            seen.add(sym)
    if universe in ("Largecap+", "Both"):
        for sym, meta in LARGECAP_NEXT50_STOCKS.items():
            if sym in seen:
                continue
            items.append((sym, meta, "Largecap+"))
            seen.add(sym)
    if universe in ("Midcap", "Both"):
        for sym, meta in MIDCAP_STOCKS.items():
            if sym in seen:
                continue
            items.append((sym, meta, "Midcap"))
            seen.add(sym)
    return items


def _fetch_symbol_snapshot(
    symbol: str,
    yahoo_ticker: str,
    uni_label: str,
) -> tuple[str, dict, pd.DataFrame, bool]:
    """Fetch analyst target, ATH, and revisions for one symbol."""
    t = yf.Ticker(yahoo_ticker)
    targets = _fetch_analyst_targets(t)
    if not targets or targets.get("numberOfAnalystOpinions", 0) == 0:
        return symbol, {}, pd.DataFrame(), False

    targets["universe"] = uni_label
    targets.update(_fetch_ath(t))
    revisions = _fetch_recent_revisions(t, TARGET_REVISION_WINDOW_DAYS)
    return symbol, targets, revisions, True


@st.cache_data(ttl=ANALYST_CACHE_TTL_SECONDS, show_spinner="Fetching analyst targets...")
def load_analyst_snapshot(universe: Universe = "Nifty 50") -> AnalystSnapshot:
    """Fetch analyst targets + revisions + 5y ATH for every stock in the universe."""
    snapshot = AnalystSnapshot(timestamp=datetime.now(IST), universe=universe)

    items = _universe_items(universe)
    max_workers = min(_ANALYST_FETCH_WORKERS, max(1, len(items)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_symbol_snapshot, symbol, yahoo_ticker, uni_label): symbol
            for symbol, (yahoo_ticker, _company, _sector), uni_label in items
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                sym, targets, revisions, covered = future.result()
                if not covered:
                    snapshot.failed.append(sym)
                    continue
                snapshot.targets[sym] = targets
                if not revisions.empty:
                    snapshot.revisions[sym] = revisions
            except Exception as e:
                logger.debug(f"Analyst fetch failed for {symbol}: {e}")
                snapshot.failed.append(symbol)

    logger.info(
        f"Analyst snapshot [{universe}]: {len(snapshot.targets)} covered, "
        f"{len(snapshot.revisions)} with revisions, "
        f"{len(snapshot.failed)} uncovered"
    )
    return snapshot
