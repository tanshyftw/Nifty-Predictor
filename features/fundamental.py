"""
Fundamental analysis feature computation.
Reads fundamental data from the database and computes features including
sector-relative metrics.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from loguru import logger

from config.nifty50_tickers import get_sector, get_symbols


def compute_fundamental_features(
    fund_data: dict | None,
    all_fundamentals: list[dict] | None = None,
) -> dict:
    """
    Compute fundamental features for a single stock.

    Input:
        fund_data: dict from fundamentals table for this stock
        all_fundamentals: list of dicts for all stocks (for sector-relative features)
    Output: dict with 14 fundamental feature values
    """
    if fund_data is None:
        return _default_features()

    features = {
        "trailing_pe": _safe_val(fund_data.get("trailing_pe")),
        "forward_pe": _safe_val(fund_data.get("forward_pe")),
        "price_to_book": _safe_val(fund_data.get("price_to_book")),
        "dividend_yield": _safe_val(fund_data.get("dividend_yield")),
        "market_cap_log": _safe_log(fund_data.get("market_cap")),
        "roe": _safe_val(fund_data.get("roe")),
        "debt_to_equity": _safe_val(fund_data.get("debt_to_equity")),
        "earnings_growth": _safe_val(fund_data.get("earnings_growth")),
        "revenue_growth": _safe_val(fund_data.get("revenue_growth")),
        "profit_margin": _safe_val(fund_data.get("profit_margin")),
    }

    # Sector-relative features (need all_fundamentals)
    if all_fundamentals:
        symbol = fund_data.get("symbol", "")
        sector = fund_data.get("sector", get_sector(symbol))
        features.update(
            _compute_relative_features(fund_data, all_fundamentals, sector)
        )
    else:
        features.update({
            "pe_vs_sector_median": 1.0,
            "pb_vs_sector_median": 1.0,
            "mcap_rank": 25,
            "pe_percentile_1y": 0.5,
        })

    return features


def _compute_relative_features(
    fund_data: dict,
    all_fundamentals: list[dict],
    sector: str,
) -> dict:
    """Compute sector-relative fundamental features."""
    # Filter to same sector
    sector_funds = [
        f for f in all_fundamentals
        if f.get("sector", get_sector(f.get("symbol", ""))) == sector
    ]

    # P/E vs sector median
    sector_pes = [
        f["trailing_pe"] for f in sector_funds
        if f.get("trailing_pe") is not None and f["trailing_pe"] > 0
    ]
    if sector_pes and fund_data.get("trailing_pe") and fund_data["trailing_pe"] > 0:
        median_pe = sorted(sector_pes)[len(sector_pes) // 2]
        pe_vs_sector = fund_data["trailing_pe"] / (median_pe + 1e-10)
    else:
        pe_vs_sector = 1.0

    # P/B vs sector median
    sector_pbs = [
        f["price_to_book"] for f in sector_funds
        if f.get("price_to_book") is not None and f["price_to_book"] > 0
    ]
    if sector_pbs and fund_data.get("price_to_book") and fund_data["price_to_book"] > 0:
        median_pb = sorted(sector_pbs)[len(sector_pbs) // 2]
        pb_vs_sector = fund_data["price_to_book"] / (median_pb + 1e-10)
    else:
        pb_vs_sector = 1.0

    # Market cap rank among all stocks
    all_mcaps = sorted(
        [f.get("market_cap", 0) or 0 for f in all_fundamentals], reverse=True
    )
    my_mcap = fund_data.get("market_cap", 0) or 0
    mcap_rank = next(
        (i + 1 for i, m in enumerate(all_mcaps) if m <= my_mcap), len(all_mcaps)
    )

    return {
        "pe_vs_sector_median": pe_vs_sector,
        "pb_vs_sector_median": pb_vs_sector,
        "mcap_rank": mcap_rank,
        "pe_percentile_1y": 0.5,  # Would need historical P/E data for this
    }


def _safe_val(val, default: float = 0.0) -> float:
    """Safely convert to float, returning default for None/NaN."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _safe_log(val, default: float = 0.0) -> float:
    """Safely compute log of a value."""
    if val is None:
        return default
    try:
        f = float(val)
        if f <= 0 or math.isnan(f) or math.isinf(f):
            return default
        return math.log(f)
    except (ValueError, TypeError):
        return default


def _default_features() -> dict:
    """Return default fundamental features when data is unavailable."""
    return {
        "trailing_pe": 0.0,
        "forward_pe": 0.0,
        "price_to_book": 0.0,
        "dividend_yield": 0.0,
        "market_cap_log": 0.0,
        "roe": 0.0,
        "debt_to_equity": 0.0,
        "earnings_growth": 0.0,
        "revenue_growth": 0.0,
        "profit_margin": 0.0,
        "pe_vs_sector_median": 1.0,
        "pb_vs_sector_median": 1.0,
        "mcap_rank": 25,
        "pe_percentile_1y": 0.5,
    }


def get_fundamental_feature_names() -> list[str]:
    """Return list of fundamental feature column names."""
    return [
        "trailing_pe", "forward_pe", "price_to_book", "dividend_yield",
        "market_cap_log", "roe", "debt_to_equity", "earnings_growth",
        "revenue_growth", "profit_margin", "pe_vs_sector_median",
        "pb_vs_sector_median", "mcap_rank", "pe_percentile_1y",
    ]
