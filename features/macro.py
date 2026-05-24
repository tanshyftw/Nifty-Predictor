"""
Macro/index feature computation.
Extracts global index returns, VIX, FII/DII flows, and sector-relative features.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger


def compute_macro_features(macro_data: dict | None) -> dict:
    """
    Extract macro features from a single day's macro record.

    Input: dict from macro_data table
    Output: dict with 16 macro feature values
    """
    if macro_data is None:
        return _default_features()

    return {
        # Global overnight returns
        "sp500_overnight_ret": _safe(macro_data.get("sp500_ret")),
        "nasdaq_overnight_ret": _safe(macro_data.get("nasdaq_ret")),
        "dow_overnight_ret": _safe(macro_data.get("dow_ret")),
        "ftse_overnight_ret": _safe(macro_data.get("ftse_ret")),
        "nikkei_overnight_ret": _safe(macro_data.get("nikkei_ret")),
        "hangseng_overnight_ret": _safe(macro_data.get("hangseng_ret")),
        # Indian market
        "nifty50_prev_ret": _safe(macro_data.get("nifty50_ret")),
        "bank_nifty_prev_ret": _safe(macro_data.get("bank_nifty_ret")),
        "india_vix": _safe(macro_data.get("india_vix")),
        "india_vix_change": _safe(macro_data.get("india_vix_change")),
        # FII/DII
        "fii_net_buy": _safe(macro_data.get("fii_net_buy")),
        "dii_net_buy": _safe(macro_data.get("dii_net_buy")),
        "fii_dii_ratio": _compute_fii_dii_ratio(
            macro_data.get("fii_net_buy"), macro_data.get("dii_net_buy")
        ),
        # Currency
        "usdinr_level": _safe(macro_data.get("usdinr")),
        "usdinr_change_1d": _safe(macro_data.get("usdinr_change")),
    }


def compute_macro_features_with_history(
    macro_records: list[dict], current_idx: int
) -> dict:
    """
    Compute macro features including rolling features that need history.
    Adds nifty50_ret_5d (5-day trailing Nifty return).
    """
    features = compute_macro_features(
        macro_records[current_idx] if current_idx < len(macro_records) else None
    )

    # Nifty 50 trailing 5-day return
    if current_idx >= 4:
        nifty_rets = []
        for i in range(max(0, current_idx - 4), current_idx + 1):
            ret = macro_records[i].get("nifty50_ret")
            if ret is not None:
                nifty_rets.append(ret)
        if nifty_rets:
            # Cumulative return over 5 days
            cum_ret = 1.0
            for r in nifty_rets:
                cum_ret *= (1 + r)
            features["nifty50_ret_5d"] = cum_ret - 1
        else:
            features["nifty50_ret_5d"] = 0.0
    else:
        features["nifty50_ret_5d"] = 0.0

    return features


def _compute_fii_dii_ratio(fii_net, dii_net) -> float:
    """Compute FII/DII ratio as a sentiment indicator."""
    fii = _safe(fii_net)
    dii = _safe(dii_net)
    if abs(dii) < 1e-6:
        return 0.0
    return fii / dii


def _safe(val, default: float = 0.0) -> float:
    """Safely convert to float."""
    if val is None:
        return default
    try:
        f = float(val)
        if pd.isna(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _default_features() -> dict:
    """Return default macro features when data is unavailable."""
    return {
        "sp500_overnight_ret": 0.0,
        "nasdaq_overnight_ret": 0.0,
        "dow_overnight_ret": 0.0,
        "ftse_overnight_ret": 0.0,
        "nikkei_overnight_ret": 0.0,
        "hangseng_overnight_ret": 0.0,
        "nifty50_prev_ret": 0.0,
        "nifty50_ret_5d": 0.0,
        "bank_nifty_prev_ret": 0.0,
        "india_vix": 15.0,  # reasonable default
        "india_vix_change": 0.0,
        "fii_net_buy": 0.0,
        "dii_net_buy": 0.0,
        "fii_dii_ratio": 0.0,
        "usdinr_level": 85.0,  # reasonable default
        "usdinr_change_1d": 0.0,
    }


def get_macro_feature_names() -> list[str]:
    """Return list of macro feature column names."""
    return [
        "sp500_overnight_ret", "nasdaq_overnight_ret", "dow_overnight_ret",
        "ftse_overnight_ret", "nikkei_overnight_ret", "hangseng_overnight_ret",
        "nifty50_prev_ret", "nifty50_ret_5d", "bank_nifty_prev_ret",
        "india_vix", "india_vix_change",
        "fii_net_buy", "dii_net_buy", "fii_dii_ratio",
        "usdinr_level", "usdinr_change_1d",
    ]
