"""
Quarterly FII/DII ownership tracking service.

This module is intentionally source-agnostic. NSE/BSE shareholding pattern
exports, screener exports, or manually curated internal files can all be
normalized into the same durable store:

    storage/shareholding/quarterly_ownership.csv

Required input concepts are simple:
    quarter, symbol, fii_pct, dii_pct

Optional columns such as company, sector, source, and updated_at are enriched
from the dashboard universe when missing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.universe import EXPANDED_UNIVERSE, get_company, get_sector, universe_tier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHAREHOLDING_DIR = PROJECT_ROOT / "storage" / "shareholding"
OWNERSHIP_STORE = SHAREHOLDING_DIR / "quarterly_ownership.csv"
INCOMING_DIR = SHAREHOLDING_DIR / "incoming"

STORE_COLUMNS = [
    "quarter",
    "as_on_date",
    "symbol",
    "company",
    "sector",
    "universe",
    "fii_pct",
    "dii_pct",
    "source",
    "updated_at",
]

_COLUMN_ALIASES = {
    "quarter": {"quarter", "qtr", "period", "report_quarter", "shareholding_quarter"},
    "as_on_date": {"as_on_date", "as_on", "date", "report_date", "shareholding_date"},
    "symbol": {"symbol", "ticker", "nse_symbol", "security", "scrip", "stock"},
    "company": {"company", "company_name", "name", "issuer"},
    "sector": {"sector", "industry"},
    "fii_pct": {
        "fii_pct",
        "fii",
        "fii_percent",
        "fii_percentage",
        "foreign_institutional_investors",
        "foreign_institution_holding",
        "foreign_institutional_holding",
    },
    "dii_pct": {
        "dii_pct",
        "dii",
        "dii_percent",
        "dii_percentage",
        "domestic_institutional_investors",
        "domestic_institution_holding",
        "domestic_institutional_holding",
    },
    "source": {"source", "source_file", "url"},
    "updated_at": {"updated_at", "ingested_at", "last_updated"},
}


@dataclass(frozen=True)
class IngestResult:
    store_path: Path
    rows_ingested: int
    total_rows: int
    quarters: tuple[str, ...]
    symbols: tuple[str, ...]


def ensure_shareholding_store(path: Path = OWNERSHIP_STORE) -> Path:
    """Create the shareholding store and incoming directory if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=STORE_COLUMNS).to_csv(path, index=False)
    return path


def load_ownership_history(path: Path = OWNERSHIP_STORE) -> pd.DataFrame:
    """Load normalized quarterly ownership history."""
    ensure_shareholding_store(path)
    df = pd.read_csv(path)
    return _normalize_store_frame(df)


def ingest_quarterly_file(
    input_path: str | Path,
    *,
    quarter: str | None = None,
    store_path: Path = OWNERSHIP_STORE,
    source: str | None = None,
) -> IngestResult:
    """Normalize a quarterly shareholding file into the durable store.

    `input_path` may be `.csv`, `.xlsx`, or `.xls`. Existing rows for the same
    (quarter, symbol) are replaced, making the operation idempotent.
    """
    input_path = Path(input_path)
    incoming = _read_tabular(input_path)
    normalized = normalize_quarterly_frame(
        incoming,
        quarter=quarter or _quarter_from_filename(input_path.name),
        source=source or input_path.name,
    )
    if normalized.empty:
        raise ValueError("No valid ownership rows found in input file.")

    current = load_ownership_history(store_path)
    if current.empty:
        merged = normalized
    else:
        keys = set(zip(normalized["quarter"], normalized["symbol"]))
        keep = ~current.apply(lambda r: (r["quarter"], r["symbol"]) in keys, axis=1)
        merged = pd.concat([current.loc[keep], normalized], ignore_index=True)

    merged = _normalize_store_frame(merged)
    merged.to_csv(ensure_shareholding_store(store_path), index=False)

    return IngestResult(
        store_path=store_path,
        rows_ingested=len(normalized),
        total_rows=len(merged),
        quarters=tuple(sorted(normalized["quarter"].dropna().unique().tolist())),
        symbols=tuple(sorted(normalized["symbol"].dropna().unique().tolist())),
    )


def ingest_quarterly_records(
    records: pd.DataFrame | list[dict],
    *,
    quarter: str | None = None,
    store_path: Path = OWNERSHIP_STORE,
    source: str = "automated",
) -> IngestResult:
    """Normalize and upsert fetched quarterly ownership records."""
    incoming = pd.DataFrame(records)
    normalized = normalize_quarterly_frame(incoming, quarter=quarter, source=source)
    if normalized.empty:
        raise ValueError("No valid ownership rows found in fetched records.")

    current = load_ownership_history(store_path)
    if current.empty:
        merged = normalized
    else:
        keys = set(zip(normalized["quarter"], normalized["symbol"]))
        keep = ~current.apply(lambda r: (r["quarter"], r["symbol"]) in keys, axis=1)
        merged = pd.concat([current.loc[keep], normalized], ignore_index=True)

    merged = _normalize_store_frame(merged)
    merged.to_csv(ensure_shareholding_store(store_path), index=False)

    return IngestResult(
        store_path=store_path,
        rows_ingested=len(normalized),
        total_rows=len(merged),
        quarters=tuple(sorted(normalized["quarter"].dropna().unique().tolist())),
        symbols=tuple(sorted(normalized["symbol"].dropna().unique().tolist())),
    )


def normalize_quarterly_frame(
    df: pd.DataFrame,
    *,
    quarter: str | None = None,
    source: str = "",
) -> pd.DataFrame:
    """Return a normalized ownership frame from a raw quarterly export."""
    if df is None or df.empty:
        return pd.DataFrame(columns=STORE_COLUMNS)

    renamed = df.rename(columns=_canonical_column_map(df.columns)).copy()

    if "symbol" not in renamed.columns:
        raise ValueError("Input file must include a symbol/ticker column.")
    if "quarter" not in renamed.columns and not quarter:
        raise ValueError("Input file must include a quarter column or pass --quarter.")
    if "fii_pct" not in renamed.columns and "dii_pct" not in renamed.columns:
        raise ValueError("Input file must include at least one FII or DII percentage column.")

    out = pd.DataFrame()
    out["quarter"] = renamed["quarter"] if "quarter" in renamed.columns else quarter
    out["quarter"] = out["quarter"].map(normalize_quarter)
    out["as_on_date"] = (
        renamed["as_on_date"].astype(str).str.strip()
        if "as_on_date" in renamed.columns
        else out["quarter"]
    )
    out["symbol"] = renamed["symbol"].astype(str).map(_normalize_symbol)

    out["company"] = (
        renamed["company"].astype(str).str.strip()
        if "company" in renamed.columns
        else out["symbol"].map(get_company)
    )
    out["sector"] = (
        renamed["sector"].astype(str).str.strip()
        if "sector" in renamed.columns
        else out["symbol"].map(get_sector)
    )
    out["universe"] = out["symbol"].map(universe_tier)
    out["fii_pct"] = _coerce_percent(renamed.get("fii_pct"))
    out["dii_pct"] = _coerce_percent(renamed.get("dii_pct"))
    out["source"] = renamed["source"].astype(str).str.strip() if "source" in renamed.columns else source
    out["updated_at"] = (
        renamed["updated_at"].astype(str).str.strip()
        if "updated_at" in renamed.columns
        else datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    out = out.dropna(subset=["quarter", "symbol"])
    out = out[out["symbol"] != ""]
    out = out[out["fii_pct"].notna() | out["dii_pct"].notna()]
    out = out.drop_duplicates(subset=["quarter", "symbol"], keep="last")
    return _normalize_store_frame(out)


def compute_ownership_deltas(history: pd.DataFrame) -> pd.DataFrame:
    """Add quarter-on-quarter FII/DII ownership delta and relative increase."""
    df = _normalize_store_frame(history)
    if df.empty:
        return df

    df["_quarter_rank"] = df["quarter"].map(_quarter_rank)
    df = df.sort_values(["symbol", "_quarter_rank"]).copy()

    for investor in ("fii", "dii"):
        col = f"{investor}_pct"
        prev_col = f"prev_{investor}_pct"
        delta_col = f"{investor}_delta_pp"
        rel_col = f"{investor}_increase_pct"
        df[prev_col] = df.groupby("symbol")[col].shift(1)
        df[delta_col] = (df[col] - df[prev_col]).round(3)
        base = df[prev_col].replace(0, np.nan)
        df[rel_col] = ((df[delta_col] / base) * 100).round(2)

    df["institutional_delta_pp"] = (df["fii_delta_pp"].fillna(0) + df["dii_delta_pp"].fillna(0)).round(3)
    df["ownership_signal"] = np.select(
        [
            df["fii_delta_pp"].fillna(0) >= 1.0,
            df["fii_delta_pp"].fillna(0) <= -1.0,
            df["dii_delta_pp"].fillna(0) >= 1.0,
            df["dii_delta_pp"].fillna(0) <= -1.0,
        ],
        ["FII accumulation", "FII distribution", "DII accumulation", "DII distribution"],
        default="Stable",
    )
    return df.drop(columns=["_quarter_rank"])


def latest_ownership_snapshot(history: pd.DataFrame) -> pd.DataFrame:
    """Return the latest row per symbol with QoQ deltas attached."""
    deltas = compute_ownership_deltas(history)
    if deltas.empty:
        return deltas
    deltas["_quarter_rank"] = deltas["quarter"].map(_quarter_rank)
    latest = (
        deltas.sort_values(["symbol", "_quarter_rank"])
        .groupby("symbol", as_index=False)
        .tail(1)
        .drop(columns=["_quarter_rank"])
        .sort_values(["fii_delta_pp", "dii_delta_pp"], ascending=False)
    )
    return latest


def sector_ownership_summary(history: pd.DataFrame, *, quarter: str | None = None) -> pd.DataFrame:
    """Aggregate latest ownership change by sector."""
    deltas = compute_ownership_deltas(history)
    if deltas.empty:
        return pd.DataFrame()
    if quarter:
        deltas = deltas[deltas["quarter"] == normalize_quarter(quarter)]
    else:
        max_rank = deltas["quarter"].map(_quarter_rank).max()
        deltas = deltas[deltas["quarter"].map(_quarter_rank) == max_rank]
    if deltas.empty:
        return pd.DataFrame()

    grouped = (
        deltas.groupby("sector", dropna=False)
        .agg(
            stock_count=("symbol", "nunique"),
            avg_fii_pct=("fii_pct", "mean"),
            avg_dii_pct=("dii_pct", "mean"),
            avg_fii_delta_pp=("fii_delta_pp", "mean"),
            avg_dii_delta_pp=("dii_delta_pp", "mean"),
            fii_accumulation_count=("fii_delta_pp", lambda s: int((s > 0).sum())),
            dii_accumulation_count=("dii_delta_pp", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )
    for col in ["avg_fii_pct", "avg_dii_pct", "avg_fii_delta_pp", "avg_dii_delta_pp"]:
        grouped[col] = grouped[col].round(2)
    return grouped.sort_values("avg_fii_delta_pp", ascending=False)


def top_ownership_accumulation(
    history: pd.DataFrame,
    *,
    investor: str = "fii",
    top_n: int = 25,
    min_previous_pct: float = 0.0,
) -> pd.DataFrame:
    """Return top QoQ ownership increases for FII or DII."""
    investor = investor.lower()
    if investor not in {"fii", "dii"}:
        raise ValueError("investor must be 'fii' or 'dii'")
    latest = latest_ownership_snapshot(history)
    if latest.empty:
        return latest
    prev_col = f"prev_{investor}_pct"
    delta_col = f"{investor}_delta_pp"
    latest = latest[latest[prev_col].fillna(0) >= min_previous_pct].copy()
    return latest.sort_values(delta_col, ascending=False).head(top_n)


def _read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input file type: {suffix}")


def _normalize_store_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=STORE_COLUMNS)
    out = df.copy()
    for col in STORE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[STORE_COLUMNS]
    out["quarter"] = out["quarter"].map(normalize_quarter)
    out["symbol"] = out["symbol"].astype(str).map(_normalize_symbol)
    out["company"] = out.apply(
        lambda r: get_company(r["symbol"]) if not str(r["company"]).strip() or str(r["company"]).lower() == "nan" else r["company"],
        axis=1,
    )
    out["sector"] = out.apply(
        lambda r: get_sector(r["symbol"]) if not str(r["sector"]).strip() or str(r["sector"]).lower() == "nan" else r["sector"],
        axis=1,
    )
    out["universe"] = out["symbol"].map(universe_tier)
    out["fii_pct"] = pd.to_numeric(out["fii_pct"], errors="coerce")
    out["dii_pct"] = pd.to_numeric(out["dii_pct"], errors="coerce")
    out = out.dropna(subset=["quarter", "symbol"])
    out = out.drop_duplicates(subset=["quarter", "symbol"], keep="last")
    out["_quarter_rank"] = out["quarter"].map(_quarter_rank)
    out = out.sort_values(["_quarter_rank", "symbol"]).drop(columns=["_quarter_rank"])
    return out.reset_index(drop=True)


def _canonical_column_map(columns) -> dict[str, str]:
    mapping: dict[str, str] = {}
    alias_lookup = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            alias_lookup[_clean_column(alias)] = canonical
    for col in columns:
        cleaned = _clean_column(str(col))
        if cleaned in alias_lookup:
            mapping[col] = alias_lookup[cleaned]
    return mapping


def _clean_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _normalize_symbol(value: object) -> str:
    symbol = str(value or "").strip().upper()
    symbol = symbol.replace(".NS", "").replace("NSE:", "")
    return symbol


def _coerce_percent(series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    cleaned = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_quarter(value: object) -> str | None:
    """Normalize common quarter strings to YYYYQn."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return None
    compact = re.sub(r"[^A-Z0-9]", "", raw.upper())

    m = re.search(r"(20\d{2})Q([1-4])", compact)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"

    m = re.search(r"Q([1-4])(20\d{2})", compact)
    if m:
        return f"{m.group(2)}Q{m.group(1)}"

    m = re.search(r"(20\d{2})(0[1-9]|1[0-2])", compact)
    if m:
        return _quarter_from_month(int(m.group(1)), int(m.group(2)))

    try:
        ts = pd.to_datetime(raw, errors="coerce")
        if pd.notna(ts):
            return _quarter_from_month(int(ts.year), int(ts.month))
    except Exception:
        pass
    return raw


def _quarter_from_filename(name: str) -> str | None:
    return normalize_quarter(name)


def _quarter_from_month(year: int, month: int) -> str:
    q = ((month - 1) // 3) + 1
    return f"{year}Q{q}"


def _quarter_rank(quarter: object) -> int:
    q = normalize_quarter(quarter) or ""
    m = re.search(r"(20\d{2})Q([1-4])", q)
    if not m:
        return 0
    return int(m.group(1)) * 10 + int(m.group(2))
