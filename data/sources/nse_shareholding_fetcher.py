"""
NSE quarterly shareholding scraper for FII/DII ownership.

Source: NSE corporate filings shareholding pattern page. The public page loads:

    /api/corporate-share-holdings-master?index=equities&symbol=<SYMBOL>
    /api/corporate-share-holdings-equities?ndsId=<recordId>&index=public-shareholder

For current-format filings the public-shareholder detail has:
    Sub-Total (B)(1) = domestic institutional shareholding (DII lens)
    Sub-Total (B)(2) = foreign institutional shareholding (FII lens)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import requests
from loguru import logger

from data.shareholding_tracker import normalize_quarter


NSE_BASE = "https://www.nseindia.com"
SHAREHOLDING_PAGE = f"{NSE_BASE}/companies-listing/corporate-filings-shareholding-pattern"
MASTER_ENDPOINT = f"{NSE_BASE}/api/corporate-share-holdings-master"
DETAIL_ENDPOINT = f"{NSE_BASE}/api/corporate-share-holdings-equities"


@dataclass(frozen=True)
class ShareholdingFetchStats:
    symbols_requested: int
    symbols_succeeded: int
    records_fetched: int
    failures: dict[str, str]


class NSEShareholdingFetcher:
    """Fetch FII/DII ownership percentages from NSE shareholding filings."""

    def __init__(
        self,
        *,
        timeout: float = 20.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        request_delay: float = 0.25,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": SHAREHOLDING_PAGE,
            "Connection": "keep-alive",
        })
        self._bootstrapped = False

    def fetch_symbols(
        self,
        symbols: Iterable[str],
        *,
        quarters: int = 4,
    ) -> tuple[pd.DataFrame, ShareholdingFetchStats]:
        """Fetch latest `quarters` shareholding records for each symbol."""
        rows: list[dict] = []
        failures: dict[str, str] = {}
        symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]

        for symbol in symbols:
            try:
                fetched = self.fetch_symbol(symbol, quarters=quarters)
                rows.extend(fetched)
                if not fetched:
                    failures[symbol] = "No usable quarterly filings found"
            except Exception as exc:
                failures[symbol] = str(exc)
                logger.warning(f"NSE shareholding fetch failed for {symbol}: {exc}")
            time.sleep(self.request_delay)

        frame = pd.DataFrame(rows)
        stats = ShareholdingFetchStats(
            symbols_requested=len(symbols),
            symbols_succeeded=len(set(frame["symbol"])) if not frame.empty and "symbol" in frame.columns else 0,
            records_fetched=len(frame),
            failures=failures,
        )
        return frame, stats

    def fetch_symbol(self, symbol: str, *, quarters: int = 4) -> list[dict]:
        """Fetch normalized ownership rows for one NSE symbol."""
        symbol = symbol.strip().upper()
        filings = self.fetch_master_filings(symbol)
        if not filings:
            return []

        rows: list[dict] = []
        for filing in filings[: max(1, quarters)]:
            record_id = filing.get("recordId")
            if not record_id:
                continue
            public_rows = self.fetch_public_shareholding(record_id)
            parsed = parse_public_shareholding(public_rows)
            if parsed["fii_pct"] is None and parsed["dii_pct"] is None:
                logger.debug(f"No FII/DII row found for {symbol} recordId={record_id}")
                continue

            rows.append({
                "quarter": normalize_quarter(filing.get("date")),
                "symbol": symbol,
                "company": filing.get("name") or symbol,
                "fii_pct": parsed["fii_pct"],
                "dii_pct": parsed["dii_pct"],
                "source": filing.get("xbrl") or SHAREHOLDING_PAGE,
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            time.sleep(self.request_delay)
        return rows

    def fetch_master_filings(self, symbol: str) -> list[dict]:
        """Return NSE master shareholding rows for a symbol, newest first."""
        payload = self._get_json(
            MASTER_ENDPOINT,
            params={"index": "equities", "symbol": symbol.strip().upper()},
        )
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            data = payload.get("data")
            return data if isinstance(data, list) else []
        return []

    def fetch_public_shareholding(self, record_id: str | int) -> list[dict]:
        """Return the Table III public-shareholder rows for one filing."""
        payload = self._get_json(
            DETAIL_ENDPOINT,
            params={"ndsId": str(record_id), "index": "public-shareholder"},
        )
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            data = payload.get("data")
            return data if isinstance(data, list) else []
        return []

    def _bootstrap(self) -> None:
        if self._bootstrapped:
            return
        response = self.session.get(SHAREHOLDING_PAGE, timeout=self.timeout)
        response.raise_for_status()
        self._bootstrapped = True

    def _get_json(self, url: str, *, params: dict) -> object:
        self._bootstrap()
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code in {401, 403}:
                    self._bootstrapped = False
                    self._bootstrap()
                    response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(f"NSE request failed for {url}: {last_error}")


def parse_public_shareholding(rows: list[dict]) -> dict[str, float | None]:
    """Extract DII/FII ownership from NSE public-shareholder detail rows."""
    by_label = {_clean_label(row.get("COL_I")): row for row in rows if row.get("COL_I") is not None}

    dii_row = _first_matching(by_label, [
        "subtotal b1",
        "subtotal b 1",
        "sub total b1",
        "sub total b 1",
        "sub-total b1",
        "sub-total b 1",
        "institutions domestic total",
    ])
    fii_row = _first_matching(by_label, [
        "subtotal b2",
        "subtotal b 2",
        "sub total b2",
        "sub total b 2",
        "sub-total b2",
        "sub-total b 2",
        "institutions foreign total",
    ])

    dii_pct = _row_pct(dii_row)
    fii_pct = _row_pct(fii_row)

    if dii_pct is None:
        dii_pct = _sum_rows(by_label, _DII_FALLBACK_LABELS)
    if fii_pct is None:
        fii_pct = _sum_rows(by_label, _FII_FALLBACK_LABELS)

    return {
        "fii_pct": round(fii_pct, 4) if fii_pct is not None else None,
        "dii_pct": round(dii_pct, 4) if dii_pct is not None else None,
    }


_DII_FALLBACK_LABELS = {
    "mutual funds",
    "venture capital funds",
    "alternate investment funds",
    "banks",
    "insurance companies",
    "provident funds pension funds",
    "pension funds",
    "other financial institutions",
}

_FII_FALLBACK_LABELS = {
    "foreign direct investment",
    "foreign venture capital investors",
    "foreign portfolio investors category i",
    "foreign portfolio investors category ii",
    "foreign portfolio investor category iii",
    "foreign institutional investors",
}


def _first_matching(by_label: dict[str, dict], needles: list[str]) -> dict | None:
    for label, row in by_label.items():
        for needle in needles:
            if needle in label:
                return row
    return None


def _sum_rows(by_label: dict[str, dict], labels: set[str]) -> float | None:
    total = 0.0
    found = False
    for label, row in by_label.items():
        if label in labels:
            value = _row_pct(row)
            if value is not None:
                total += value
                found = True
    return total if found else None


def _row_pct(row: dict | None) -> float | None:
    if not row:
        return None
    for col in ("COL_VIII", "COL_IX_TotalABC", "COL_XI"):
        value = _to_float(row.get(col))
        if value is not None:
            return value
    return None


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).replace(",", "").replace("%", "").strip()
        if text in {"", "-", "None", "null"}:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _clean_label(value: object) -> str:
    text = str(value or "").lower()
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() else " ")
    return " ".join("".join(keep).split())
