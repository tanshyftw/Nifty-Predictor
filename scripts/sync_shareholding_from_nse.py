"""Fetch quarterly FII/DII ownership from NSE and upsert the local store.

Examples:
    python scripts/sync_shareholding_from_nse.py --universe nifty50 --quarters 4
    python scripts/sync_shareholding_from_nse.py --symbols RELIANCE,TCS,HDFCBANK --quarters 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.largecap_extra_tickers import LARGECAP_NEXT50_STOCKS
from config.midcap_tickers import MIDCAP_STOCKS
from config.nifty50_tickers import NIFTY50_STOCKS
from data.shareholding_tracker import ingest_quarterly_records
from data.sources.nse_shareholding_fetcher import NSEShareholdingFetcher


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync quarterly FII/DII ownership from NSE.")
    parser.add_argument(
        "--universe",
        choices=["nifty50", "largecap", "midcap", "all"],
        default="nifty50",
        help="Tracked universe to scrape when --symbols is not supplied.",
    )
    parser.add_argument("--symbols", help="Comma-separated NSE symbols to scrape.")
    parser.add_argument("--quarters", type=int, default=4, help="Latest quarters per symbol.")
    parser.add_argument("--delay", type=float, default=0.35, help="Delay between NSE requests in seconds.")
    args = parser.parse_args()

    symbols = _symbols_from_args(args.symbols, args.universe)
    fetcher = NSEShareholdingFetcher(request_delay=args.delay)
    frame, stats = fetcher.fetch_symbols(symbols, quarters=args.quarters)

    if frame.empty:
        print(f"No rows fetched. Failures: {stats.failures}", file=sys.stderr)
        return 1

    result = ingest_quarterly_records(frame, source="NSE corporate shareholding API")
    print(
        f"Fetched {stats.records_fetched} rows for {stats.symbols_succeeded}/"
        f"{stats.symbols_requested} symbols; imported {result.rows_ingested} rows "
        f"across {len(result.quarters)} quarters into {result.store_path}."
    )
    if stats.failures:
        print("Failures:")
        for symbol, reason in stats.failures.items():
            print(f"  {symbol}: {reason}")
    return 0


def _symbols_from_args(raw_symbols: str | None, universe: str) -> list[str]:
    if raw_symbols:
        return [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]
    stocks = {}
    if universe in {"all", "midcap"}:
        stocks.update(MIDCAP_STOCKS)
    if universe in {"all", "largecap"}:
        stocks.update(LARGECAP_NEXT50_STOCKS)
    if universe in {"all", "nifty50"}:
        stocks.update(NIFTY50_STOCKS)
    return list(stocks.keys())


if __name__ == "__main__":
    raise SystemExit(main())
