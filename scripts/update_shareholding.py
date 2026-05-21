"""Import quarterly FII/DII ownership data into the dashboard store.

Examples:
    python scripts/update_shareholding.py storage/shareholding/incoming/q1_2026.csv --quarter 2026Q1
    python scripts/update_shareholding.py storage/shareholding/incoming/*.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.shareholding_tracker import ingest_quarterly_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Import quarterly FII/DII ownership data.")
    parser.add_argument("files", nargs="+", help="CSV/XLSX files to import.")
    parser.add_argument("--quarter", help="Quarter override, e.g. 2026Q1. Required if file has no quarter column.")
    args = parser.parse_args()

    for raw in args.files:
        path = Path(raw)
        if not path.exists():
            print(f"Missing file: {path}", file=sys.stderr)
            return 1
        result = ingest_quarterly_file(path, quarter=args.quarter)
        print(
            f"Imported {result.rows_ingested} rows from {path.name}; "
            f"quarters={', '.join(result.quarters)}; store={result.store_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
