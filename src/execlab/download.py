"""Download Binance spot aggTrades daily files from data.binance.vision (public, no key).

    python -m execlab.download --start 2026-10-27 --end 2026-10-29 --out ../data/binance
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

URL = "https://data.binance.vision/data/spot/daily/aggTrades/{s}/{s}-aggTrades-{d}.zip"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--start", required=True, type=date.fromisoformat)
    ap.add_argument("--end", required=True, type=date.fromisoformat)
    ap.add_argument("--out", type=Path, default=Path("../data/binance"))
    a = ap.parse_args()
    import requests

    a.out.mkdir(parents=True, exist_ok=True)
    d = a.start
    while d <= a.end:
        for s in a.symbols:
            f = a.out / f"{s}-aggTrades-{d}.zip"
            if f.exists() and f.stat().st_size > 0:
                continue
            r = requests.get(URL.format(s=s, d=d), timeout=300)
            if r.status_code == 404:
                print(f"{f.name}: not published")
                continue
            r.raise_for_status()
            f.write_bytes(r.content)
            print(f"{f.name}: {len(r.content) / 1e6:.1f} MB")
        d += timedelta(days=1)


if __name__ == "__main__":
    main()
