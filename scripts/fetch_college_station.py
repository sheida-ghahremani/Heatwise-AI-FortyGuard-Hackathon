"""Fetch and cache one College Station FortyGuard heatmap.

Run from the repository root. The API key is read from .env and is never
written into the output.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from fortyguard import FortyGuardClient
from heatwise.live_data import TAMU_AOI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().date().isoformat())
    parser.add_argument("--time", default="15:00")
    parser.add_argument("--granularity", type=int, choices=[60, 80, 100], default=100)
    args = parser.parse_args()

    load_dotenv(".env")
    response = FortyGuardClient().create_heatmap(
        polygon_aoi=TAMU_AOI,
        start_date=args.date,
        start_time=args.time,
        filter_type=1,
        granularity=args.granularity,
        verbose=True,
    )
    output = Path("data/heatmaps") / f"heatmap_college_station_{args.date}_{args.time.replace(':', '')}_{args.granularity}m.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(response["result"], indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
