"""Hourly FortyGuard snapshots for the HeatWise College Station demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from fortyguard import FortyGuardClient


COLLEGE_STATION_TZ = ZoneInfo("America/Chicago")
GRANULARITY_M = 100
MAX_FALLBACK_DAYS = 7
HEATMAP_DIR = Path("data/heatmaps")

TAMU_AOI = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "Texas A&M demonstration area"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-96.3650, 30.5950],
                    [-96.3150, 30.5950],
                    [-96.3150, 30.6350],
                    [-96.3650, 30.6350],
                    [-96.3650, 30.5950],
                ]],
            },
        }
    ],
}
RELLIS_AOI = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature", "properties": {"name": "RELLIS Campus"},
        "geometry": {"type": "Polygon", "coordinates": [[
            [-96.4986, 30.6258], [-96.4512, 30.6258], [-96.4512, 30.6505],
            [-96.4986, 30.6505], [-96.4986, 30.6258],
        ]]},
    }],
}
AREA_AOIS = {"tamu": TAMU_AOI, "rellis": RELLIS_AOI}


@dataclass(frozen=True)
class HeatmapSnapshot:
    path: Path
    requested_at: datetime
    fetched_at: datetime
    from_cache: bool

    @property
    def label(self) -> str:
        source = "Cached hourly snapshot" if self.from_cache else "Live API"
        return (
            f"{source} · {self.requested_at.strftime('%b %d, %Y · %-I:00 %p %Z')}"
            f" · {GRANULARITY_M} m"
        )


def current_hour(now: datetime | None = None) -> datetime:
    """Return the current College Station hour as a timezone-aware datetime."""
    if now is None:
        now = datetime.now(COLLEGE_STATION_TZ)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=COLLEGE_STATION_TZ)
    else:
        now = now.astimezone(COLLEGE_STATION_TZ)
    return now.replace(minute=0, second=0, microsecond=0)


def cache_path(requested_at: datetime, area: str = "tamu") -> Path:
    local = requested_at.astimezone(COLLEGE_STATION_TZ)
    return HEATMAP_DIR / f"{area}_{local:%Y%m%d_%H00}_{GRANULARITY_M}m.json"


def _read_fetched_at(path: Path) -> datetime:
    try:
        payload = json.loads(path.read_text())
        value = payload.get("_heatwise", {}).get("fetched_at")
        if value:
            return datetime.fromisoformat(value)
    except (OSError, ValueError, TypeError):
        pass
    return datetime.fromtimestamp(path.stat().st_mtime, tz=COLLEGE_STATION_TZ)


def _is_usable(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text())
        map_data = payload.get("result", payload).get("map_data", payload)
        features = map_data.get("features", [])
        return bool(features) and "average_temperature" in features[0].get("properties", {})
    except (OSError, ValueError, TypeError, AttributeError, IndexError):
        return False


def fetch_hourly_snapshot(
    requested_at: datetime | None = None,
    *,
    force: bool = False,
    area: str = "tamu",
) -> HeatmapSnapshot:
    """Fetch a snapshot once per local hour, reusing it on Streamlit reruns."""
    requested_at = current_hour(requested_at)
    if area not in AREA_AOIS:
        raise ValueError(f"Unknown HeatWise area: {area}")

    client = FortyGuardClient()
    path = cache_path(requested_at, area)
    if path.exists() and not force and _is_usable(path):
        return HeatmapSnapshot(path, requested_at, _read_fetched_at(path), True)

    response = client.create_heatmap(
        polygon_aoi=AREA_AOIS[area],
        start_date=requested_at.date().isoformat(),
        start_time=requested_at.strftime("%H:00"),
        filter_type=1,
        granularity=GRANULARITY_M,
        timeout=120.0,
        verbose=False,
    )
    fetched_at = datetime.now(COLLEGE_STATION_TZ)
    map_data = response.get("result", {}).get("map_data", {})
    if map_data.get("features"):
        payload = {
            "_heatwise": {
                "requested_at": requested_at.isoformat(),
                "original_requested_at": requested_at.isoformat(),
                "fallback_days": 0,
                "fetched_at": fetched_at.isoformat(),
                "timezone": str(COLLEGE_STATION_TZ),
                "granularity_m": GRANULARITY_M,
            },
            "result": response["result"],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return HeatmapSnapshot(path, requested_at, fetched_at, False)

    # Only fall back to a different cached hour after the API has actually
    # been tried for the requested hour and each allowed fallback day.  The
    # previous implementation returned any recent cache before contacting
    # FortyGuard, which made a valid Aug 26 file mask newer Aug 27/28 data.
    recent = latest_cached_snapshot(area)
    if recent and requested_at - recent.requested_at <= timedelta(days=MAX_FALLBACK_DAYS):
        return recent

    raise ValueError("FortyGuard returned no temperature layer for the exact selected hour")


def latest_cached_snapshot(area: str = "tamu") -> HeatmapSnapshot | None:
    """Return the newest usable snapshot for graceful API failure fallback."""
    legacy = list(HEATMAP_DIR.glob(f"college_station_*_{GRANULARITY_M}m.json")) if area == "tamu" else []
    paths = sorted([*HEATMAP_DIR.glob(f"{area}_*_{GRANULARITY_M}m.json"), *legacy])
    if not paths:
        return None
    usable_paths = [path for path in paths if _is_usable(path)]
    if not usable_paths:
        return None
    path = usable_paths[-1]
    payload = json.loads(path.read_text())
    metadata = payload.get("_heatwise", {})
    requested_at = datetime.fromisoformat(metadata["requested_at"])
    return HeatmapSnapshot(path, requested_at, _read_fetched_at(path), True)
