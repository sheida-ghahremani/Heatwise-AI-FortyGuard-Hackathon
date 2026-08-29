from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .shade import solar_position


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
COLLEGE_STATION_LAT = 30.6187
COLLEGE_STATION_LON = -96.3365
CACHE_DIR = Path("cache/weather")
NWS_HEADERS = {"User-Agent": "HeatWise-AI/1.0 (university research application)", "Accept": "application/geo+json"}


def _same_day_cached_payload(area: str, selected: datetime) -> tuple[dict, Path] | None:
    """Reuse any same-day response containing the exact requested local hour."""
    target = selected.strftime("%Y-%m-%dT%H:00")
    pattern = f"{area}_v4_{selected:%Y%m%d}_*.json"
    for candidate in sorted(CACHE_DIR.glob(pattern), reverse=True):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if target in payload.get("hourly", {}).get("time", []):
                return payload, candidate
        except (OSError, ValueError, TypeError):
            continue
    return None


@dataclass(frozen=True)
class WeatherSnapshot:
    observed_at: datetime
    humidity_pct: float
    wind_mps: float
    wind_direction_deg: float
    shortwave_radiation_wm2: float
    direct_normal_irradiance_wm2: float
    diffuse_radiation_wm2: float
    cloud_cover_pct: float
    precipitation_mm: float
    source: str
    cache_path: Path


def _duration_hours(value: str) -> float:
    match = re.fullmatch(r"PT(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?", value)
    if not match:
        return 1.0
    return float(match.group(1) or 0) + float(match.group(2) or 0) / 60 or 1.0


def _grid_value(series: dict, selected: datetime) -> tuple[float, float]:
    target = selected.timestamp()
    for item in series.get("values", []):
        start_text, duration_text = item["validTime"].split("/", 1)
        start = datetime.fromisoformat(start_text.replace("Z", "+00:00"))
        duration = _duration_hours(duration_text)
        if start.timestamp() <= target < (start + timedelta(hours=duration)).timestamp():
            return float(item["value"] or 0.0), duration
    raise ValueError("NWS grid did not contain the selected hour")


def _fetch_nws_weather(selected: datetime, latitude: float, longitude: float, cache_path: Path) -> WeatherSnapshot:
    """Exact-hour US National Weather Service fallback for shared-host rate limits."""
    point = requests.get(
        f"https://api.weather.gov/points/{latitude:.4f},{longitude:.4f}",
        headers=NWS_HEADERS, timeout=20,
    )
    point.raise_for_status()
    point_props = point.json()["properties"]
    hourly_response = requests.get(point_props["forecastHourly"], headers=NWS_HEADERS, timeout=20)
    hourly_response.raise_for_status()
    target = selected.timestamp()
    period = next(
        (
            item for item in hourly_response.json()["properties"]["periods"]
            if datetime.fromisoformat(item["startTime"]).timestamp() <= target
            < datetime.fromisoformat(item["endTime"]).timestamp()
        ),
        None,
    )
    if period is None:
        raise ValueError("NWS hourly forecast did not contain the selected hour")

    grid_response = requests.get(point_props["forecastGridData"], headers=NWS_HEADERS, timeout=20)
    grid_response.raise_for_status()
    grid = grid_response.json()["properties"]
    cloud_cover, _ = _grid_value(grid["skyCover"], selected)
    precipitation_total, precipitation_hours = _grid_value(grid["quantitativePrecipitation"], selected)
    precipitation_rate = precipitation_total / precipitation_hours

    speed_match = re.search(r"([0-9.]+)", period.get("windSpeed", "0"))
    wind_mps = float(speed_match.group(1)) * 0.44704 if speed_match else 0.0
    directions = {name: index * 22.5 for index, name in enumerate(
        ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
    )}

    # Haurwitz clear-sky global horizontal irradiance, attenuated using the
    # NWS hourly cloud fraction. Direct/diffuse partition is an explicit
    # modeled fallback because NWS does not publish surface solar irradiance.
    elevation, _ = solar_position(selected, latitude, longitude)
    cos_zenith = max(0.0, math.sin(math.radians(elevation)))
    clear_sky = 1098.0 * cos_zenith * math.exp(-0.059 / max(cos_zenith, 1e-6)) if cos_zenith else 0.0
    shortwave = clear_sky * (1.0 - 0.75 * (cloud_cover / 100.0) ** 3.4)
    diffuse_fraction = min(0.85, 0.15 + 0.70 * cloud_cover / 100.0)
    diffuse = shortwave * diffuse_fraction
    direct_horizontal = max(0.0, shortwave - diffuse)
    direct_normal = direct_horizontal / max(cos_zenith, 0.1) if cos_zenith else 0.0

    return WeatherSnapshot(
        observed_at=selected,
        humidity_pct=float(period["relativeHumidity"]["value"]),
        wind_mps=wind_mps,
        wind_direction_deg=float(directions.get(period.get("windDirection", "N"), 0.0)),
        shortwave_radiation_wm2=shortwave,
        direct_normal_irradiance_wm2=direct_normal,
        diffuse_radiation_wm2=diffuse,
        cloud_cover_pct=cloud_cover,
        precipitation_mm=precipitation_rate,
        source="NWS/NOAA exact-hour forecast; Haurwitz + cloud-adjusted solar irradiance",
        cache_path=cache_path,
    )


def fetch_hourly_weather(
    when: datetime, *, force: bool = False, latitude: float = COLLEGE_STATION_LAT,
    longitude: float = COLLEGE_STATION_LON, area: str = "college_station"
) -> WeatherSnapshot:
    """Return hourly humidity and 10 m wind for College Station from Open-Meteo."""
    if when.tzinfo is None:
        raise ValueError("Weather time must be timezone-aware")
    selected = when.replace(minute=0, second=0, microsecond=0)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{area}_v4_{selected.strftime('%Y%m%d_%H%z')}.json"

    if cache_path.exists() and not force:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        source = "Open-Meteo hourly cache"
    else:
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=0.7,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retry))
        try:
            response = session.get(
                OPEN_METEO_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "hourly": (
                        "relative_humidity_2m,wind_speed_10m,wind_direction_10m,"
                        "shortwave_radiation,direct_normal_irradiance,diffuse_radiation,cloud_cover,precipitation"
                    ),
                    "wind_speed_unit": "ms",
                    "timezone": "America/Chicago",
                    "start_date": selected.date().isoformat(),
                    "end_date": selected.date().isoformat(),
                },
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            source = "Open-Meteo live hourly forecast"
        except (requests.RequestException, ValueError):
            # A forced refresh may fail transiently. Reuse only this exact
            # local hour; never substitute a different hour or day.
            same_day_cache = _same_day_cached_payload(area, selected)
            if cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                source = "Open-Meteo exact-hour cache after refresh failure"
            elif same_day_cache:
                payload, reused_path = same_day_cache
                source = f"Open-Meteo same-day exact-hour cache ({reused_path.name})"
            else:
                return _fetch_nws_weather(selected, latitude, longitude, cache_path)

    hourly = payload["hourly"]
    target = selected.strftime("%Y-%m-%dT%H:00")
    try:
        index = hourly["time"].index(target)
    except ValueError as error:
        raise ValueError(f"Open-Meteo did not return weather for {target}") from error

    return WeatherSnapshot(
        observed_at=selected,
        humidity_pct=float(hourly["relative_humidity_2m"][index]),
        wind_mps=float(hourly["wind_speed_10m"][index]),
        wind_direction_deg=float(hourly["wind_direction_10m"][index]),
        shortwave_radiation_wm2=float(hourly["shortwave_radiation"][index]),
        direct_normal_irradiance_wm2=float(hourly["direct_normal_irradiance"][index]),
        diffuse_radiation_wm2=float(hourly["diffuse_radiation"][index]),
        cloud_cover_pct=float(hourly["cloud_cover"][index]),
        precipitation_mm=float(hourly["precipitation"][index]),
        source=source,
        cache_path=cache_path,
    )
