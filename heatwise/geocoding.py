from __future__ import annotations

from dataclasses import dataclass

import requests


MAPBOX_FORWARD_URL = "https://api.mapbox.com/search/searchbox/v1/forward"
COLLEGE_STATION_BBOX = "-96.38769849,30.51649795,-96.19658949,30.65843070"


@dataclass(frozen=True)
class GeocodingResult:
    label: str
    latitude: float
    longitude: float


def search_college_station(
    query: str, access_token: str, limit: int = 5, *, bbox: str = COLLEGE_STATION_BBOX,
    proximity: str = "-96.3365,30.6187",
) -> list[GeocodingResult]:
    """Search addresses and POIs with Mapbox Search Box, constrained to College Station."""
    clean_query = " ".join(query.strip().split())
    if not clean_query:
        return []
    response = requests.get(
        MAPBOX_FORWARD_URL,
        params={
            "q": clean_query,
            "access_token": access_token,
            "bbox": bbox,
            "proximity": proximity,
            "country": "us",
            "limit": min(10, max(1, limit)),
        },
        timeout=15,
    )
    response.raise_for_status()
    results: list[GeocodingResult] = []
    for feature in response.json().get("features", []):
        properties = feature.get("properties", {})
        coordinates = properties.get("coordinates", {})
        routable_points = coordinates.get("routable_points") or []
        preferred_point = routable_points[0] if routable_points else coordinates
        lon = preferred_point.get("longitude")
        lat = preferred_point.get("latitude")
        if lon is None or lat is None:
            geometry = feature.get("geometry", {}).get("coordinates", [])
            if len(geometry) >= 2:
                lon, lat = geometry[:2]
        if lon is None or lat is None:
            continue
        name = properties.get("name") or feature.get("name") or clean_query
        context = properties.get("full_address") or properties.get("place_formatted") or "College Station, Texas"
        label = name if context.lower() in name.lower() else f"{name} — {context}"
        results.append(GeocodingResult(label=label, latitude=float(lat), longitude=float(lon)))
    return results
