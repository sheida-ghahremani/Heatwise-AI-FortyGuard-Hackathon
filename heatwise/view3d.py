"""Scientific 3D scene for explaining route-scale heat and shade."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pydeck as pdk
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from shapely.geometry import box, mapping

from .map_view import ROUTE_COLORS
from .routing import RouteResult, route_coordinates


def _route_bounds(graph: Any, routes: list[RouteResult], padding_deg: float = 0.0025) -> tuple[float, float, float, float]:
    points = [point for route in routes for point in route_coordinates(graph, route)]
    lats = [point[0] for point in points]
    lons = [point[1] for point in points]
    return (
        min(lons) - padding_deg,
        min(lats) - padding_deg,
        max(lons) + padding_deg,
        max(lats) + padding_deg,
    )


@lru_cache(maxsize=6)
def _load_buildings(path: str) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


def _height_from_levels(value: Any) -> float:
    try:
        return max(3.5, float(str(value).split(";")[0]) * 3.2)
    except (TypeError, ValueError):
        return 7.0


def building_records(building_geojson: str | Path, height_raster: str | Path, bounds: tuple[float, float, float, float]) -> list[dict]:
    """Clip OSM footprints to the route and attach LiDAR-derived heights."""
    buildings = _load_buildings(str(building_geojson))
    clipped = buildings[buildings.geometry.intersects(box(*bounds))].copy()
    if clipped.empty:
        return []

    records: list[dict] = []
    with rasterio.open(height_raster) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        for _, feature in clipped.iterrows():
            geometry = feature.geometry
            projected = gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs(src.crs).iloc[0]
            window = rasterio.windows.from_bounds(*projected.bounds, transform=src.transform)
            window = window.round_offsets().round_lengths()
            fallback = _height_from_levels(feature.get("building:levels"))
            lidar_height = 0.0
            try:
                data = src.read(1, window=window, masked=True)
                transform = src.window_transform(window)
                mask = geometry_mask([mapping(projected)], data.shape, transform, invert=True)
                values = np.asarray(data.filled(0.0))[mask]
                if values.size:
                    lidar_height = float(np.nanpercentile(values, 90))
            except (ValueError, rasterio.errors.WindowError):
                pass
            height = lidar_height if lidar_height >= 2.5 else fallback
            records.append({
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": {
                    "height_m": round(min(max(height, 3.0), 80.0), 1),
                    "name": feature.get("name") or feature.get("short_name") or "Building",
                    "height_source": "LiDAR 90th percentile" if lidar_height >= 2.5 else "OSM levels/default",
                },
            })
    return records


def canopy_records(canopy_raster: str | Path, bounds: tuple[float, float, float, float], *, max_points: int = 3500) -> list[dict]:
    """Downsample LiDAR canopy height into lightweight 3D crown columns."""
    west, south, east, north = bounds
    with rasterio.open(canopy_raster) as src:
        to_raster = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        left, bottom = to_raster.transform(west, south)
        right, top = to_raster.transform(east, north)
        window = rasterio.windows.from_bounds(min(left, right), min(bottom, top), max(left, right), max(bottom, top), src.transform)
        window = window.round_offsets().round_lengths()
        data = src.read(1, window=window, masked=True).filled(0.0)
        transform = src.window_transform(window)
        valid_count = int(np.count_nonzero(data >= 2.0))
        stride = max(2, int(np.ceil(np.sqrt(max(valid_count, 1) / max_points))))
        sampled = data[::stride, ::stride]
        rows, cols = np.where(sampled >= 2.0)
        if not len(rows):
            return []
        source_rows = rows * stride
        source_cols = cols * stride
        xs, ys = rasterio.transform.xy(transform, source_rows, source_cols, offset="center")
        to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        lons, lats = to_wgs84.transform(xs, ys)
        cell_size = float(abs(src.transform.a) * stride)
        return [
            {
                "position": [float(lon), float(lat)],
                "height_m": round(float(sampled[row, col]), 1),
                "radius_m": max(2.5, cell_size * 0.58),
            }
            for lon, lat, row, col in zip(lons, lats, rows, cols)
        ]


def network_records(graph: Any, bounds: tuple[float, float, float, float]) -> list[dict]:
    west, south, east, north = bounds
    records = []
    for u, v, data in graph.edges(data=True):
        a = [float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"])]
        b = [float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"])]
        lon, lat = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        if not (west <= lon <= east and south <= lat <= north):
            continue
        temperature = float(data.get("temperature_c", 32.0))
        shade = float(data.get("shade_fraction", 0.0))
        t = min(1.0, max(0.0, (temperature - 27.0) / 13.0))
        records.append({
            "path": [a, b],
            "temperature_c": round(temperature, 1),
            "shade_pct": round(100 * shade),
            "heat_color": [int(245 - 25 * t), int(190 - 135 * t), int(45 + 10 * t), 165],
            "shade_color": [15, 118, 110, int(35 + 180 * shade)],
        })
    return records


def build_3d_deck(
    graph: Any,
    routes: list[RouteResult],
    selected_name: str,
    building_geojson: str | Path,
    building_height: str | Path,
    canopy_height: str | Path,
    *,
    show_buildings: bool = True,
    show_canopy: bool = True,
    show_heat: bool = True,
    show_shade: bool = False,
    mapbox_token: str | None = None,
) -> tuple[pdk.Deck, dict[str, int]]:
    """Build a toggled 3D explanatory view centered on the selected routes."""
    bounds = _route_bounds(graph, routes)
    layers = []
    buildings = building_records(building_geojson, building_height, bounds) if show_buildings else []
    canopy = canopy_records(canopy_height, bounds) if show_canopy else []
    network = network_records(graph, bounds) if show_heat or show_shade else []

    if show_buildings and buildings:
        layers.append(pdk.Layer(
            "GeoJsonLayer", {"type": "FeatureCollection", "features": buildings}, id="lidar-buildings", pickable=True, extruded=True,
            get_fill_color=[100, 116, 139, 190], get_line_color=[226, 232, 240, 220],
            get_elevation="properties.height_m", elevation_scale=1, wireframe=True,
        ))
    if show_canopy and canopy:
        layers.append(pdk.Layer(
            "ColumnLayer", canopy, id="lidar-canopy", pickable=True, disk_resolution=8,
            get_position="position", get_elevation="height_m", radius=5,
            get_fill_color=[22, 163, 74, 150], elevation_scale=1,
        ))
    if show_heat and network:
        layers.append(pdk.Layer(
            "PathLayer", network, id="fortyguard-heat", pickable=True,
            get_path="path", get_color="heat_color", get_width=2.5, width_min_pixels=2,
        ))
    if show_shade and network:
        layers.append(pdk.Layer(
            "PathLayer", network, id="modeled-shade", pickable=True,
            get_path="path", get_color="shade_color", get_width=5, width_min_pixels=3,
        ))

    for route in routes:
        selected = route.name == selected_name
        color = ROUTE_COLORS[route.name].lstrip("#")
        rgb = [int(color[i:i + 2], 16) for i in (0, 2, 4)]
        path = [[lon, lat, 2.0] for lat, lon in route_coordinates(graph, route)]
        layers.append(pdk.Layer(
            "PathLayer", [{"path": path, "name": route.name}], id=f"route-{route.name}",
            pickable=True, get_path="path", get_color=rgb + [245 if selected else 95],
            get_width=7 if selected else 3, width_min_pixels=3,
        ))

    west, south, east, north = bounds
    view = pdk.ViewState(
        latitude=(south + north) / 2, longitude=(west + east) / 2,
        zoom=14.6, pitch=58, bearing=-18,
    )
    provider = "mapbox" if mapbox_token else "carto"
    style = "mapbox://styles/mapbox/light-v11" if mapbox_token else "light"
    deck = pdk.Deck(
        layers=layers, initial_view_state=view, map_provider=provider, map_style=style,
        api_keys={"mapbox": mapbox_token} if mapbox_token else None,
        tooltip={"html": "<b>{name}</b><br/>Height: {height_m} m<br/>Temperature: {temperature_c} °C<br/>Shade: {shade_pct}%"},
    )
    return deck, {"buildings": len(buildings), "canopy_samples": len(canopy), "network_segments": len(network)}
