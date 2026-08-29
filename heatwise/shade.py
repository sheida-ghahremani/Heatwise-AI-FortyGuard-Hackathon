from __future__ import annotations

from datetime import datetime
from math import acos, asin, atan2, cos, degrees, pi, radians, sin, tan
from pathlib import Path
from typing import Any

import numpy as np


def _project_shadow(
    height_raster: np.ma.MaskedArray,
    solar_elevation_deg: float,
    solar_azimuth_deg: float,
    resolution_m: float,
    observer_height_m: float,
    max_search_m: float,
    sample_step_m: float,
) -> np.ndarray:
    """Project raster obstacles onto target cells with vectorized translations."""
    heights = height_raster.filled(0)
    shadow = np.zeros(heights.shape, dtype=bool)
    azimuth = radians(solar_azimuth_deg)
    tangent = tan(radians(solar_elevation_deg))
    limit = min(max_search_m, max(0.0, (60.0 - observer_height_m) / max(tangent, 1e-6)))
    seen_offsets: set[tuple[int, int]] = set()
    height, width = heights.shape

    for distance in np.arange(0, limit + sample_step_m, sample_step_m):
        delta_row = int(round(-cos(azimuth) * distance / resolution_m))
        delta_col = int(round(sin(azimuth) * distance / resolution_m))
        if (delta_row, delta_col) in seen_offsets:
            continue
        seen_offsets.add((delta_row, delta_col))
        if delta_row >= 0:
            target_rows, source_rows = slice(0, height - delta_row), slice(delta_row, height)
        else:
            target_rows, source_rows = slice(-delta_row, height), slice(0, height + delta_row)
        if delta_col >= 0:
            target_cols, source_cols = slice(0, width - delta_col), slice(delta_col, width)
        else:
            target_cols, source_cols = slice(-delta_col, width), slice(0, width + delta_col)
        ray_height = observer_height_m + distance * tangent
        shadow[target_rows, target_cols] |= heights[source_rows, source_cols] >= ray_height
    return shadow


def solar_position(when: datetime, latitude: float, longitude: float) -> tuple[float, float]:
    """Approximate NOAA solar elevation and azimuth (degrees, clockwise from north)."""
    if when.tzinfo is None:
        raise ValueError("Solar position requires a timezone-aware datetime")
    local = when
    day = local.timetuple().tm_yday
    hour = local.hour + local.minute / 60 + local.second / 3600
    gamma = 2 * pi / 365 * (day - 1 + (hour - 12) / 24)
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * cos(gamma)
        - 0.032077 * sin(gamma)
        - 0.014615 * cos(2 * gamma)
        - 0.040849 * sin(2 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * cos(gamma)
        + 0.070257 * sin(gamma)
        - 0.006758 * cos(2 * gamma)
        + 0.000907 * sin(2 * gamma)
        - 0.002697 * cos(3 * gamma)
        + 0.00148 * sin(3 * gamma)
    )
    offset_hours = local.utcoffset().total_seconds() / 3600
    true_solar_minutes = (hour * 60 + equation_of_time + 4 * longitude - 60 * offset_hours) % 1440
    hour_angle = radians(true_solar_minutes / 4 - 180)
    lat = radians(latitude)
    cos_zenith = np.clip(
        sin(lat) * sin(declination) + cos(lat) * cos(declination) * cos(hour_angle), -1, 1
    )
    zenith = acos(float(cos_zenith))
    elevation = 90 - degrees(zenith)
    azimuth = (degrees(atan2(sin(hour_angle), cos(hour_angle) * sin(lat) - tan(declination) * cos(lat))) + 180) % 360
    return elevation, azimuth


def apply_dynamic_shade(
    graph: Any,
    canopy_height_path: str | Path,
    when: datetime,
    *,
    building_height_path: str | Path | None = None,
    observer_height_m: float = 1.5,
    max_search_m: float = 100.0,
    sample_step_m: float = 2.0,
) -> tuple[Any, dict[str, float]]:
    import rasterio
    from pyproj import Transformer

    """Assign time-dependent shade by ray testing edge midpoints against a CHM.

    For each edge midpoint, samples are taken from that point toward the sun.
    A canopy cell shades the midpoint when its height exceeds the solar ray at
    that distance. Values are fractions used by the routing risk model.
    """
    # The caller supplies a request-local graph; avoid another full graph copy.
    latitude = float(np.mean([float(data["y"]) for _, data in graph.nodes(data=True)]))
    longitude = float(np.mean([float(data["x"]) for _, data in graph.nodes(data=True)]))
    elevation, azimuth = solar_position(when, latitude, longitude)

    if elevation <= 0:
        for *_, data in graph.edges(data=True):
            data["shade_fraction"] = 0.0
        return graph, {"solar_elevation_deg": elevation, "solar_azimuth_deg": azimuth, "shaded_edges_pct": 0.0}

    with rasterio.open(canopy_height_path) as src:
        chm = src.read(1, masked=True)
        building_src = rasterio.open(building_height_path) if building_height_path else None
        buildings = building_src.read(1, masked=True) if building_src else None
        if building_src and (
            building_src.shape != src.shape
            or building_src.transform != src.transform
            or building_src.crs != src.crs
        ):
            building_src.close()
            raise ValueError("Building and canopy height rasters must use the same grid and CRS")
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        resolution = float(abs(src.transform.a))
        tree_shadow = _project_shadow(
            chm, elevation, azimuth, resolution, observer_height_m, max_search_m, sample_step_m
        )
        building_shadow = (
            _project_shadow(
                buildings, elevation, azimuth, resolution, observer_height_m, max_search_m, sample_step_m
            )
            if buildings is not None
            else np.zeros(chm.shape, dtype=bool)
        )
        tree_shaded_count = 0
        building_shaded_count = 0

        edge_records = list(graph.edges(data=True))
        longitudes = np.asarray([
            (float(graph.nodes[u]["x"]) + float(graph.nodes[v]["x"])) / 2
            for u, v, _ in edge_records
        ])
        latitudes = np.asarray([
            (float(graph.nodes[u]["y"]) + float(graph.nodes[v]["y"])) / 2
            for u, v, _ in edge_records
        ])
        xs, ys = transformer.transform(longitudes, latitudes)
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows, cols = np.asarray(rows), np.asarray(cols)
        inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)

        for index, (_, _, data) in enumerate(edge_records):
            tree_obstruction = bool(inside[index] and tree_shadow[rows[index], cols[index]])
            building_obstruction = bool(inside[index] and building_shadow[rows[index], cols[index]])
            if building_obstruction:
                data["shade_fraction"] = 0.95
                data["shade_source"] = "Building LiDAR height + solar geometry"
            elif tree_obstruction:
                data["shade_fraction"] = 0.9
                data["shade_source"] = "Tree LiDAR height + solar geometry"
            else:
                data["shade_fraction"] = 0.05
                data["shade_source"] = "Open sun"
            tree_shaded_count += int(tree_obstruction and not building_obstruction)
            building_shaded_count += int(building_obstruction)

        if building_src:
            building_src.close()

    edge_count = max(graph.number_of_edges(), 1)
    return graph, {
        "solar_elevation_deg": elevation,
        "solar_azimuth_deg": azimuth,
        "tree_shaded_edges_pct": 100 * tree_shaded_count / edge_count,
        "building_shaded_edges_pct": 100 * building_shaded_count / edge_count,
        "shaded_edges_pct": 100 * (tree_shaded_count + building_shaded_count) / edge_count,
    }


def apply_dynamic_tree_shade(
    graph: Any,
    canopy_height_path: str | Path,
    when: datetime,
    **kwargs,
) -> tuple[Any, dict[str, float]]:
    """Backward-compatible tree-only wrapper."""
    return apply_dynamic_shade(graph, canopy_height_path, when, **kwargs)
