from __future__ import annotations

from math import log
from pathlib import Path
from typing import Any

import numpy as np


def logarithmic_wind_speed(
    wind_reference_mps: float,
    *,
    target_height_m: float = 2.0,
    reference_height_m: float = 10.0,
    roughness_length_m: float = 0.1,
) -> float:
    """Adjust wind height using the neutral logarithmic wind profile."""
    z0 = min(max(float(roughness_length_m), 0.001), target_height_m * 0.45)
    ratio = log(target_height_m / z0) / log(reference_height_m / z0)
    return max(0.0, float(wind_reference_mps) * ratio)


def _window_density(mask: np.ndarray, rows: np.ndarray, cols: np.ndarray, radius: int) -> np.ndarray:
    """Return obstacle fraction in square neighborhoods using an integral image."""
    integral = np.pad(mask.astype("uint32"), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    r1 = np.clip(rows - radius, 0, mask.shape[0])
    r2 = np.clip(rows + radius + 1, 0, mask.shape[0])
    c1 = np.clip(cols - radius, 0, mask.shape[1])
    c2 = np.clip(cols + radius + 1, 0, mask.shape[1])
    occupied = integral[r2, c2] - integral[r1, c2] - integral[r2, c1] + integral[r1, c1]
    area = np.maximum((r2 - r1) * (c2 - c1), 1)
    return occupied / area


def apply_local_pedestrian_wind(
    graph: Any,
    wind_10m_mps: float,
    canopy_height_path: str | Path,
    building_height_path: str | Path,
    *,
    neighborhood_radius_m: float = 50.0,
) -> tuple[Any, dict[str, float]]:
    import rasterio
    from pyproj import Transformer

    """Assign edge-level 2 m wind using locally estimated aerodynamic roughness."""
    # The caller supplies a request-local graph; avoid another full graph copy.
    with rasterio.open(canopy_height_path) as canopy_src, rasterio.open(building_height_path) as building_src:
        if (
            canopy_src.shape != building_src.shape
            or canopy_src.transform != building_src.transform
            or canopy_src.crs != building_src.crs
        ):
            raise ValueError("Canopy and building rasters must use the same grid")
        canopy = ~np.ma.getmaskarray(canopy_src.read(1, masked=True))
        buildings = ~np.ma.getmaskarray(building_src.read(1, masked=True))
        obstacles = canopy | buildings
        records = list(graph.edges(data=True))
        lons = np.asarray([
            (float(graph.nodes[u]["x"]) + float(graph.nodes[v]["x"])) / 2 for u, v, _ in records
        ])
        lats = np.asarray([
            (float(graph.nodes[u]["y"]) + float(graph.nodes[v]["y"])) / 2 for u, v, _ in records
        ])
        tx = Transformer.from_crs("EPSG:4326", canopy_src.crs, always_xy=True)
        xs, ys = tx.transform(lons, lats)
        rows, cols = rasterio.transform.rowcol(canopy_src.transform, xs, ys)
        rows, cols = np.asarray(rows), np.asarray(cols)
        inside = (rows >= 0) & (rows < canopy_src.height) & (cols >= 0) & (cols < canopy_src.width)
        safe_rows = np.clip(rows, 0, canopy_src.height - 1)
        safe_cols = np.clip(cols, 0, canopy_src.width - 1)
        radius = max(1, int(round(neighborhood_radius_m / abs(canopy_src.transform.a))))
        density = _window_density(obstacles, safe_rows, safe_cols, radius)
        density[~inside] = 0.0

    # Empirical morphology classes expressed as aerodynamic roughness lengths.
    z0 = np.select(
        [density < 0.02, density < 0.10, density < 0.25],
        [0.03, 0.10, 0.30],
        default=0.80,
    )
    wind_2m = np.asarray([
        logarithmic_wind_speed(wind_10m_mps, roughness_length_m=value) for value in z0
    ])
    for index, (_, _, data) in enumerate(records):
        data["wind_10m_mps"] = float(wind_10m_mps)
        data["wind_mps"] = float(wind_2m[index])
        data["roughness_length_m"] = float(z0[index])
        data["obstacle_density_50m"] = float(density[index])
        data["wind_source"] = "Open-Meteo 10 m + local logarithmic roughness adjustment"

    return graph, {
        "mean_wind_2m_mps": float(wind_2m.mean()),
        "minimum_wind_2m_mps": float(wind_2m.min()),
        "maximum_wind_2m_mps": float(wind_2m.max()),
        "mean_roughness_length_m": float(z0.mean()),
    }
