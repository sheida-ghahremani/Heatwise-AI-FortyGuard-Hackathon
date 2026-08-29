from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import numpy as np


def apply_lidar_sky_view_factor(
    graph: nx.Graph,
    canopy_height_path: str | Path,
    building_height_path: str | Path,
    *,
    observer_height_m: float = 1.1,
    radius_m: float = 80.0,
    azimuth_count: int = 16,
) -> tuple[nx.Graph, dict[str, float]]:
    """Estimate pedestrian sky-view factor from LiDAR horizon angles.

    SVF is calculated as the azimuthal mean of cos²(maximum horizon angle).
    Cells outside the available LiDAR footprint receive SVF=1 and are flagged.
    """
    import rasterio
    from rasterio.warp import transform as transform_coordinates

    # Mutate the per-request graph in place. Copying the full OSM graph here
    # doubles peak RAM and exceeds the 512 MB production container limit.
    with rasterio.open(canopy_height_path) as canopy_src, rasterio.open(building_height_path) as building_src:
        if canopy_src.crs != building_src.crs or canopy_src.transform != building_src.transform:
            raise ValueError("Canopy and building rasters must share a grid and CRS")
        canopy = canopy_src.read(1, masked=True).filled(0.0).astype("float32")
        buildings = building_src.read(1, masked=True).filled(0.0).astype("float32")
        obstacles = np.maximum(canopy, buildings)
        transform = canopy_src.transform
        raster_crs = canopy_src.crs
        height, width = obstacles.shape

    edges = list(graph.edges(data=True))
    lons = [(float(graph.nodes[u]["x"]) + float(graph.nodes[v]["x"])) / 2 for u, v, _ in edges]
    lats = [(float(graph.nodes[u]["y"]) + float(graph.nodes[v]["y"])) / 2 for u, v, _ in edges]
    xs, ys = transform_coordinates("EPSG:4326", raster_crs, lons, lats)
    inv = ~transform
    distances = np.arange(2.0, radius_m + 0.01, 4.0)
    azimuths = np.linspace(0.0, 2.0 * math.pi, azimuth_count, endpoint=False)
    covered = 0
    values: list[float] = []

    for x, y, (_, _, data) in zip(xs, ys, edges):
        col_f, row_f = inv * (x, y)
        row0, col0 = int(round(row_f)), int(round(col_f))
        if not (0 <= row0 < height and 0 <= col0 < width):
            svf = 1.0
            data["svf_source"] = "Outside LiDAR coverage"
        else:
            horizon_angles = []
            for azimuth in azimuths:
                sample_x = x + distances * math.sin(azimuth)
                sample_y = y + distances * math.cos(azimuth)
                cols, rows = inv * (sample_x, sample_y)
                rows = np.rint(rows).astype(int)
                cols = np.rint(cols).astype(int)
                valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
                if not valid.any():
                    horizon_angles.append(0.0)
                    continue
                obstacle_heights = obstacles[rows[valid], cols[valid]]
                angles = np.arctan2(np.maximum(0.0, obstacle_heights - observer_height_m), distances[valid])
                horizon_angles.append(float(np.max(angles, initial=0.0)))
            svf = float(np.mean(np.cos(np.asarray(horizon_angles)) ** 2))
            data["svf_source"] = "LiDAR 16-direction horizon"
            covered += 1
        data["sky_view_factor"] = min(1.0, max(0.0, svf))
        values.append(data["sky_view_factor"])

    return graph, {
        "mean_svf": float(np.mean(values)) if values else 1.0,
        "lidar_coverage_pct": 100.0 * covered / len(edges) if edges else 0.0,
        "azimuth_count": float(azimuth_count),
        "radius_m": float(radius_m),
    }
