from __future__ import annotations

import base64
import hashlib
import json
import math
import zlib
from pathlib import Path

import networkx as nx
import numpy as np

from .shade import solar_position


def _edge_records(graph: nx.Graph):
    if graph.is_multigraph():
        return list(graph.edges(keys=True, data=True))
    return [(u, v, 0, data) for u, v, data in graph.edges(data=True)]


def _edge_fingerprint(records) -> str:
    payload = "\n".join(f"{u}|{v}|{key}" for u, v, key, _ in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_horizon_profile(
    graph: nx.Graph,
    canopy_height_path: str | Path,
    building_height_path: str | Path,
    output_path: str | Path,
    *,
    observer_height_m: float = 1.5,
    radius_m: float = 100.0,
    azimuth_count: int = 16,
    sample_step_m: float = 4.0,
) -> Path:
    import rasterio
    from rasterio.warp import transform as transform_coordinates

    """Precompute directional LiDAR horizon angles for lightweight hourly shade."""
    with rasterio.open(canopy_height_path) as canopy_src, rasterio.open(building_height_path) as building_src:
        if canopy_src.crs != building_src.crs or canopy_src.transform != building_src.transform:
            raise ValueError("Canopy and building rasters must share a grid and CRS")
        canopy = canopy_src.read(1, masked=True).filled(0.0).astype("float32")
        buildings = building_src.read(1, masked=True).filled(0.0).astype("float32")
        transform = canopy_src.transform
        raster_crs = canopy_src.crs
        height, width = canopy.shape

    records = _edge_records(graph)
    lons = [(float(graph.nodes[u]["x"]) + float(graph.nodes[v]["x"])) / 2 for u, v, _, _ in records]
    lats = [(float(graph.nodes[u]["y"]) + float(graph.nodes[v]["y"])) / 2 for u, v, _, _ in records]
    xs, ys = transform_coordinates("EPSG:4326", raster_crs, lons, lats)
    inv = ~transform
    distances = np.arange(sample_step_m, radius_m + 0.01, sample_step_m)
    azimuths_deg = np.linspace(0.0, 360.0, azimuth_count, endpoint=False)
    tree_angles = np.zeros((len(records), azimuth_count), dtype=np.uint8)
    building_angles = np.zeros_like(tree_angles)
    covered = np.zeros(len(records), dtype=np.uint8)

    for edge_index, (x, y) in enumerate(zip(xs, ys)):
        col_f, row_f = inv * (x, y)
        if not (0 <= round(row_f) < height and 0 <= round(col_f) < width):
            continue
        covered[edge_index] = 1
        for azimuth_index, azimuth_deg in enumerate(azimuths_deg):
            azimuth = math.radians(float(azimuth_deg))
            sample_x = x + distances * math.sin(azimuth)
            sample_y = y + distances * math.cos(azimuth)
            cols, rows = inv * (sample_x, sample_y)
            rows = np.rint(rows).astype(int)
            cols = np.rint(cols).astype(int)
            valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            if not valid.any():
                continue
            valid_distances = distances[valid]
            tree = np.arctan2(
                np.maximum(0.0, canopy[rows[valid], cols[valid]] - observer_height_m), valid_distances
            )
            building = np.arctan2(
                np.maximum(0.0, buildings[rows[valid], cols[valid]] - observer_height_m), valid_distances
            )
            tree_angles[edge_index, azimuth_index] = round(np.degrees(tree).max(initial=0.0))
            building_angles[edge_index, azimuth_index] = round(np.degrees(building).max(initial=0.0))

    def encode(array: np.ndarray) -> str:
        return base64.b64encode(zlib.compress(array.tobytes(), level=9)).decode("ascii")

    payload = {
        "version": 1,
        "method": "LiDAR directional horizon; observer 1.5 m; 16 azimuths; 100 m radius",
        "edge_count": len(records),
        "edge_fingerprint": _edge_fingerprint(records),
        "azimuths_deg": azimuths_deg.tolist(),
        "tree_horizon_u8_zlib": encode(tree_angles),
        "building_horizon_u8_zlib": encode(building_angles),
        "covered_u8_zlib": encode(covered),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return output


def apply_horizon_profile(
    graph: nx.Graph,
    profile_path: str | Path,
    when,
    latitude: float,
    longitude: float,
) -> tuple[nx.Graph, dict[str, float]]:
    """Apply hourly tree/building shade and SVF from a precomputed horizon profile."""
    payload = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    records = _edge_records(graph)
    if payload["edge_count"] != len(records) or payload["edge_fingerprint"] != _edge_fingerprint(records):
        raise ValueError("Horizon profile does not match the selected routing graph")

    def decode(name: str, shape) -> np.ndarray:
        raw = zlib.decompress(base64.b64decode(payload[name]))
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape)

    count = len(records)
    azimuths = np.asarray(payload["azimuths_deg"], dtype=float)
    tree = decode("tree_horizon_u8_zlib", (count, len(azimuths))).astype(float)
    buildings = decode("building_horizon_u8_zlib", (count, len(azimuths))).astype(float)
    covered = decode("covered_u8_zlib", (count,)).astype(bool)
    combined = np.maximum(tree, buildings)
    svf = np.mean(np.cos(np.radians(combined)) ** 2, axis=1)

    elevation, azimuth = solar_position(when, latitude, longitude)
    bin_width = 360.0 / len(azimuths)
    position = (azimuth % 360.0) / bin_width
    lower = int(math.floor(position)) % len(azimuths)
    upper = (lower + 1) % len(azimuths)
    fraction = position - math.floor(position)
    tree_horizon = tree[:, lower] * (1 - fraction) + tree[:, upper] * fraction
    building_horizon = buildings[:, lower] * (1 - fraction) + buildings[:, upper] * fraction
    building_shaded = covered & (elevation > 0) & (elevation <= building_horizon)
    tree_shaded = covered & ~building_shaded & (elevation > 0) & (elevation <= tree_horizon)

    for index, (*_, data) in enumerate(records):
        data["sky_view_factor"] = float(svf[index]) if covered[index] else 1.0
        data["svf_source"] = "LiDAR 16-direction horizon" if covered[index] else "Outside LiDAR coverage"
        if elevation <= 0:
            data["shade_fraction"] = 0.0
            data["shade_source"] = "Nighttime (no direct solar load)"
        elif building_shaded[index]:
            data["shade_fraction"] = 0.95
            data["shade_source"] = "Building LiDAR horizon + hourly solar geometry"
        elif tree_shaded[index]:
            data["shade_fraction"] = 0.90
            data["shade_source"] = "Tree LiDAR horizon + hourly solar geometry"
        else:
            data["shade_fraction"] = 0.05
            data["shade_source"] = "Open sun"

    denominator = max(count, 1)
    return graph, {
        "solar_elevation_deg": float(elevation),
        "solar_azimuth_deg": float(azimuth),
        "tree_shaded_edges_pct": 100.0 * float(tree_shaded.sum()) / denominator,
        "building_shaded_edges_pct": 100.0 * float(building_shaded.sum()) / denominator,
        "shaded_edges_pct": 100.0 * float((tree_shaded | building_shaded).sum()) / denominator,
        "mean_svf": float(svf[covered].mean()) if covered.any() else 1.0,
        "lidar_coverage_pct": 100.0 * float(covered.sum()) / denominator,
    }
