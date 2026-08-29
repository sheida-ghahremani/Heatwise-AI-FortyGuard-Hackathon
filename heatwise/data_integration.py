from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point


def load_fortyguard_heatmap(path: str | Path) -> gpd.GeoDataFrame:
    """Load a cached FortyGuard response or its nested ``map_data`` GeoJSON."""
    payload = json.loads(Path(path).read_text())
    geojson = payload.get("result", payload).get("map_data", payload)
    frame = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
    if "average_temperature" not in frame:
        raise ValueError("Expected FortyGuard tcm tiles with average_temperature.")
    return frame


def assign_heatmap_to_graph(
    graph: nx.Graph,
    heatmap: gpd.GeoDataFrame,
    *,
    fallback_c: float | None = None,
) -> nx.Graph:
    """Assign each edge the nearest 100 m FortyGuard tile temperature."""
    # The graph was freshly loaded for this cached snapshot; mutate it in place
    # to avoid a second expanded NetworkX graph in memory.
    edge_data = []
    edge_xy = []
    for u, v, data in graph.edges(data=True):
        lat = (float(graph.nodes[u]["y"]) + float(graph.nodes[v]["y"])) / 2
        lon = (float(graph.nodes[u]["x"]) + float(graph.nodes[v]["x"])) / 2
        edge_xy.append((lon, lat))
        edge_data.append(data)

    # FortyGuard's TCM response is a regular 100 m polygon grid. A KD-tree over
    # tile centres gives the same cell assignment for network midpoints while
    # avoiding a large GeoDataFrame spatial join on memory-limited deployment.
    bounds = heatmap.geometry.bounds
    tile_xy = np.column_stack(
        ((bounds.minx.to_numpy() + bounds.maxx.to_numpy()) / 2,
         (bounds.miny.to_numpy() + bounds.maxy.to_numpy()) / 2)
    )
    temperatures = heatmap["average_temperature"].to_numpy(dtype=float)
    default = float(fallback_c if fallback_c is not None else heatmap["average_temperature"].median())
    if len(tile_xy):
        _, nearest = cKDTree(tile_xy).query(np.asarray(edge_xy, dtype=float), k=1)
        values = temperatures[nearest]
    else:
        values = np.full(len(edge_data), default)
    for value, data in zip(values, edge_data):
        data["temperature_c"] = default if not np.isfinite(value) else float(value)
    return graph


def apply_canopy_shade(graph: nx.Graph, canopy_path: str | Path, buffer_m: float = 12.0) -> nx.Graph:
    """Use intersection with local vegetation polygons as a transparent shade proxy."""
    graph = graph.copy()
    canopy = gpd.read_file(canopy_path).to_crs("EPSG:2277")
    rows, edge_data = [], []
    for index, (u, v, data) in enumerate(graph.edges(data=True)):
        lon = (float(graph.nodes[u]["x"]) + float(graph.nodes[v]["x"])) / 2
        lat = (float(graph.nodes[u]["y"]) + float(graph.nodes[v]["y"])) / 2
        rows.append({"edge_index": index, "geometry": Point(lon, lat)})
        edge_data.append(data)
    points = gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs("EPSG:2277")
    buffered = points.copy()
    buffered["geometry"] = buffered.buffer(buffer_m / 0.3048)
    joined = gpd.sjoin(buffered, canopy[["geometry"]], predicate="intersects", how="left")
    shaded = set(joined.loc[joined["index_right"].notna(), "edge_index"].astype(int))
    for index, data in enumerate(edge_data):
        data["shade_fraction"] = 0.75 if index in shaded else float(data.get("shade_fraction", 0.1))
    return graph
