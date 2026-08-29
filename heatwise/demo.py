from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True)
class DemoWeather:
    temperature_c: float = 36.0
    humidity_pct: float = 52.0
    wind_mps: float = 2.2


LANDMARKS = {
    "Kyle Field": (30.6102, -96.3404),
    "Evans Library": (30.6169, -96.3398),
    "MSC": (30.6123, -96.3413),
    "Zachry Engineering": (30.6210, -96.3408),
    "Reed Arena": (30.6055, -96.3466),
    "Bonfire Memorial": (30.6223, -96.3370),
}


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = math.radians(b[0] - a[0])
    dl = math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def build_demo_graph(size: int = 13) -> nx.Graph:
    """Create a connected campus-like network with spatial heat/shade variation."""
    graph = nx.Graph()
    lat0, lon0 = 30.6147, -96.3417
    step_lat, step_lon = 0.00135, 0.00155
    half = size // 2

    for row in range(size):
        for col in range(size):
            node = row * size + col
            lat = lat0 + (row - half) * step_lat
            lon = lon0 + (col - half) * step_lon
            graph.add_node(node, y=lat, x=lon)

    for row in range(size):
        for col in range(size):
            node = row * size + col
            for dr, dc in ((1, 0), (0, 1)):
                rr, cc = row + dr, col + dc
                if rr >= size or cc >= size:
                    continue
                other = rr * size + cc
                a = (graph.nodes[node]["y"], graph.nodes[node]["x"])
                b = (graph.nodes[other]["y"], graph.nodes[other]["x"])
                length = _haversine_m(a, b)
                mid_lat, mid_lon = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                # Cooler, shaded corridor on the north/west; hotter plaza southeast.
                # A north-south shaded promenade one block west of the direct route.
                green_corridor = max(0.0, 1.0 - abs(mid_lon + 96.34325) / 0.0035)
                north_trees = max(0.0, min(1.0, (mid_lat - 30.611) / 0.010))
                shade = min(0.95, 0.08 + 0.78 * green_corridor + 0.18 * north_trees)
                hot_plaza = math.exp(-(((mid_lat - 30.6105) / 0.0035) ** 2 + ((mid_lon + 96.3375) / 0.004) ** 2))
                temp = 36.0 + 2.8 * hot_plaza - 5.0 * shade
                graph.add_edge(
                    node,
                    other,
                    length=length,
                    temperature_c=temp,
                    shade_fraction=shade,
                )
    return graph


def nearest_node(graph: nx.Graph, coordinate: tuple[float, float]) -> int:
    lat, lon = coordinate
    return min(
        graph.nodes,
        key=lambda n: (graph.nodes[n]["y"] - lat) ** 2 + (graph.nodes[n]["x"] - lon) ** 2,
    )
