from __future__ import annotations

import networkx as nx


SURFACE_ALBEDO = {
    "asphalt": 0.12,
    "concrete": 0.30,
    "concrete:plates": 0.30,
    "paving_stones": 0.25,
    "sett": 0.25,
    "brick": 0.30,
    "gravel": 0.20,
    "fine_gravel": 0.20,
    "compacted": 0.20,
    "ground": 0.17,
    "dirt": 0.17,
    "wood": 0.35,
}


def apply_osm_surface_albedo(graph: nx.Graph, default: float = 0.20) -> tuple[nx.Graph, dict[str, float]]:
    """Assign literature-range short-wave albedo by OSM path surface."""
    # Enrich the request-local graph without duplicating the full OSM network.
    mapped = 0
    values: list[float] = []
    for _, _, data in graph.edges(data=True):
        surface = data.get("surface", "")
        if isinstance(surface, (list, tuple)):
            surface = surface[0] if surface else ""
        key = str(surface).lower()
        value = SURFACE_ALBEDO.get(key, default)
        mapped += int(key in SURFACE_ALBEDO)
        data["surface_albedo"] = value
        data["albedo_source"] = f"OSM surface: {key}" if key in SURFACE_ALBEDO else "Urban default"
        values.append(value)
    return graph, {
        "mean_albedo": sum(values) / len(values) if values else default,
        "osm_surface_coverage_pct": 100.0 * mapped / len(values) if values else 0.0,
    }
