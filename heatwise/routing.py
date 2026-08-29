from __future__ import annotations

from dataclasses import dataclass
import math

import networkx as nx
import numpy as np

from .models import UserProfile
from .risk import UTCI_NO_HEAT_STRESS_C, mean_radiant_temperature_c, utci_c, utci_stress_category


@dataclass
class RouteResult:
    name: str
    nodes: list[int]
    distance_m: float
    duration_min: float
    heat_score: float
    average_temp_c: float
    minimum_temp_c: float
    maximum_temp_c: float
    shade_pct: float
    average_wind_mps: float
    average_mrt_c: float
    average_utci_c: float
    average_pet_c: float
    pet_category: str
    pet_exposure_load: float
    utci_exposure_load: float
    risk: str


def pet_stress_category(pet_c: float) -> str:
    """Common temperate-climate PET thermal-stress classes (Matzarakis scale)."""
    if pet_c < 29:
        return "Slight or no heat stress"
    if pet_c < 35:
        return "Moderate heat stress"
    if pet_c < 41:
        return "Strong heat stress"
    return "Extreme heat stress"


def _bearing_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    lat1, lat2 = math.radians(y1), math.radians(y2)
    dlon = math.radians(x2 - x1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def prepare_graph(
    graph: nx.Graph,
    profile: UserProfile,
    humidity_pct: float,
    wind_mps: float,
    radiation_wm2: float = 0.0,
    direct_normal_radiation_wm2: float | None = None,
    diffuse_radiation_wm2: float = 0.0,
    cloud_cover_pct: float = 0.0,
    surface_albedo: float = 0.20,
    solar_elevation_deg: float = 45.0,
    solar_azimuth_deg: float = 180.0,
) -> nx.Graph:
    # All upstream enrichment is request-local, so mutate in place to keep the
    # production memory footprint below the Render free-tier limit.
    speed_mpm = profile.speed_kph * 1000 / 60
    edge_records = list(graph.edges(data=True))
    for u, v, data in edge_records:
        duration = float(data.get("length", 0.0)) / speed_mpm
        heading = _bearing_deg(
            float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"]),
            float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"]),
        )
        angle = abs((solar_azimuth_deg - heading + 180.0) % 360.0 - 180.0)
        sharp = min(angle, 180.0)
        temp = float(data.get("temperature_c", 35.0))
        shade = float(data.get("shade_fraction", 0.0))
        svf = float(data.get("sky_view_factor", 1.0))
        edge_albedo = float(data.get("surface_albedo", surface_albedo))
        direct = radiation_wm2 if direct_normal_radiation_wm2 is None else direct_normal_radiation_wm2
        mrt = mean_radiant_temperature_c(
            air_temperature_c=temp,
            direct_normal_radiation_wm2=direct,
            diffuse_radiation_wm2=diffuse_radiation_wm2,
            solar_elevation_deg=solar_elevation_deg,
            solar_horizontal_angle_deg=sharp,
            shade_fraction=shade,
            sky_view_factor=svf,
            relative_humidity_pct=humidity_pct,
            cloud_cover_pct=cloud_cover_pct,
            surface_albedo=edge_albedo,
        )
        utci_value = utci_c(
            temp_c=temp, mean_radiant_temp_c=mrt,
            humidity_pct=humidity_pct, wind_10m_mps=wind_mps,
        )
        data["duration_min"] = duration
        data["mean_radiant_temperature_c"] = mrt
        data["utci_c"] = utci_value
        data["utci_heat_cost"] = max(0.0, utci_value - UTCI_NO_HEAT_STRESS_C) * duration
        # UTCI supplies the network-wide thermal weight. PET is substantially
        # more expensive, so it is evaluated scientifically only on the three
        # candidate routes in _summarize rather than on every campus edge.
        data["heat_cost"] = data["utci_heat_cost"]
        data["balanced_cost"] = duration + 0.04 * data["heat_cost"]
        data["pet_rh_pct"] = humidity_pct
        data["pet_met"] = profile.metabolic_rate_met
        data["pet_clo"] = profile.summer_clothing_clo
        data["pet_age"] = profile.representative_age
    return graph


def _edge_for_weight(graph: nx.Graph, a: int, b: int, weight: str) -> dict:
    data = graph.get_edge_data(a, b)
    if graph.is_multigraph():
        return min(data.values(), key=lambda item: float(item.get(weight, 0.0)))
    return data


def _summarize(
    graph: nx.Graph, nodes: list[int], name: str, weight: str,
    profile: UserProfile | None = None,
) -> RouteResult:
    edges = [_edge_for_weight(graph, a, b, weight) for a, b in zip(nodes, nodes[1:])]
    distance = sum(float(e["length"]) for e in edges)
    duration = sum(float(e["duration_min"]) for e in edges)
    heat = sum(float(e["heat_cost"]) for e in edges)
    utci_load = sum(float(e["utci_heat_cost"]) for e in edges)
    if distance:
        temp = sum(float(e["temperature_c"]) * float(e["length"]) for e in edges) / distance
        shade = sum(float(e["shade_fraction"]) * float(e["length"]) for e in edges) / distance
        average_wind = sum(float(e.get("wind_mps", 0.0)) * float(e["length"]) for e in edges) / distance
        average_mrt = sum(float(e["mean_radiant_temperature_c"]) * float(e["duration_min"]) for e in edges) / duration
        average_utci = sum(float(e["utci_c"]) * float(e["duration_min"]) for e in edges) / duration
        # Report route-representative PET from duration/distance-weighted
        # environmental conditions. This retains the scientific PET model and
        # personalization while avoiding hundreds of iterative PET solves per
        # user interaction on small public instances.
        pet_profile = profile or UserProfile()
        from pythermalcomfort.models import pet_steady
        average_pet = float(np.asarray(pet_steady(
            tdb=temp,
            tr=average_mrt,
            v=max(0.1, average_wind),
            rh=float(edges[0]["pet_rh_pct"]),
            met=pet_profile.metabolic_rate_met,
            clo=pet_profile.summer_clothing_clo,
            position="standing",
            age=pet_profile.representative_age,
        ).pet, dtype=float).reshape(-1)[0])
    else:
        temp, shade, average_wind, average_mrt, average_utci, average_pet = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return RouteResult(
        name=name,
        nodes=nodes,
        distance_m=distance,
        duration_min=duration,
        heat_score=heat,
        average_temp_c=temp,
        minimum_temp_c=min((float(e["temperature_c"]) for e in edges), default=0.0),
        maximum_temp_c=max((float(e["temperature_c"]) for e in edges), default=0.0),
        shade_pct=100 * shade,
        average_wind_mps=average_wind,
        average_mrt_c=average_mrt,
        average_utci_c=average_utci,
        average_pet_c=average_pet,
        pet_category=pet_stress_category(average_pet),
        pet_exposure_load=max(0.0, average_pet - 29.0) * duration,
        utci_exposure_load=utci_load,
        risk=utci_stress_category(average_utci),
    )


def calculate_routes(
    graph: nx.Graph, origin: int, destination: int,
    profile: UserProfile | None = None,
) -> list[RouteResult]:
    specs = [
        ("Fastest", "duration_min"),
        ("Lowest Heat Risk", "heat_cost"),
        ("Balanced", "balanced_cost"),
    ]
    results: list[RouteResult] = []
    seen: set[tuple[int, ...]] = set()
    for name, weight in specs:
        nodes = nx.shortest_path(graph, origin, destination, weight=weight)
        key = tuple(nodes)
        if key not in seen:
            results.append(_summarize(graph, nodes, name, weight, profile))
            seen.add(key)
        else:
            # Keep all three decisions visible even when two objectives coincide.
            results.append(_summarize(graph, nodes, name, weight, profile))
    return results


def route_temperature_profile(graph: nx.Graph, route: RouteResult) -> list[dict[str, float]]:
    """Return cumulative-distance temperature/shade samples along a route."""
    profile: list[dict[str, float]] = []
    cumulative_m = 0.0
    for a, b in zip(route.nodes, route.nodes[1:]):
        edge = _edge_for_weight(graph, a, b, "duration_min")
        length = float(edge["length"])
        profile.append(
            {
                "distance_km": (cumulative_m + length / 2) / 1000,
                "temperature_c": float(edge["temperature_c"]),
                "shade_pct": 100 * float(edge.get("shade_fraction", 0.0)),
                "latitude": (float(graph.nodes[a]["y"]) + float(graph.nodes[b]["y"])) / 2,
                "longitude": (float(graph.nodes[a]["x"]) + float(graph.nodes[b]["x"])) / 2,
            }
        )
        cumulative_m += length
    return profile


def route_coordinates(graph: nx.Graph, route: RouteResult) -> list[tuple[float, float]]:
    return [(float(graph.nodes[n]["y"]), float(graph.nodes[n]["x"])) for n in route.nodes]
