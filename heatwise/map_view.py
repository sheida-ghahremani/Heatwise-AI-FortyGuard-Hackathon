from __future__ import annotations

import folium
from branca.colormap import LinearColormap

from .routing import RouteResult, route_coordinates, route_temperature_profile


ROUTE_COLORS = {
    "Fastest": "#ef4444",
    "Lowest Heat Risk": "#10b981",
    "Balanced": "#3b82f6",
}

MAPBOX_STYLES = {
    "Streets": "streets-v12",
    "Outdoors": "outdoors-v12",
    "Satellite": "satellite-streets-v12",
}


def build_map(
    graph,
    routes: list[RouteResult],
    selected_name: str,
    boundary: dict | None = None,
    *,
    mapbox_token: str | None = None,
    mapbox_style: str = "Streets",
    origin_coordinate: tuple[float, float] | None = None,
    destination_coordinate: tuple[float, float] | None = None,
) -> folium.Map:
    selected = next(route for route in routes if route.name == selected_name)
    coords = route_coordinates(graph, selected)
    center = [sum(p[0] for p in coords) / len(coords), sum(p[1] for p in coords) / len(coords)]
    fmap = folium.Map(location=center, zoom_start=15, tiles=None, control_scale=True)
    if mapbox_token:
        style_id = MAPBOX_STYLES.get(mapbox_style, MAPBOX_STYLES["Streets"])
        folium.TileLayer(
            tiles=(
                f"https://api.mapbox.com/styles/v1/mapbox/{style_id}/tiles/512/"
                f"{{z}}/{{x}}/{{y}}?access_token={mapbox_token}"
            ),
            attr="© Mapbox © OpenStreetMap",
            name=f"Mapbox {mapbox_style}",
            tile_size=512,
            zoom_offset=-1,
            max_zoom=22,
            overlay=False,
            control=True,
        ).add_to(fmap)
    else:
        folium.TileLayer("CartoDB positron", name="CartoDB Positron").add_to(fmap)

    # A token-free fallback is useful if the Mapbox account or network is unavailable.
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap fallback", show=False).add_to(fmap)

    if boundary:
        folium.GeoJson(
            boundary,
            name="College Station boundary",
            style_function=lambda _feature: {
                "color": "#0f766e",
                "weight": 2,
                "fillOpacity": 0.02,
                "dashArray": "6 5",
            },
            tooltip="Official College Station boundary",
        ).add_to(fmap)

    route_lats = [p[0] for route in routes for p in route_coordinates(graph, route)]
    route_lons = [p[1] for route in routes for p in route_coordinates(graph, route)]
    south, north = min(route_lats) - 0.003, max(route_lats) + 0.003
    west, east = min(route_lons) - 0.003, max(route_lons) + 0.003
    visible_edges = []
    for a, b, data in graph.edges(data=True):
        pa = (float(graph.nodes[a]["y"]), float(graph.nodes[a]["x"]))
        pb = (float(graph.nodes[b]["y"]), float(graph.nodes[b]["x"]))
        mid_lat, mid_lon = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
        if not (south <= mid_lat <= north and west <= mid_lon <= east):
            continue
        visible_edges.append((pa, pb, float(data["temperature_c"])))

    mapped_temperatures = [temperature for _, _, temperature in visible_edges]
    if mapped_temperatures:
        scale_min, scale_max = min(mapped_temperatures), max(mapped_temperatures)
    else:
        graph_temperatures = [float(data["temperature_c"]) for *_, data in graph.edges(data=True)]
        scale_min, scale_max = min(graph_temperatures), max(graph_temperatures)
    if scale_min == scale_max:
        scale_min -= 0.1
        scale_max += 0.1
    heat_scale = LinearColormap(
        ["#fde68a", "#fb923c", "#dc2626"],
        vmin=scale_min,
        vmax=scale_max,
        caption="Mapped FortyGuard air temperature (°C)",
    )
    for pa, pb, edge_temperature in visible_edges:
        folium.PolyLine([pa, pb], color=heat_scale(edge_temperature), weight=2, opacity=0.38).add_to(fmap)
    heat_scale.add_to(fmap)

    for route in routes:
        visible = route.name == selected_name
        folium.PolyLine(
            route_coordinates(graph, route),
            color=ROUTE_COLORS[route.name],
            weight=7 if visible else 4,
            opacity=0.95 if visible else 0.35,
            tooltip=(f"{route.name}: {route.duration_min:.1f} min · UTCI {route.average_utci_c:.1f} °C · "
                     f"activity PET {route.average_pet_c:.1f} °C · {route.pet_category}"),
        ).add_to(fmap)

    display_origin = origin_coordinate or coords[0]
    display_destination = destination_coordinate or coords[-1]
    folium.Marker(display_origin, tooltip="Selected origin", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker(display_destination, tooltip="Selected destination", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
    if display_origin != coords[0]:
        folium.PolyLine([display_origin, coords[0]], color="#16a34a", weight=3, dash_array="5 5", opacity=0.8,
                        tooltip="Walk from the selected place to the nearest routable path").add_to(fmap)
    if display_destination != coords[-1]:
        folium.PolyLine([coords[-1], display_destination], color="#dc2626", weight=3, dash_array="5 5", opacity=0.8,
                        tooltip="Walk from the routable path to the selected place").add_to(fmap)
    profile = route_temperature_profile(graph, selected)
    if profile:
        hottest = max(profile, key=lambda point: point["temperature_c"])
        folium.CircleMarker(
            [hottest["latitude"], hottest["longitude"]],
            radius=7,
            color="#7f1d1d",
            fill=True,
            fill_color="#ef4444",
            fill_opacity=0.9,
            tooltip=f"Hottest route segment: {hottest['temperature_c']:.1f} °C",
        ).add_to(fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap
