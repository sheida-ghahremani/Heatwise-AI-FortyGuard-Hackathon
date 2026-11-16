"""Route-segment UTCI maps with a shared continuous color scale."""
import folium
from branca.colormap import LinearColormap
from .routing import _edge_for_weight


def build_utci_route_map(graph, route, minimum, maximum):
    coordinates = [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in route.nodes]
    fmap = folium.Map(location=coordinates[0], tiles="OpenStreetMap", control_scale=True)
    scale = LinearColormap(
        ["#195ab6", "#278f9a", "#e3a316", "#c84024", "#8a183b"],
        vmin=minimum, vmax=maximum,
        caption="UTCI (°C)",
    )
    scale.width = 220
    scale.height = 50
    for u, v in zip(route.nodes, route.nodes[1:]):
        edge = _edge_for_weight(graph, u, v, route.routing_weight)
        geometry = edge.get("geometry")
        points = ([(y, x) for x, y in geometry.coords] if geometry is not None
                  else [(graph.nodes[u]["y"], graph.nodes[u]["x"]),
                        (graph.nodes[v]["y"], graph.nodes[v]["x"])])
        value = float(edge["utci_c"])
        folium.PolyLine(points, color="white", weight=8, opacity=1).add_to(fmap)
        folium.PolyLine(points, color=scale(value), weight=5, opacity=1,
                        tooltip=f"UTCI {value:.1f} °C").add_to(fmap)
    for label, point in [("A", coordinates[0]), ("B", coordinates[-1])]:
        folium.Marker(point, tooltip=label, icon=folium.DivIcon(
            html=f'<b style="background:white;border:1px solid black;padding:4px">{label}</b>'
        )).add_to(fmap)
    scale.add_to(fmap)
    return fmap


def render_utci_tab(graph, routes):
    import math
    import streamlit as st
    from streamlit_folium import st_folium

    st.subheader("Route-segment UTCI")
    st.caption("All routes use the same extent and continuous UTCI scale. Routing objectives use personalized PET. UTCI describes environmental thermal stress.")
    values = [float(_edge_for_weight(graph, u, v, r.routing_weight)["utci_c"])
              for r in routes for u, v in zip(r.nodes, r.nodes[1:])]
    minimum, maximum = math.floor(min(values)), math.ceil(max(values))
    if minimum == maximum:
        maximum += 1
    points = [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for r in routes for n in r.nodes]
    bounds = [[min(p[0] for p in points), min(p[1] for p in points)],
              [max(p[0] for p in points), max(p[1] for p in points)]]
    for column, route in zip(st.columns(len(routes)), routes):
        with column:
            st.markdown(f"**{route.name}**")
            st.caption(f"{route.duration_min:.1f} min · {route.distance_m/1000:.2f} km · mean UTCI {route.average_utci_c:.1f} °C")
            fmap = build_utci_route_map(graph, route, minimum, maximum)
            fmap.fit_bounds(bounds, padding=(20, 20))
            st_folium(fmap, height=560, use_container_width=True,
                      returned_objects=[], key="utci_" + route.name)
