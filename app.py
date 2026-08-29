from __future__ import annotations

from pathlib import Path
import json
import os
from datetime import datetime, timedelta
import time

import requests
import streamlit as st
from streamlit_js_eval import get_geolocation
from streamlit_searchbox import st_searchbox
from dotenv import load_dotenv

from heatwise.geocoding import GeocodingResult, search_college_station
from heatwise.live_data import COLLEGE_STATION_TZ, current_hour, fetch_hourly_snapshot, latest_cached_snapshot
from heatwise.models import Activity, AgeGroup, Clothing, UserProfile
from heatwise.weather import fetch_hourly_weather


MAPBOX_STYLES = {
    "Streets": "streets-v12",
    "Outdoors": "outdoors-v12",
    "Satellite": "satellite-streets-v12",
}
LANDMARKS = {
    "Kyle Field": (30.6102, -96.3404),
    "Evans Library": (30.6169, -96.3398),
    "MSC": (30.6123, -96.3413),
    "Zachry Engineering": (30.6210, -96.3408),
    "Reed Arena": (30.6055, -96.3466),
    "Bonfire Memorial": (30.6223, -96.3370),
}


def routing_graph_exists(path: str | Path) -> bool:
    graph_path = Path(path)
    return graph_path.exists() or graph_path.with_suffix(".pickle").exists()


load_dotenv(".env")
st.set_page_config(page_title="HeatWise AI", page_icon="☀️", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {background:#ffffff; border:1px solid #e5e7eb; padding:14px; border-radius:14px;}
    [data-testid="stMetricLabel"] p {font-size:.78rem !important; line-height:1.15 !important; white-space:normal !important;}
    [data-testid="stMetricValue"] {font-size:1.55rem !important; line-height:1.15 !important;}
    .hero {padding:22px 26px;border-radius:18px;background:linear-gradient(120deg,#083344,#0f766e);color:white;margin:2px 0 16px;text-align:center}
    .hero h1 {margin:0;font-size:2.2rem}.hero p {margin:7px 0 0;color:#ccfbf1}
    .notice {background:#fffbeb;border-left:4px solid #f59e0b;padding:10px 14px;border-radius:8px;margin-bottom:12px}
    .api-badge {display:inline-block;margin-top:12px;padding:6px 11px;border-radius:999px;background:#ecfdf5;color:#065f46;font-weight:700;font-size:.84rem}
    .personal-risk {padding:14px 18px;border-radius:14px;background:#fff7ed;border:1px solid #fdba74;margin:8px 0 14px}
    .personal-risk b {color:#9a3412;font-size:1.05rem}
    </style>
    """,
    unsafe_allow_html=True,
)
brand_left, brand_center, brand_right = st.columns([1.05, 5.4, 1.05], vertical_alignment="center")
with brand_left:
    st.image("assets/branding/tamu_logo.png", width=145)
with brand_center:
    st.markdown(
        """
        <div class="hero">
          <h1>HeatWise AI</h1>
          <p>Personalized thermal-risk-aware routes for safer movement in extreme heat</p>
          <span class="api-badge">Powered by FortyGuard Temperature API®</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with brand_right:
    st.image("assets/branding/uch_lab_logo.png", width=145)
AREA_CONFIG = {
    "TAMU Campus": {
        "key": "tamu", "graph": Path("data/osm/TAMU_Full_walk.graphml"),
        "bike_graph": Path("data/osm/TAMU_Full_bike.graphml"),
        "prepared_graph": Path("data/osm/TAMU_Full_walk_prepared.graphml"),
        "prepared_bike_graph": Path("data/osm/TAMU_Full_bike_prepared.graphml"),
        "walk_horizon": Path("data/lidar/TAMU_Full_walk_horizon.json"),
        "bike_horizon": Path("data/lidar/TAMU_Full_bike_horizon.json"),
        "canopy": Path("data/lidar/TAMU_Full_CanopyHeight_2m.tif"),
        "buildings": Path("data/lidar/TAMU_Full_BuildingHeight_2m.tif"),
        "building_footprints": Path("data/osm/TAMU_Full_buildings.geojson"),
        "latitude": 30.614, "longitude": -96.350,
        "bbox": "-96.375,30.592,-96.326,30.636", "proximity": "-96.350,30.614",
        "origin_default": "Kyle Field", "destination_default": "Zachry Engineering Building",
        "origin_coord": (30.6103, -96.3404), "destination_coord": (30.6211, -96.3409),
    },
    "RELLIS Campus": {
        "key": "rellis", "graph": Path("data/osm/RELLIS_walk.graphml"),
        "bike_graph": Path("data/osm/RELLIS_bike.graphml"),
        "canopy": Path("data/lidar/RELLIS_CanopyHeight_2m.tif"),
        "buildings": Path("data/lidar/RELLIS_BuildingHeight_2m.tif"),
        "building_footprints": Path("data/osm/RELLIS_buildings.geojson"),
        "latitude": 30.638, "longitude": -96.475,
        "bbox": "-96.4986,30.6258,-96.4512,30.6505", "proximity": "-96.475,30.638",
        "origin_default": "RELLIS Academic Complex", "destination_default": "Blinn RELLIS Administration Building",
        "origin_coord": (30.6350, -96.4680), "destination_coord": (30.6340, -96.4695),
    },
}
CITY_BOUNDARY = Path("data/boundaries/CollegeStation.geojson")


@st.cache_resource(max_entries=1)
def load_live_graph(path: str):
    from heatwise.graph_io import load_routing_graph
    return load_routing_graph(path)


@st.cache_resource(max_entries=2, show_spinner=False)
def load_snapshot_graph(graph_path: str, heatmap_path: str, canopy_path: str, building_path: str):
    from heatwise.data_integration import assign_heatmap_to_graph, load_fortyguard_heatmap
    from heatwise.graph_io import load_routing_graph
    from heatwise.svf import apply_lidar_sky_view_factor
    started = time.perf_counter()
    print(f"[snapshot] start graph={Path(graph_path).name}", flush=True)
    graph = load_routing_graph(graph_path)
    print(f"[snapshot] graph loaded edges={len(graph.edges)} elapsed={time.perf_counter() - started:.2f}s", flush=True)
    heatmap = load_fortyguard_heatmap(heatmap_path)
    print(f"[snapshot] heatmap loaded tiles={len(heatmap)} elapsed={time.perf_counter() - started:.2f}s", flush=True)
    graph = assign_heatmap_to_graph(graph, heatmap)
    print(f"[snapshot] temperatures assigned elapsed={time.perf_counter() - started:.2f}s", flush=True)
    svf_values = [float(data.get("sky_view_factor", 1.0)) for *_, data in graph.edges(data=True)]
    has_prepared_svf = any("sky_view_factor" in data for *_, data in graph.edges(data=True))
    svf_meta = {
        "mean_svf": sum(svf_values) / len(svf_values) if svf_values else 1.0,
        "lidar_coverage_pct": 100.0 if has_prepared_svf else 0.0,
    }
    if not has_prepared_svf and Path(canopy_path).exists() and Path(building_path).exists():
        graph, svf_meta = apply_lidar_sky_view_factor(graph, canopy_path, building_path)
    print(f"[snapshot] complete elapsed={time.perf_counter() - started:.2f}s", flush=True)
    return graph, float(heatmap["average_temperature"].mean()), svf_meta


@st.cache_data
def load_boundary(path: str):
    return json.loads(Path(path).read_text())


@st.cache_data(ttl=3600, show_spinner=False)
def cached_location_search(query: str, access_token: str, bbox: str, proximity: str):
    """Return display/result pairs for the live Mapbox autocomplete widgets."""
    if len(query.strip()) < 2:
        return []
    response = requests.get(
        "https://api.mapbox.com/search/searchbox/v1/forward",
        params={"q": query.strip(), "access_token": access_token, "bbox": bbox,
                "proximity": proximity, "country": "us", "limit": 8}, timeout=15,
    )
    response.raise_for_status()
    options = []
    for feature in response.json().get("features", []):
        props = feature.get("properties", {})
        coords = props.get("coordinates", {})
        point = (coords.get("routable_points") or [coords])[0]
        lon, lat = point.get("longitude"), point.get("latitude")
        if lon is None or lat is None:
            lon, lat = feature.get("geometry", {}).get("coordinates", [None, None])[:2]
        if lon is None or lat is None:
            continue
        name = props.get("name") or query.strip()
        context = props.get("full_address") or props.get("place_formatted") or "Texas"
        from heatwise.geocoding import GeocodingResult
        item = GeocodingResult(f"{name} — {context}", float(lat), float(lon))
        options.append((item.label, item))
    return options


live_available = all(config["graph"].exists() and config["bike_graph"].exists() for config in AREA_CONFIG.values())

with st.sidebar:
    st.header("Plan your route")
    source_options = ["Live College Station data", "Prototype scenario"] if live_available else ["Prototype scenario"]
    data_source = st.selectbox("Data source", source_options)
    study_area = st.selectbox("Study area", list(AREA_CONFIG)) if data_source == "Live College Station data" else "TAMU Campus"
    area_config = AREA_CONFIG[study_area]
    if st.session_state.get("active_study_area") != study_area:
        for key in (
            "origin_search_result", "destination_search_result",
            "origin_location_mode", "destination_location_mode",
            "current_origin_coordinate", "current_destination_coordinate",
            "geolocation_target",
        ):
            st.session_state.pop(key, None)
        st.session_state.active_study_area = study_area
    LIVE_GRAPH = area_config["graph"]
    CANOPY_HEIGHT = area_config["canopy"]
    BUILDING_HEIGHT = area_config["buildings"]
    mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN")
    if mapbox_token:
        map_style = st.selectbox("Map style", list(MAPBOX_STYLES))
        st.caption("Basemap: Mapbox · Routing engine: OSMnx")
    else:
        map_style = "Streets"
    if data_source == "Live College Station data" and mapbox_token:
        st.subheader("Search locations")
        st.caption("Start typing an address, campus building, or place and choose a Mapbox suggestion.")

        def mapbox_suggestions(query: str):
            try:
                return cached_location_search(query, mapbox_token, area_config["bbox"], area_config["proximity"])
            except Exception:
                return []

        def default_location_option(name: str, coordinate: tuple[float, float]):
            """Keep the app usable during a temporary Mapbox outage."""
            try:
                options = cached_location_search(name, mapbox_token, area_config["bbox"], area_config["proximity"])
                if options:
                    return options
            except Exception:
                pass
            fallback = GeocodingResult(name, float(coordinate[0]), float(coordinate[1]))
            return [(name, fallback)]

        def use_origin_search(_result):
            st.session_state.origin_location_mode = "search"
            if _result:
                st.session_state.origin_search_result = _result

        def use_destination_search(_result):
            st.session_state.destination_location_mode = "search"
            if _result:
                st.session_state.destination_search_result = _result

        origin_is_current = (
            st.session_state.get("origin_location_mode") == "current"
            and st.session_state.get("current_origin_coordinate")
        )
        destination_is_current = (
            st.session_state.get("destination_location_mode") == "current"
            and st.session_state.get("current_destination_coordinate")
        )

        if origin_is_current:
            st.text_input("Origin", value="Current location", disabled=True, key="origin_current_label")
            origin_result = None
            if st.button("Search a different origin", use_container_width=True):
                st.session_state.origin_location_mode = "search"
                st.rerun()
        else:
            origin_result = st_searchbox(
                mapbox_suggestions,
                label="Origin",
                placeholder="Search address or place (e.g., Atmospheric Sciences)…",
                default_searchterm=area_config["origin_default"],
                default_options=default_location_option(area_config["origin_default"], area_config["origin_coord"]),
                debounce=350,
                edit_after_submit="current",
                submit_function=use_origin_search,
                key=f"mapbox_origin_search_{area_config['key']}",
            )

        if destination_is_current:
            st.text_input("Destination", value="Current location", disabled=True, key="destination_current_label")
            destination_result = None
            if st.button("Search a different destination", use_container_width=True):
                st.session_state.destination_location_mode = "search"
                st.rerun()
        else:
            destination_result = st_searchbox(
                mapbox_suggestions,
                label="Destination",
                placeholder="Search address, building, or place…",
                default_searchterm=area_config["destination_default"],
                default_options=default_location_option(area_config["destination_default"], area_config["destination_coord"]),
                debounce=350,
                edit_after_submit="current",
                submit_function=use_destination_search,
                key=f"mapbox_destination_search_{area_config['key']}",
            )

        location_cols = st.columns(2)
        if location_cols[0].button("📍 Use my location as origin", use_container_width=True):
            st.session_state.geolocation_target = "origin"
        if location_cols[1].button("📍 Use my location as destination", use_container_width=True):
            st.session_state.geolocation_target = "destination"

        geolocation_target = st.session_state.get("geolocation_target")
        if geolocation_target:
            geolocation = get_geolocation(component_key="heatwise_browser_geolocation")
            if geolocation and geolocation.get("coords"):
                coords = geolocation["coords"]
                current_coordinate = (float(coords["latitude"]), float(coords["longitude"]))
                st.session_state[f"current_{geolocation_target}_coordinate"] = current_coordinate
                st.session_state[f"{geolocation_target}_location_mode"] = "current"
                st.session_state.geolocation_target = None
                st.rerun()
            else:
                st.info("Allow location access in the browser to use your current position.")

        if st.session_state.get("origin_location_mode") == "current" and st.session_state.get("current_origin_coordinate"):
            origin_name = "Current location"
            origin_coordinate = st.session_state.current_origin_coordinate
        elif origin_result or st.session_state.get("origin_search_result"):
            selected_origin = origin_result or st.session_state.origin_search_result
            st.session_state.origin_search_result = selected_origin
            origin_name = selected_origin.label
            origin_coordinate = (selected_origin.latitude, selected_origin.longitude)
        else:
            origin_name, origin_coordinate = area_config["origin_default"], area_config["origin_coord"]

        if st.session_state.get("destination_location_mode") == "current" and st.session_state.get("current_destination_coordinate"):
            destination_name = "Current location"
            destination_coordinate = st.session_state.current_destination_coordinate
        elif destination_result or st.session_state.get("destination_search_result"):
            selected_destination = destination_result or st.session_state.destination_search_result
            st.session_state.destination_search_result = selected_destination
            destination_name = selected_destination.label
            destination_coordinate = (selected_destination.latitude, selected_destination.longitude)
        else:
            destination_name, destination_coordinate = area_config["destination_default"], area_config["destination_coord"]
    else:
        origin_name = st.selectbox("Origin", list(LANDMARKS), index=0)
        destination_name = st.selectbox("Destination", list(LANDMARKS), index=3)
        origin_coordinate, destination_coordinate = LANDMARKS[origin_name], LANDMARKS[destination_name]
    age = st.selectbox("Age group", [item.value for item in AgeGroup])
    activity = st.selectbox("Activity", [item.value for item in Activity])
    clothing = st.selectbox(
        "Clothing",
        [item.value for item in Clothing],
        index=1,
        help="Clothing insulation is entered directly into the PET calculation. Choose the option closest to what you are wearing.",
    )
    if data_source == "Live College Station data" and activity == Activity.CYCLING.value:
        LIVE_GRAPH = area_config.get("prepared_bike_graph", area_config["bike_graph"])
        if not routing_graph_exists(LIVE_GRAPH):
            LIVE_GRAPH = area_config["bike_graph"]
        st.caption("Cycling uses the dedicated OSM bicycle network.")
    else:
        LIVE_GRAPH = area_config.get("prepared_graph", area_config["graph"])
        if not routing_graph_exists(LIVE_GRAPH):
            LIVE_GRAPH = area_config["graph"]
        pass
    if data_source == "Live College Station data":
        use_latest_hour = st.toggle("Use latest complete hour", value=True)
        now_hour = current_hour()
        latest_complete_hour = now_hour - timedelta(hours=1)
        selected_time = st.time_input(
            "Temperature hour (College Station)",
            value=latest_complete_hour.time(),
            step=3600,
            disabled=use_latest_hour,
        )
        requested_at = (
            latest_complete_hour
            if use_latest_hour
            else datetime.combine(now_hour.date(), selected_time, tzinfo=COLLEGE_STATION_TZ)
        )
    pending_request = {
        "data_source": data_source,
        "study_area": study_area,
        "map_style": map_style,
        "origin_name": origin_name,
        "origin_coordinate": tuple(origin_coordinate),
        "destination_name": destination_name,
        "destination_coordinate": tuple(destination_coordinate),
        "age": age,
        "activity": activity,
        "clothing": clothing,
        "requested_at": requested_at.isoformat() if data_source == "Live College Station data" else None,
    }
    calculate_clicked = st.button(
        "Calculate routes",
        type="primary",
        use_container_width=True,
        help="Applies the selected locations, profile and synchronized environmental hour.",
    )
    if calculate_clicked:
        st.session_state.applied_route_request = pending_request

    applied_request = st.session_state.get("applied_route_request")
    if applied_request != pending_request:
        st.info("Selections are ready. Click **Calculate routes** to update the results.")
        st.stop()

    st.divider()
    st.subheader("Weather inputs")
    if data_source == "Prototype scenario":
        temperature = st.slider("Air temperature (°C)", 25.0, 45.0, 36.0, 0.5)
        humidity = st.slider("Relative humidity (%)", 10, 100, 52)
        wind = st.slider("Wind speed (m/s)", 0.0, 8.0, 2.2, 0.1)
        radiation = st.slider("Shortwave radiation (W/m²)", 0, 1200, 700, 25)
        direct_normal_radiation = st.slider("Direct normal irradiance (W/m²)", 0, 1000, 800, 25)
        diffuse_radiation = st.slider("Diffuse radiation (W/m²)", 0, 500, 100, 10)
        cloud_cover = st.slider("Cloud cover (%)", 0, 100, 20)
        precipitation = st.slider("Precipitation (mm/h)", 0.0, 20.0, 0.0, 0.1)
    else:
        force_refresh = st.button(
            "Refresh selected hour",
            help="Refreshes FortyGuard temperature and Open-Meteo humidity/wind; normal reruns use hourly caches.",
        )
        try:
            weather_snapshot = fetch_hourly_weather(
                requested_at, force=force_refresh, latitude=area_config["latitude"],
                longitude=area_config["longitude"], area=area_config["key"],
            )
            humidity = weather_snapshot.humidity_pct
            wind = weather_snapshot.wind_mps
            radiation = weather_snapshot.shortwave_radiation_wm2
            direct_normal_radiation = weather_snapshot.direct_normal_irradiance_wm2
            diffuse_radiation = weather_snapshot.diffuse_radiation_wm2
            cloud_cover = weather_snapshot.cloud_cover_pct
            precipitation = weather_snapshot.precipitation_mm
            weather_cols = st.columns(2)
            weather_cols[0].metric("Live humidity", f"{humidity:.0f}%")
            weather_cols[1].metric("Wind at 10 m", f"{wind:.1f} m/s")
            condition_cols = st.columns(3)
            condition_cols[0].metric("Radiation", f"{radiation:.0f} W/m²")
            condition_cols[1].metric("Cloud cover", f"{cloud_cover:.0f}%")
            condition_cols[2].metric("Precipitation", f"{precipitation:.1f} mm")
            st.caption(
                f"Hourly weather · 2 m relative humidity · 10 m wind from "
                f"{weather_snapshot.wind_direction_deg:.0f}° · {weather_snapshot.source}."
            )
        except Exception as weather_error:
            st.error(
                "Exact-hour weather is unavailable, so thermal risk was not calculated with "
                f"unsynchronized manual values. Please retry. ({type(weather_error).__name__})"
            )
            st.stop()

# Heavy geospatial and thermal libraries load only after the user explicitly
# requests a calculation. Location search and profile controls therefore stay
# responsive during judging, even after a free-tier cold start.
import pandas as pd
from streamlit_folium import st_folium
from heatwise.assistant import answer_question
from heatwise.albedo import apply_osm_surface_albedo
from heatwise.demo import DemoWeather, build_demo_graph, nearest_node
from heatwise.data_integration import assign_heatmap_to_graph, load_fortyguard_heatmap
from heatwise.graph_io import load_routing_graph
from heatwise.horizon import apply_horizon_profile
from heatwise.map_view import ROUTE_COLORS, build_map
from heatwise.routing import calculate_routes, prepare_graph, route_temperature_profile
from heatwise.shade import apply_dynamic_shade, solar_position
from heatwise.svf import apply_lidar_sky_view_factor
from heatwise.view3d import build_3d_deck
from heatwise.wind import apply_local_pedestrian_wind, logarithmic_wind_speed

if origin_name == destination_name:
    st.warning("Please choose different origin and destination points.")
    st.stop()

profile = UserProfile(AgeGroup(age), Activity(activity), Clothing(clothing))
if data_source == "Live College Station data":
    try:
        with st.spinner("Loading the current FortyGuard temperature layer..."):
            snapshot = fetch_hourly_snapshot(requested_at, force=force_refresh, area=area_config["key"])
    except Exception as error:
        snapshot = latest_cached_snapshot(area_config["key"])
        if snapshot is None:
            st.error(f"FortyGuard is unavailable and no hourly cache exists: {error}")
            st.stop()
        st.warning(f"FortyGuard refresh failed; using the latest cached snapshot. ({type(error).__name__})")
    if snapshot.requested_at != requested_at:
        st.error(
            "FortyGuard did not return the exact selected hour. To prevent mixing times, "
            f"thermal risk was not calculated with the {snapshot.requested_at.strftime('%b %d, %Y at %-I:00 %p %Z')} layer."
        )
        st.stop()
    base_graph, temperature, svf_meta = load_snapshot_graph(
        str(LIVE_GRAPH), str(snapshot.path), str(CANOPY_HEIGHT), str(BUILDING_HEIGHT)
    )
    prepared_lidar = LIVE_GRAPH.stem.endswith("_prepared")
    horizon_path = area_config.get("bike_horizon" if activity == "Cycling" else "walk_horizon")
    if prepared_lidar and horizon_path and horizon_path.exists():
        base_graph, shade_meta = apply_horizon_profile(
            base_graph, horizon_path, requested_at, area_config["latitude"], area_config["longitude"]
        )
        svf_meta.update({key: shade_meta[key] for key in ("mean_svf", "lidar_coverage_pct")})
        shade_description = (
            "hourly LiDAR tree/building horizon shade + solar geometry "
            f"(sun elevation {shade_meta['solar_elevation_deg']:.1f}°, "
            f"azimuth {shade_meta['solar_azimuth_deg']:.0f}°, {shade_meta['shaded_edges_pct']:.0f}% shaded)"
        )
    elif prepared_lidar:
        elevation, azimuth = solar_position(requested_at, area_config["latitude"], area_config["longitude"])
        shade_meta = {"solar_elevation_deg": elevation, "solar_azimuth_deg": azimuth}
        shade_description = "precomputed LiDAR shade (horizon profile unavailable)"
    elif CANOPY_HEIGHT.exists():
        base_graph, shade_meta = apply_dynamic_shade(
            base_graph,
            CANOPY_HEIGHT,
            requested_at,
            building_height_path=BUILDING_HEIGHT if BUILDING_HEIGHT.exists() else None,
        )
        shade_description = (
            f"LiDAR tree/building height + solar geometry "
            f"(sun elevation {shade_meta['solar_elevation_deg']:.1f}°, "
            f"azimuth {shade_meta['solar_azimuth_deg']:.0f}°)"
        )
        if BUILDING_HEIGHT.exists():
            shade_description += (
                f" + LiDAR SVF {svf_meta['mean_svf']:.2f} mean "
                f"({svf_meta['lidar_coverage_pct']:.0f}% network coverage)"
            )
    else:
        shade_description = "College Station vegetation proxy"
    if prepared_lidar:
        modeled_winds = []
        for *_, edge in base_graph.edges(data=True):
            z0 = float(edge.get("roughness_length_m", 0.1))
            edge["wind_10m_mps"] = float(wind)
            edge["wind_mps"] = logarithmic_wind_speed(wind, roughness_length_m=z0)
            modeled_winds.append(edge["wind_mps"])
        mean_wind = sum(modeled_winds) / len(modeled_winds) if modeled_winds else wind
        wind_description = f"modeled pedestrian wind {mean_wind:.1f} m/s mean (precomputed LiDAR roughness)"
    elif CANOPY_HEIGHT.exists() and BUILDING_HEIGHT.exists():
        base_graph, wind_meta = apply_local_pedestrian_wind(
            base_graph,
            wind,
            CANOPY_HEIGHT,
            BUILDING_HEIGHT,
        )
        wind_description = (
            f"modeled pedestrian wind {wind_meta['mean_wind_2m_mps']:.1f} m/s mean "
            f"(local log-profile roughness)"
        )
    else:
        wind_description = f"10 m wind {wind:.1f} m/s"
    st.markdown(f'<div class="notice"><b>Real-time mode:</b> {snapshot.label}. FortyGuard temperature grid + OpenStreetMap walk network + {shade_description}.</div>', unsafe_allow_html=True)
    st.caption(f"Area mean from this FortyGuard layer: {temperature:.1f} °C. Updated {snapshot.fetched_at.strftime('%b %d at %-I:%M %p %Z')}.")
    st.caption(
        f"Weather at {requested_at.strftime('%-I:%M %p')}: relative humidity {humidity:.0f}% · "
        f"10 m wind {wind:.1f} m/s."
    )
    st.success(
        f"Time synchronized: FortyGuard temperature, hourly weather, solar position, "
        f"and LiDAR shade all use {requested_at.strftime('%b %d, %Y · %-I:00 %p %Z')}."
    )
    st.caption(
        f"Shortwave radiation {radiation:.0f} W/m² · cloud cover {cloud_cover:.0f}% · "
        f"precipitation {precipitation:.1f} mm/h · {wind_description}."
    )
    if precipitation >= 0.2:
        st.warning("Precipitation is expected for the selected hour; walking surfaces may be wet or slippery.")
else:
    st.markdown('<div class="notice"><b>Prototype mode:</b> temperatures and shade are simulated over the Texas A&M campus.</div>', unsafe_allow_html=True)
    base_graph = build_demo_graph()
    temp_shift = temperature - 36.0
    for _, _, edge in base_graph.edges(data=True):
        edge["temperature_c"] += temp_shift
    shade_meta = {"solar_elevation_deg": 55.0, "solar_azimuth_deg": 180.0}
base_graph, albedo_meta = apply_osm_surface_albedo(base_graph)
weather = DemoWeather(temperature, float(humidity), wind)
graph = prepare_graph(
    base_graph,
    profile,
    weather.humidity_pct,
    weather.wind_mps,
    radiation_wm2=float(radiation),
    direct_normal_radiation_wm2=float(direct_normal_radiation),
    diffuse_radiation_wm2=float(diffuse_radiation),
    cloud_cover_pct=float(cloud_cover),
    solar_elevation_deg=float(shade_meta["solar_elevation_deg"]),
    solar_azimuth_deg=float(shade_meta["solar_azimuth_deg"]),
)
origin = nearest_node(graph, origin_coordinate)
destination = nearest_node(graph, destination_coordinate)
routes = calculate_routes(graph, origin, destination, profile)

tab_map, tab_3d, tab_compare, tab_chat, tab_method = st.tabs(
    ["Route map", "3D Urban Heat", "Compare routes", "Ask HeatWise AI", "How it works"]
)
with tab_map:
    selected_name = st.radio("Show route", [r.name for r in routes], horizontal=True)
    selected = next(r for r in routes if r.name == selected_name)
    st.markdown(
        f'<div class="personal-risk"><b>Personal Heat Risk (PET): {selected.pet_category}</b><br>'
        f'Activity-sensitive PET-equivalent temperature: {selected.average_pet_c:.1f} °C for {age}, '
        f'{activity.lower()} and {clothing.lower()}. UTCI is reported separately as environmental heat stress; '
        f'route duration is included in the exposure-load metric below.</div>',
        unsafe_allow_html=True,
    )
    first_metrics = st.columns(5)
    first_metrics[0].metric("Travel time", f"{selected.duration_min:.1f} min")
    first_metrics[1].metric("Distance", f"{selected.distance_m / 1000:.2f} km")
    first_metrics[2].metric("Temperature range", f"{selected.minimum_temp_c:.1f}–{selected.maximum_temp_c:.1f} °C")
    first_metrics[3].metric("Average temperature", f"{selected.average_temp_c:.1f} °C")
    first_metrics[4].metric("Time-dependent shade", f"{selected.shade_pct:.0f}%")
    second_metrics = st.columns(4)
    second_metrics[0].metric("Mean UTCI equivalent", f"{selected.average_utci_c:.1f} °C")
    second_metrics[1].metric("UTCI category", selected.risk)
    second_metrics[2].metric("Activity PET equivalent", f"{selected.average_pet_c:.1f} °C")
    second_metrics[3].metric("Personal heat risk", selected.pet_category)
    st.caption(
        f"Route-average mean radiant temperature (SolarCal estimate): {selected.average_mrt_c:.1f} °C · "
        f"clothing insulation: {profile.summer_clothing_clo:.1f} clo · "
        f"UTCI exposure load: {selected.utci_exposure_load:.1f} °C·min above 26 °C · "
        f"activity-sensitive PET load: {selected.pet_exposure_load:.1f} °C·min above 29 °C "
        f"({selected.pet_category})."
    )
    st.caption(
        f"Route-average modeled pedestrian wind at 2 m: {selected.average_wind_mps:.1f} m/s. "
        "Derived from 10 m wind using local tree/building roughness; not a direct 2 m observation."
    )
    boundary = load_boundary(str(CITY_BOUNDARY)) if CITY_BOUNDARY.exists() else None
    st_folium(
        build_map(
            graph,
            routes,
            selected_name,
            boundary,
            mapbox_token=mapbox_token,
            mapbox_style=map_style,
            origin_coordinate=origin_coordinate,
            destination_coordinate=destination_coordinate,
        ),
        use_container_width=True,
        height=580,
        returned_objects=[],
    )
    profile_frame = pd.DataFrame(route_temperature_profile(graph, selected))
    if not profile_frame.empty:
        st.subheader("Temperature along the selected route")
        st.line_chart(
            profile_frame.set_index("distance_km")[["temperature_c"]],
            x_label="Distance along route (km)",
            y_label="Air temperature (°C)",
            color="#dc2626",
        )
        hottest = profile_frame.loc[profile_frame["temperature_c"].idxmax()]
        st.caption(
            f"Hottest segment: {hottest['temperature_c']:.1f} °C at approximately "
            f"{hottest['distance_km']:.2f} km from the origin. The red marker identifies it on the map."
        )
    fastest = routes[0]
    if selected.name == "Lowest Heat Risk":
        difference = selected.duration_min - fastest.duration_min
        reduction = 100 * (fastest.heat_score - selected.heat_score) / max(fastest.heat_score, 1)
        st.success(f"This route reduces modeled heat exposure by {reduction:.0f}% for about {difference:.1f} additional minutes.")

with tab_3d:
    st.subheader("3D Urban Heat Explorer")
    st.caption(
        "An explanatory view, not a decorative city model: OSM footprints are extruded with LiDAR height, "
        "tree crowns come from the LiDAR canopy-height model, and heat/shade values are the same values used by routing."
    )
    if st.toggle("Load the 3D scene", False, help="Loaded on demand to keep the public app responsive."):
        layer_cols = st.columns(4)
        show_3d_buildings = layer_cols[0].checkbox("LiDAR buildings", True)
        show_3d_canopy = layer_cols[1].checkbox("LiDAR tree canopy", True)
        show_3d_heat = layer_cols[2].checkbox("Temperature network", True)
        show_3d_shade = layer_cols[3].checkbox("Modeled shade", False)
        with st.spinner("Preparing the route-scale 3D scene..."):
            deck, scene_meta = build_3d_deck(
                graph, routes, selected_name,
                area_config["building_footprints"], BUILDING_HEIGHT, CANOPY_HEIGHT,
                show_buildings=show_3d_buildings,
                show_canopy=show_3d_canopy,
                show_heat=show_3d_heat,
                show_shade=show_3d_shade,
                mapbox_token=mapbox_token,
            )
        st.pydeck_chart(deck, width="stretch", height=650)
        stats = st.columns(3)
        stats[0].metric("Buildings in view", f"{scene_meta['buildings']:,}")
        stats[1].metric("Canopy samples", f"{scene_meta['canopy_samples']:,}")
        stats[2].metric("Network segments", f"{scene_meta['network_segments']:,}")
        st.caption(
            "Building height is the 90th percentile of LiDAR pixels inside each OSM footprint. "
            "Canopy is downsampled only for rendering; the full-resolution raster remains the input to shade, SVF and routing calculations."
        )
    else:
        st.info("Turn on the 3D scene when you want to inspect LiDAR buildings, canopy, heat, and shade.")
with tab_compare:
    for route in routes:
        st.markdown(f"### <span style='color:{ROUTE_COLORS[route.name]}'>{route.name}</span>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Time", f"{route.duration_min:.1f} min")
        c2.metric("Distance", f"{route.distance_m/1000:.2f} km")
        c3.metric("Temperature", f"{route.minimum_temp_c:.1f}–{route.maximum_temp_c:.1f} °C")
        c4.metric("Estimated shade", f"{route.shade_pct:.0f}%")
        c5.metric("Mean UTCI equivalent", f"{route.average_utci_c:.1f} °C")
        c6.metric("UTCI category", route.risk)
        c7.metric("Activity PET equivalent", f"{route.average_pet_c:.1f} °C")

with tab_chat:
    st.subheader("Ask HeatWise AI")
    if os.getenv("GROQ_API_KEY"):
        st.success("Conversational AI is active: Groq · GPT-OSS 20B")
    elif os.getenv("OPENAI_API_KEY"):
        st.info("OpenAI is configured. If billing is inactive, HeatWise automatically uses its local fallback.")
    else:
        st.info("Local assistant mode is active. Add GROQ_API_KEY to .env for full conversational AI.")
    st.caption("Ask about route tradeoffs, shade, exposure, age, activity, or departure conditions. Not medical advice.")
    if "heatwise_messages" not in st.session_state:
        st.session_state.heatwise_messages = [
            {"role": "assistant", "content": "Hi! Ask me which route best matches your current age and activity profile."}
        ]
    for message in st.session_state.heatwise_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Why is the lowest-risk route better for me?"):
        st.session_state.heatwise_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Reviewing your calculated routes..."):
                try:
                    reply, assistant_mode = answer_question(
                        prompt, routes, profile, origin_name, destination_name,
                        weather.humidity_pct, weather.wind_mps,
                    )
                except Exception as error:
                    reply = f"The online assistant is temporarily unavailable. Please check the OpenAI API key and billing. ({type(error).__name__})"
                    assistant_mode = "Error fallback"
                st.markdown(reply)
                st.caption(f"Mode: {assistant_mode}")
        st.session_state.heatwise_messages.append({"role": "assistant", "content": reply})

with tab_method:
    st.subheader("Standards-based outdoor thermal stress")
    st.write(
        "Each network segment uses ASHRAE 55 SolarCal to estimate mean radiant temperature and "
        "ISO 14505-2 UTCI to combine air temperature, mean radiant temperature, relative humidity, "
        "and 10 m wind. UTCI remains the activity-independent environmental index. The MEMI steady-state "
        "PET model adds age, activity MET and summer clothing for activity-sensitive routing."
    )
    st.code("UTCI load = Σ[max(0, UTCI − 26) × duration]\nActivity PET load = Σ[max(0, PET − 29) × duration]")
    st.info(
        "Walking, cycling and jogging use 3.3, 6.8 and 7.0 MET respectively (2024 Adult Compendium). "
        "Cycling uses an OSM bicycle network; walking and jogging use the pedestrian network. Age is not "
        "multiplied into UTCI, but a representative age is supplied to the PET physiology model."
    )
    st.markdown(
        "**Scientific references:** [UTCI operational model](https://doi.org/10.1007/s00484-011-0454-1) · "
        "[ISO 14505-2 UTCI](https://www.iso.org/standard/79771.html) · "
        "[PET/MEMI implementation](https://pythermalcomfort.readthedocs.io/) · "
        "[2024 Compendium of Physical Activities](https://pacompendium.com/) · "
        "[ASHRAE 55 thermal-environment standard](https://www.ashrae.org/technical-resources/bookstore/standard-55-thermal-environmental-conditions-for-human-occupancy)."
    )
    st.caption(
        "Current limitation: SolarCal supplies an engineering MRT estimate with air temperature as the long-wave "
        "baseline. Full urban-radiation validation would require measured/modelled long-wave fluxes, sky-view factor, "
        "surface albedo and a model such as SOLWEIG."
    )
    st.markdown(
        "**Live-data design:** FortyGuard temperature grid + weather-station humidity/wind + "
        "time-dependent LiDAR tree/building shade + Open-Meteo hourly humidity and 10 m wind + "
        "OSM walking/cycling network."
    )
