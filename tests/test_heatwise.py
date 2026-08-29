import networkx as nx
import pickle
from datetime import datetime, timezone
from unittest.mock import patch

from heatwise.assistant import answer_question
from heatwise.demo import build_demo_graph, nearest_node
from heatwise.data_integration import load_fortyguard_heatmap
from heatwise.live_data import MAX_FALLBACK_DAYS, cache_path, current_hour
from heatwise.graph_io import load_routing_graph, routing_graph_exists
from heatwise.models import Activity, AgeGroup, Clothing, UserProfile
from heatwise.risk import apparent_temperature_c, segment_heat_cost
from heatwise.routing import calculate_routes, prepare_graph


def test_hot_humid_conditions_raise_apparent_temperature():
    assert apparent_temperature_c(36, 70, 1) > apparent_temperature_c(36, 30, 1)


def test_age_does_not_alter_standard_utci():
    kwargs = dict(temp_c=36, humidity_pct=55, wind_mps=2, shade_fraction=0.2, duration_minutes=10)
    walker = segment_heat_cost(**kwargs, profile=UserProfile(AgeGroup.YOUNG, Activity.WALKING))
    older_jogger = segment_heat_cost(**kwargs, profile=UserProfile(AgeGroup.OLDER, Activity.JOGGING))
    assert older_jogger == walker


def test_shade_reduces_cost():
    profile = UserProfile()
    exposed = segment_heat_cost(temp_c=38, humidity_pct=50, wind_mps=1, direct_normal_radiation_wm2=800, shade_fraction=0, duration_minutes=10, profile=profile)
    shaded = segment_heat_cost(temp_c=38, humidity_pct=50, wind_mps=1, direct_normal_radiation_wm2=800, shade_fraction=0.8, duration_minutes=10, profile=profile)
    assert shaded < exposed


def test_demo_routes_are_connected_and_summarized():
    graph = prepare_graph(build_demo_graph(), UserProfile(), 52, 2.2)
    origin = nearest_node(graph, (30.6102, -96.3404))
    destination = nearest_node(graph, (30.6210, -96.3408))
    routes = calculate_routes(graph, origin, destination)
    assert [r.name for r in routes] == ["Fastest", "Lowest Heat Risk", "Balanced"]
    assert all(r.distance_m > 0 and r.duration_min > 0 for r in routes)
    assert all(nx.is_path(graph, r.nodes) for r in routes)


def test_cached_fortyguard_heatmap_loader():
    frame = load_fortyguard_heatmap(
        "data/heatmaps/heatmap_parcel_diridon_san_jose_2024-07-15_tcm.json"
    )
    assert len(frame) > 0
    assert frame.crs.to_epsg() == 4326
    assert frame["average_temperature"].notna().all()


def test_local_assistant_answers_without_openai_key():
    graph = prepare_graph(build_demo_graph(), UserProfile(), 52, 2.2)
    routes = calculate_routes(graph, 46, 150)
    with patch.dict("os.environ", {}, clear=True):
        answer, mode = answer_question(
            "Why is the cooler route different?", routes, UserProfile(),
            "Kyle Field", "Zachry Engineering", 52, 2.2,
        )
    assert mode == "Local fallback"
    assert "lowest-risk" in answer.lower()


def test_current_hour_uses_college_station_timezone():
    utc = datetime(2026, 8, 25, 5, 37, tzinfo=timezone.utc)
    local = current_hour(utc)
    assert local.isoformat() == "2026-08-25T00:00:00-05:00"
    assert cache_path(local).name == "tamu_20260825_0000_100m.json"
    assert cache_path(local, "rellis").name == "rellis_20260825_0000_100m.json"


def test_fortyguard_fallback_window_is_bounded():
    assert MAX_FALLBACK_DAYS == 7


def test_user_selected_clothing_sets_pet_insulation():
    profile = UserProfile(AgeGroup.OLDER, Activity.WALKING, Clothing.PROTECTIVE)
    assert profile.summer_clothing_clo == 1.0


def test_graph_loader_prefers_binary_sidecar(tmp_path):
    graphml_path = tmp_path / "campus.graphml"
    cached_graph = nx.MultiDiGraph()
    cached_graph.add_node(1, x=-96.34, y=30.61)
    with graphml_path.with_suffix(".pickle").open("wb") as target:
        pickle.dump(cached_graph, target, protocol=5)
    loaded = load_routing_graph(graphml_path)
    assert list(loaded.nodes) == [1]
    assert routing_graph_exists(graphml_path)
