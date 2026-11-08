import json
from dataclasses import replace
from unittest.mock import patch
import networkx as nx
import numpy as np
import pytest
from shapely.geometry import shape
from heatwise.demo import build_demo_graph,nearest_node
from heatwise.models import UserProfile
from heatwise.routing import prepare_graph,calculate_routes
from heatwise.relief import (load_catalog,place_connection,routes_via_place,
                            comparison_frame,places_geojson)
from heatwise.planning import segment_frame,planning_grid,shade_scenario


@pytest.fixture(scope='module')
def setup():
    p=UserProfile();environment=dict(humidity_pct=44,wind_mps=2,radiation_wm2=700,
        direct_normal_radiation_wm2=650,diffuse_radiation_wm2=150,cloud_cover_pct=20,
        solar_elevation_deg=55,solar_azimuth_deg=180)
    g=prepare_graph(build_demo_graph(size=5),p,**environment)
    return g,p,environment


def test_catalog_has_distinct_evidence_categories_and_valid_geometry():
    c=load_catalog();assert len(c['items'])==21
    assert sum(p['category']=='cooling_center' for p in c['items'])==10
    assert sum(p['category']=='indoor_refuge' for p in c['items'])==5
    assert sum(p['category']=='park' for p in c['items'])==6
    for p in c['items']:
        assert p['source_url'].startswith('https://') and p['verified_on']=='2026-11-08'
        assert 30<p['latitude']<31 and -97<p['longitude']<-96
        if p.get('geometry'):assert shape(p['geometry']).is_valid
        if p['category']=='cooling_center':assert p['area']=='regional'
    assert len(places_geojson(c['items'])['features'])==21


def test_regional_far_and_disconnected_places_cannot_be_routed(setup):
    g,_,_=setup
    p={'area':'regional','latitude':30.75,'longitude':-96.3417}
    assert not place_connection(g,p,'tamu')['routable']
    p.update(area='tamu',latitude=30.7)
    assert not place_connection(g,p,'tamu')['routable']
    g=g.copy();g.add_node(99,x=-96.3417,y=30.61471)
    result=place_connection(g,{'area':'tamu','latitude':30.61471,'longitude':-96.3417},'tamu',origin=0)
    assert result['node']==99 and not result['routable']
    assert result['reason'].startswith('No connected route')


def test_required_stop_routes_are_connected_and_match_objective_sums(setup):
    g,p,_=setup;stop={'area':'tamu','latitude':g.nodes[20]['y'],'longitude':g.nodes[20]['x'],'name':'Test park','category':'park'}
    direct=calculate_routes(g,0,24,p);rs,conn=routes_via_place(g,0,24,p,stop,'tamu')
    for r in rs:
        assert 20 in r.nodes and nx.is_path(g,r.nodes)
        assert r.pet_exposure_load==pytest.approx(sum(g[a][b]['pet_heat_cost'] for a,b in zip(r.nodes,r.nodes[1:])))
    assert rs[0].duration_min>=direct[0].duration_min-1e-9
    assert rs[1].pet_exposure_load>=direct[1].pet_exposure_load-1e-9
    f=comparison_frame(direct,rs,stop,{'environmental_time':'example'})
    assert not f.indoor_recovery_modeled.any() and not f.connector_and_stop_time_included.any()


def test_grid_preserves_network_length_and_flags_unknown_access(setup):
    g,_,_=setup;s=segment_frame(g,[],'tamu');grid=planning_grid(s)
    assert grid.network_km.sum()==pytest.approx(sum(e['length'] for _,_,e in g.edges(data=True))/1000)
    assert grid.unknown_refuge_access_km.sum()==pytest.approx(grid.network_km.sum())
    assert grid.hot_refuge_gap_km.sum()==0
    assert (grid.hot_low_shade_km<=grid.network_km+1e-9).all()
    assert len(json.loads(grid.to_crs(4326).to_json())['features'])==len(grid)


def test_reverse_edges_are_not_counted_twice(setup):
    g,_,_=setup;d=nx.MultiDiGraph(g)
    s=segment_frame(d,[],'tamu')
    assert len(s)==g.number_of_edges()
    assert s.length_m.sum()==pytest.approx(sum(e['length'] for _,_,e in g.edges(data=True)))


def test_network_refuge_distance_follows_connectivity(setup):
    g,_,_=setup;n=0
    place={'category':'indoor_refuge','area':'tamu','latitude':g.nodes[n]['y'],'longitude':g.nodes[n]['x']}
    s=segment_frame(g,[place],'tamu')
    assert s.refuge_distance_m.notna().all() and (s.refuge_distance_m>0).all()
    distances=nx.single_source_dijkstra_path_length(g,n,weight='length')
    first=s.iloc[0];a,b=first.edges[0]
    assert first.refuge_distance_m==pytest.approx(min(distances[a],distances[b])+first.length_m/2)


def test_shade_scenario_zero_is_identity_and_does_not_mutate_graph(setup):
    g,p,e=setup;s=segment_frame(g,[],'tamu');grid=planning_grid(s);cell=grid.iloc[0].cell_id
    old=[x['shade_fraction'] for _,_,x in g.edges(data=True)]
    zero=shade_scenario(g,s,cell,200,0,p,e)
    assert zero['modeled_pet_change_c']==pytest.approx(0,abs=.001)
    more=shade_scenario(g,s,cell,200,30,p,e)
    assert more['scenario_shade_pct']>more['baseline_shade_pct']
    assert more['modeled_pet_change_c']<0
    assert old==[x['shade_fraction'] for _,_,x in g.edges(data=True)]


def test_long_segments_are_clipped_and_length_is_conserved():
    g=nx.Graph();g.add_node(0,x=-96.34,y=30.61);g.add_node(1,x=-96.33,y=30.61)
    g.add_edge(0,1,length=1000,pet_c=45,utci_c=39,temperature_c=36,shade_fraction=.1)
    grid=planning_grid(segment_frame(g,[],'tamu'))
    assert len(grid)>=4
    assert grid.network_km.sum()==pytest.approx(1)
    assert grid.network_km.max()<.4
    assert grid.hot_low_shade_km.sum()==pytest.approx(1)


def test_directed_access_does_not_invent_a_reverse_path():
    g=nx.DiGraph();g.add_node(0,x=-96.34,y=30.61);g.add_node(1,x=-96.339,y=30.61)
    g.add_edge(0,1,length=100,pet_c=45,utci_c=39,temperature_c=36,shade_fraction=.1)
    p={'category':'indoor_refuge','area':'tamu','latitude':30.61,'longitude':-96.34}
    s=segment_frame(g,[p],'tamu')
    assert s.refuge_distance_m.isna().all()


def test_regional_facility_inside_prepared_extent_is_eligible(setup):
    g,_,_=setup
    p={'area':'regional','latitude':g.nodes[12]['y'],'longitude':g.nodes[12]['x']}
    result=place_connection(g,p,'tamu',0,24)
    assert result['routable'] and result['node']==12


def test_weekly_hours_and_closed_stop_guard(setup):
    from datetime import datetime
    from heatwise.relief import listed_closed,schedule_state
    g,p,_=setup
    place=next(x for x in load_catalog()['items'] if x['id']=='city-cooling-7')
    assert listed_closed(place,datetime(2026,8,1,14))  # Saturday
    assert not listed_closed(place,datetime(2026,8,3,14))
    with pytest.raises(ValueError,match='Closed'):
        routes_via_place(g,0,24,p,place,'tamu',when=datetime(2026,8,1,14))
    library=next(x for x in load_catalog()['items'] if x['id']=='indoor-249571643')
    assert 'unconfirmed' in schedule_state(library,datetime(2026,8,1,14))
