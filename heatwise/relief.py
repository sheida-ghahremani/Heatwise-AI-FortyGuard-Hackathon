"""Source-attributed relief places and network-constrained stop routing."""
from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import asdict

import networkx as nx
import pandas as pd
from .demo import _haversine_m, nearest_node
from .routing import calculate_routes

CATEGORY_LABELS = {'cooling_center': 'City-listed cooling facility',
                   'indoor_refuge': 'Campus indoor refuge candidate',
                   'park': 'Park / garden'}


def load_catalog(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or Path(__file__).with_name('relief_catalog.json')).read_text())


def schedule_state(place: dict, when=None) -> str:
    """Weekly source schedule only; never claims live or historical verification."""
    if when is None or not place.get('weekly_hours'):
        return 'Hours unconfirmed; check official calendar'
    minute=when.hour*60+when.minute
    allowed=any(a<=minute<b for a,b in place['weekly_hours'].get(str(when.weekday()),[]))
    return 'Listed schedule allows this hour; confirm exceptions' if allowed else 'Closed in listed weekly schedule'


def listed_closed(place: dict, when=None) -> bool:
    return schedule_state(place,when)=='Closed in listed weekly schedule'


def place_connection(graph, place: dict, area: str, origin: int | None = None,
                     destination: int | None = None, max_connector_m: float = 150) -> dict:
    """Never snap an out-of-extent, distant or disconnected place onto the graph."""
    if place['area'] not in (area,'regional'):
        return {'routable': False, 'reason': 'Outside this prepared study network', 'node': None, 'connector_m': None}
    if place['area']=='regional':
        xs=[float(d['x']) for _,d in graph.nodes(data=True)];ys=[float(d['y']) for _,d in graph.nodes(data=True)]
        if not (min(xs)<=place['longitude']<=max(xs) and min(ys)<=place['latitude']<=max(ys)):
            return {'routable': False, 'reason': 'Outside this prepared study network', 'node': None, 'connector_m': None}
    node = nearest_node(graph, (place['latitude'], place['longitude']))
    distance = _haversine_m((place['latitude'], place['longitude']),
                            (float(graph.nodes[node]['y']), float(graph.nodes[node]['x'])))
    if distance > max_connector_m:
        return {'routable': False, 'reason': 'No mapped access path within 150 m', 'node': node, 'connector_m': distance}
    if ((origin is not None and not nx.has_path(graph, origin, node)) or
            (destination is not None and not nx.has_path(graph, node, destination))):
        return {'routable': False, 'reason': 'No connected route for this activity/network', 'node': node, 'connector_m': distance}
    return {'routable': True, 'reason': 'Mapped path available; confirm entrance and opening hours',
            'node': node, 'connector_m': distance}


def catalog_frame(graph, area: str, origin: int, destination: int, *, include_regional=False, when=None) -> pd.DataFrame:
    rows=[]
    start=(float(graph.nodes[origin]['y']),float(graph.nodes[origin]['x']))
    for p in load_catalog()['items']:
        if p['area'] not in (area,'regional'):continue
        connection=place_connection(graph,p,area,origin,destination)
        if p['area']=='regional' and not include_regional and not connection['routable']:continue
        rows.append({**{k:v for k,v in p.items() if k!='geometry'}, **connection,
                     'type':CATEGORY_LABELS[p['category']], 'schedule_status':schedule_state(p,when),
                     'route_stop_eligible':connection['routable'] and not listed_closed(p,when),
                     'straight_line_from_origin_m':_haversine_m(start,(p['latitude'],p['longitude']))})
    return pd.DataFrame(rows)


def routes_via_place(graph, origin: int, destination: int, profile, place: dict, area: str, *, when=None):
    if listed_closed(place,when):raise ValueError('Closed in the listed weekly schedule at the selected hour')
    connection=place_connection(graph,place,area,origin,destination)
    if not connection['routable']:
        raise ValueError(connection['reason'])
    return calculate_routes(graph,origin,destination,profile,via=connection['node']),connection


def comparison_frame(direct, via, place: dict, context: dict) -> pd.DataFrame:
    rows=[]
    for baseline,r in zip(direct,via):
        row={k:v for k,v in asdict(r).items() if k!='nodes'}
        row.update(context)
        row.update(stop=place['name'],stop_category=place['category'],
                   added_outdoor_minutes=r.duration_min-baseline.duration_min,
                   added_pet_load_c_min=r.pet_exposure_load-baseline.pet_exposure_load,
                   indoor_recovery_modeled=False,connector_and_stop_time_included=False)
        rows.append(row)
    return pd.DataFrame(rows)


def places_geojson(items: list[dict]) -> dict:
    return {'type':'FeatureCollection','attribution':'Park geometry and campus building coordinates © OpenStreetMap contributors, ODbL. Facility information: cited public agencies.','catalog_reviewed_on':'2026-11-08','features':[{'type':'Feature',
        'geometry':p.get('geometry') or {'type':'Point','coordinates':[p['longitude'],p['latitude']]},
        'properties':{k:v for k,v in p.items() if k!='geometry'}} for p in items]}


def build_relief_map(items, graph=None, routes=None, origin=None, destination=None):
    import folium
    from html import escape
    from .routing import route_coordinates
    from .map_view import ROUTE_COLORS
    points=[(p['latitude'],p['longitude']) for p in items]
    if graph is not None:
        points += [(float(graph.nodes[n]['y']),float(graph.nodes[n]['x'])) for n in (origin,destination) if n is not None]
        if routes:
            for route in routes:points.extend(route_coordinates(graph,route))
    if not points:points=[(30.614,-96.35)]
    south,north=min(p[0] for p in points),max(p[0] for p in points)
    west,east=min(p[1] for p in points),max(p[1] for p in points)
    center=[(south+north)/2,(west+east)/2]
    cos_lat=math.cos(math.radians(center[0]))
    span=max(north-south,(east-west)*cos_lat*430/800,.0001)
    zoom=max(3,min(16,int(math.floor(math.log2(156543.03392*cos_lat*430*.8/(111320*span))))))
    # Avoid Leaflet fitBounds while Streamlit initializes an inactive tab at
    # zero width: that otherwise leaves the catalog map at world zoom.
    m=folium.Map(location=center,zoom_start=zoom,tiles='OpenStreetMap',control_scale=True)
    colors={'cooling_center':'#2563eb','indoor_refuge':'#7c3aed','park':'#168264'}
    for cat,label in CATEGORY_LABELS.items():
        layer=folium.FeatureGroup(name=label)
        for p in items:
            if p['category']!=cat:continue
            if p.get('geometry'):
                folium.GeoJson(p['geometry'],style_function=lambda _,c=colors[cat]:{'color':c,'fillColor':c,'fillOpacity':.13,'weight':1}).add_to(layer)
            text=(f"<b>{escape(p['name'])}</b><br>{escape(label)}<br>{escape(p['access'])}<br>"
                  f"{escape(p['hours'])}<br>{escape(p['designation'])}<br>"
                  f"Reviewed {escape(p['verified_on'])}<br><a target='_blank' href='{escape(p['source_url'],quote=True)}'>Official/source information</a>")
            folium.CircleMarker([p['latitude'],p['longitude']],radius=7,color=colors[cat],fill=True,fill_opacity=.9,tooltip=p['name'],popup=folium.Popup(text,max_width=360)).add_to(layer)
        layer.add_to(m)
    if graph is not None and routes:
        for r in routes:
            folium.PolyLine(route_coordinates(graph,r),color=ROUTE_COLORS[r.name],weight=5,opacity=.8,tooltip=r.name+' via selected stop').add_to(m)
    if graph is not None:
        for n,label in [(origin,'Origin'),(destination,'Destination')]:
            if n is not None:folium.Marker([graph.nodes[n]['y'],graph.nodes[n]['x']],tooltip=label).add_to(m)
    folium.LayerControl().add_to(m)

    return m
