"""Reproducible campus screening, not population exposure or clinical risk."""
from __future__ import annotations

import math
import json
import networkx as nx
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, box
import shapely
import hashlib
import numpy as np
from pyproj import Transformer
from .relief import place_connection


def segment_frame(graph, indoor_places: list[dict], area: str) -> gpd.GeoDataFrame:
    """Group reverse edges by node pair/length; use mean modeled directional PET.

    Network distance uses the reachable forward exit from each directed edge.
    Actual segment geometry is retained for clipping into the planning grid.
    """
    targets=[]
    for p in indoor_places:
        if p['category'] not in ('indoor_refuge','cooling_center'):continue
        c=place_connection(graph,p,area)
        if c['routable']:targets.append(c['node'])
    reverse=graph.reverse(copy=False) if graph.is_directed() else graph
    distances=nx.multi_source_dijkstra_path_length(reverse,targets,weight='length') if targets else {}
    grouped={}
    records=graph.edges(keys=True,data=True) if graph.is_multigraph() else ((u,v,None,e) for u,v,e in graph.edges(data=True))
    for a,b,key_index,e in records:
        length=float(e.get('length',0))
        if length<=0:continue
        pair=tuple(sorted((str(a),str(b))))
        geom=e.get('geometry')
        if geom is None:geom=LineString([(float(graph.nodes[a]['x']),float(graph.nodes[a]['y'])),(float(graph.nodes[b]['x']),float(graph.nodes[b]['y']))])
        geometry_id=hashlib.sha1(geom.normalize().wkb).hexdigest()[:12]
        key=(*pair,round(length,1),geometry_id)
        lon=(float(graph.nodes[a]['x'])+float(graph.nodes[b]['x']))/2
        lat=(float(graph.nodes[a]['y'])+float(graph.nodes[b]['y']))/2
        if graph.is_directed():
            distance=distances.get(b,math.inf)+length/2
        else:
            distance=min(distances.get(a,math.inf),distances.get(b,math.inf))+length/2
        row=grouped.setdefault(key,{'segment_id':'-'.join(map(str,key)), 'length_m':length,
            'longitude':lon,'latitude':lat,'pets':[],'utcis':[],'temps':[],'shades':[],
            'refuge_distance_m':math.inf,'edges':[],'edge_refs':[],'headings':[],'geometry':geom})
        row['pets'].append(float(e['pet_c']));row['utcis'].append(float(e['utci_c']))
        row['temps'].append(float(e['temperature_c']));row['shades'].append(float(e.get('shade_fraction',0)))
        row['refuge_distance_m']=min(row['refuge_distance_m'],distance)
        row['edges'].append((a,b));row['edge_refs'].append((a,b,key_index))
        from .routing import _bearing_deg
        row['headings'].append(float(e.get('thermal_heading_deg',_bearing_deg(float(graph.nodes[a]['x']),float(graph.nodes[a]['y']),float(graph.nodes[b]['x']),float(graph.nodes[b]['y'])))))
    rows=[]
    for r in grouped.values():
        for k,values in [('pet_c','pets'),('utci_c','utcis'),('temperature_c','temps'),('shade_fraction','shades')]:
            x=r.pop(values);r[k]=sum(x)/len(x)
        if not math.isfinite(r['refuge_distance_m']):r['refuge_distance_m']=float('nan')
        rows.append(r)
    if not rows:raise ValueError('No modeled network segments available')
    return gpd.GeoDataFrame(rows,geometry='geometry',crs=4326).to_crs(32614)


def _clip_segments(segments, cell_size_m):
    s=segments.to_crs(32614).copy()
    xmin,ymin,xmax,ymax=s.total_bounds
    cells=[]
    for cx in range(math.floor(xmin/cell_size_m),math.floor(xmax/cell_size_m)+1):
        for cy in range(math.floor(ymin/cell_size_m),math.floor(ymax/cell_size_m)+1):
            cells.append({'cx':cx,'cy':cy,'geometry':box(cx*cell_size_m,cy*cell_size_m,(cx+1)*cell_size_m,(cy+1)*cell_size_m)})
    cells=gpd.GeoDataFrame(cells,crs=32614)
    joined=gpd.sjoin(s,cells,how='inner',predicate='intersects')
    lengths=shapely.length(shapely.intersection(joined.geometry.to_numpy(),cells.geometry.to_numpy()[joined.index_right.to_numpy()]))
    joined=joined.assign(clipped_geometry_m=lengths)
    joined=joined[joined.clipped_geometry_m>1e-8].copy()
    # Normalize per source segment, including the rare line exactly on a grid
    # boundary. This preserves network length without double counting.
    totals=joined.groupby('segment_id').clipped_geometry_m.transform('sum')
    joined['length_m']=joined.length_m*joined.clipped_geometry_m/totals
    return joined


def planning_grid(segments: gpd.GeoDataFrame, *, cell_size_m: int = 200,
                  pet_threshold_c: float = 41, shade_threshold_pct: float = 20,
                  access_threshold_m: float = 500) -> gpd.GeoDataFrame:
    if cell_size_m not in (100,200,400):raise ValueError('Grid size must be 100, 200 or 400 m')
    if not (0<=shade_threshold_pct<=100) or access_threshold_m<=0:raise ValueError('Invalid screening thresholds')
    s=_clip_segments(segments,cell_size_m)
    rows=[]
    for (cx,cy),g in s.groupby(['cx','cy'],sort=True):
        length=g.length_m.sum()
        weighted=lambda col:float((g[col]*g.length_m).sum()/length)
        hot=g.pet_c>=pet_threshold_c
        shade_gap=hot&(100*g.shade_fraction<shade_threshold_pct)
        known=g.refuge_distance_m.notna()
        access_gap=hot&known&(g.refuge_distance_m>access_threshold_m)
        rows.append({'cell_id':f'{cell_size_m}-{cx}-{cy}', 'network_km':length/1000,
            'mean_pet_c':weighted('pet_c'),'mean_utci_c':weighted('utci_c'),
            'mean_air_temperature_c':weighted('temperature_c'),'mean_shade_pct':100*weighted('shade_fraction'),
            'hot_network_km':float(g.loc[hot,'length_m'].sum()/1000),
            'hot_low_shade_km':float(g.loc[shade_gap,'length_m'].sum()/1000),
            'hot_refuge_gap_km':float(g.loc[access_gap,'length_m'].sum()/1000),
            'unknown_refuge_access_km':float(g.loc[~known,'length_m'].sum()/1000),
            'mean_refuge_distance_m':float((g.loc[known,'refuge_distance_m']*g.loc[known,'length_m']).sum()/g.loc[known,'length_m'].sum()) if known.any() else float('nan'),
            'physical_segments':len(g),'pet_threshold_c':pet_threshold_c,
            'shade_threshold_pct':shade_threshold_pct,'access_threshold_m':access_threshold_m,
            'geometry':box(cx*cell_size_m,cy*cell_size_m,(cx+1)*cell_size_m,(cy+1)*cell_size_m)})
    result=gpd.GeoDataFrame(rows,crs=32614).sort_values(['hot_low_shade_km','mean_pet_c','cell_id'],ascending=[False,False,True]).reset_index(drop=True)
    result.insert(0,'screening_rank',range(1,len(result)+1))
    return result


def shade_scenario(graph, segments, cell_id: str, cell_size_m: int, extra_shade_pct: float,
                   profile, environment: dict):
    """Shade-only sensitivity on the clipped parts of a selected grid cell."""
    from .routing import segment_pet_c
    from .risk import mean_radiant_temperature_c
    if not 0<=extra_shade_pct<=100:raise ValueError('Shade increment must be between 0 and 100')
    _,cx,cy=cell_id.split('-');cx,cy=int(cx),int(cy)
    clipped=_clip_segments(segments,cell_size_m)
    selected=clipped[(clipped['cx']==cx)&(clipped['cy']==cy)].copy()
    if selected.empty:raise ValueError('Selected grid cell has no segments')
    after_pet=[];after_shade=[]
    for _,row in selected.iterrows():
        pet_values=[];shade_values=[]
        for (u,v,k),heading in zip(row.edge_refs,row.headings):
            e=graph[u][v][k] if graph.is_multigraph() else graph[u][v]
            shade=min(1,float(e.get('shade_fraction',0))+extra_shade_pct/100)
            sharp=abs((environment['solar_azimuth_deg']-heading+180)%360-180)
            direct=environment.get('direct_normal_radiation_wm2',environment.get('radiation_wm2',0))
            mrt=mean_radiant_temperature_c(air_temperature_c=float(e['temperature_c']),
                direct_normal_radiation_wm2=direct,diffuse_radiation_wm2=environment.get('diffuse_radiation_wm2',0),
                solar_elevation_deg=environment['solar_elevation_deg'],solar_horizontal_angle_deg=min(sharp,180),
                shade_fraction=shade,sky_view_factor=float(e.get('sky_view_factor',1)),
                relative_humidity_pct=environment['humidity_pct'],cloud_cover_pct=environment.get('cloud_cover_pct',0),
                surface_albedo=float(e.get('surface_albedo',.2)))
            pet_values.append(segment_pet_c(float(e['temperature_c']),mrt,float(e.get('wind_mps',environment['wind_mps'])),environment['humidity_pct'],profile))
            shade_values.append(shade)
        after_pet.append(sum(pet_values)/len(pet_values));after_shade.append(sum(shade_values)/len(shade_values))
    selected['scenario_pet_c']=after_pet;selected['scenario_shade']=after_shade
    weight=lambda col:float((selected[col]*selected.length_m).sum()/selected.length_m.sum())
    return {'cell_id':cell_id,'extra_shade_percentage_points':extra_shade_pct,
        'network_km':float(selected.length_m.sum()/1000),
        'baseline_mean_pet_c':weight('pet_c'),'scenario_mean_pet_c':weight('scenario_pet_c'),
        'modeled_pet_change_c':weight('scenario_pet_c')-weight('pet_c'),
        'baseline_shade_pct':100*weight('shade_fraction'),
        'scenario_shade_pct':100*weight('scenario_shade'),
        'assumptions':'Shade-only sensitivity on clipped segments; temperature, SVF, wind and albedo fixed. No measured cooling or health benefit.'}


def build_planning_map(grid, places, metric='mean_pet_c'):
    import folium
    from branca.colormap import LinearColormap
    geo=grid.to_crs(4326)
    center=geo.geometry.union_all().centroid
    m=folium.Map(location=[center.y,center.x],zoom_start=14,tiles='OpenStreetMap',control_scale=True)
    vals=geo[metric].dropna();lo=float(vals.min()) if len(vals) else 0;hi=float(vals.max()) if len(vals) else 1
    scale=LinearColormap(['#ffffcc','#78c679','#006837'] if metric=='mean_shade_pct' else ['#fff7bc','#fc8d59','#b30000'],vmin=lo,vmax=hi if hi>lo else lo+1,caption=metric)
    features=json.loads(geo.to_json())
    def style(f):
        v=f['properties'].get(metric)
        return {'fillColor':scale(v) if v is not None else '#bdbdbd','color':'#555','weight':.4,'fillOpacity':.65}
    folium.GeoJson(features,name='Campus screening cells',style_function=style,
        tooltip=folium.GeoJsonTooltip(fields=['cell_id','mean_pet_c','mean_shade_pct','hot_low_shade_km','mean_refuge_distance_m'],aliases=['Cell','Modeled PET (C)','Shade (%)','Hot + low shade (km)','Listed indoor site distance (m)'],localize=True)).add_to(m)
    for p in places:
        if p['category'] in ('indoor_refuge','cooling_center'):folium.CircleMarker([p['latitude'],p['longitude']],radius=5,color='#7c3aed',fill=True,tooltip=p['name']).add_to(m)
    scale.add_to(m);return m
