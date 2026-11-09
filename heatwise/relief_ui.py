from __future__ import annotations

import json
from datetime import datetime
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from .relief import (load_catalog,catalog_frame,places_geojson,routes_via_place,
                     comparison_frame,build_relief_map,CATEGORY_LABELS,place_connection,listed_closed)
from .planning import segment_frame,planning_grid,build_planning_map,shade_scenario


def _signature(context):
    return json.dumps(context,sort_keys=True,default=str)


@st.fragment
def render_relief_tab(graph,routes,profile,area,origin,destination,context):
    st.subheader('Cooling facilities, indoor refuges and parks')
    st.caption('Source-reviewed catalog: 8 November 2026. Facility hours are reference information, not live or historical opening confirmation. Park labels do not imply a measured cooling benefit.')
    regional=st.checkbox('Show city-listed facilities beyond the prepared study extent',False,key='relief_regional')
    selected_at=datetime.fromisoformat(context["environmental_time"])
    frame=catalog_frame(graph,area,origin,destination,include_regional=regional,when=selected_at)
    kind=st.radio('Places to display',['All places','Indoor facilities','Parks and gardens'],horizontal=True,key='relief_kind')
    if kind=='Indoor facilities':frame=frame[frame.category!='park']
    if kind=='Parks and gardens':frame=frame[frame.category=='park']
    items=[p for p in load_catalog()['items'] if p['id'] in set(frame.id)]
    counts=st.columns(3)
    counts[0].metric('City-listed facilities shown',sum(p['category']=='cooling_center' for p in items))
    counts[1].metric('Campus indoor sites shown',sum(p['category']=='indoor_refuge' for p in items))
    counts[2].metric('Parks / gardens shown',sum(p['category']=='park' for p in items))
    st_folium(build_relief_map(items,graph=graph,origin=origin,destination=destination),height=430,use_container_width=True,returned_objects=[],key='relief_catalog_map')
    if frame.empty:
        st.info('No source-reviewed places in this category for this campus. Regional facilities can be displayed with the checkbox above.')
        return
    display=frame[['name','type','access','hours','routable','schedule_status','reason','straight_line_from_origin_m','source_url']].copy()
    display=display.rename(columns={'straight_line_from_origin_m':'Straight-line distance from origin (m)','source_url':'Source'})
    st.dataframe(display,hide_index=True,column_config={'Source':st.column_config.LinkColumn('Source')})
    st.download_button('Download place catalog (CSV)',frame.drop(columns=['node']).assign(**context).to_csv(index=False).encode(),file_name='heatwise_relief_places.csv',mime='text/csv',on_click='ignore')
    st.download_button('Download parks and facilities (GeoJSON)',json.dumps(places_geojson(items),ensure_ascii=False),file_name='heatwise_relief_places.geojson',mime='application/geo+json',on_click='ignore')
    st.markdown('#### Compare a route with a relief stop')
    by_id={p['id']:p for p in items}
    ordered=sorted(by_id,key=lambda k:(by_id[k]['category']!='indoor_refuge',by_id[k]['name']))
    choice=st.selectbox('Relief stop',ordered,format_func=lambda k:by_id[k]['name'],key='relief_stop_'+area)
    p=by_id[choice];row=frame[frame.id==choice].iloc[0]
    st.write(p['designation'])
    st.write('Access: '+p['access']+'. Hours: '+p['hours']+'.')
    st.link_button('Check official/source information',p.get('hours_url',p['source_url']))
    st.write(row.schedule_status)
    if listed_closed(p,selected_at):
        st.info('The published weekly schedule lists this facility as closed at the selected hour. Select another stop or departure time.')
        return
    if not bool(row.routable):
        st.info(row.reason+'. The tool does not extend routing beyond the prepared study extent.')
        return
    st.caption(f"Place reference is {row.connector_m:.0f} m from the nearest mapped path. The calculation ends at that path access point. Unmapped connectors, time at the stop and indoor recovery are not included.")
    sig=(_signature(context),choice)
    if st.button('Calculate routes via this stop',type='primary',key='relief_calculate'):
        with st.spinner('Comparing the same three objectives with a required stop...'):
            detours,connection=routes_via_place(graph,origin,destination,profile,p,area,when=selected_at)
            result=comparison_frame(routes,detours,p,context)
            st.session_state.relief_result=(sig,detours,result)
    stored=st.session_state.get('relief_result')
    if stored and stored[0]==sig:
        detours,result=stored[1:]
        st.dataframe(result[['name','duration_min','distance_m','average_pet_c','pet_exposure_load','added_outdoor_minutes','added_pet_load_c_min']],hide_index=True)
        st.caption('Positive deltas mean additional outdoor travel/exposure. This is a detour comparison, not a calculation of physiological recovery inside a building or a park.')
        st_folium(build_relief_map([p],graph,detours,origin,destination),height=440,use_container_width=True,returned_objects=[],key='relief_detour_map')
        st.download_button('Download stop-route comparison (CSV)',result.to_csv(index=False).encode(),file_name='heatwise_relief_stop_comparison.csv',mime='text/csv',on_click='ignore')


def render_planner_tab(graph,profile,area,context,environment):
    st.subheader('Campus heat planning dashboard')
    st.write('Screen the modeled network for hot segments, limited shade and distance to listed campus indoor sites. Select thresholds, inspect priority cells and export the results.')
    st.caption('This dashboard summarizes network conditions for the selected hour and profile. Known weekly closures are excluded. Unknown opening hours remain unconfirmed. It does not estimate population exposure, verify live access, or predict health outcomes.')
    cols=st.columns(4)
    size=cols[0].selectbox('Grid size (m)',[100,200,400],index=1,key='planner_size')
    pet=cols[1].number_input('PET threshold (°C)',value=41.0,min_value=20.0,max_value=70.0,key='planner_pet')
    shade=cols[2].number_input('Low-shade threshold (%)',value=20.0,min_value=0.0,max_value=100.0,key='planner_shade')
    distance=cols[3].number_input('Indoor-site distance threshold (m)',value=500.0,min_value=50.0,max_value=5000.0,step=50.0,key='planner_distance')
    selected_at=datetime.fromisoformat(context['environmental_time'])
    places=[p for p in load_catalog()['items'] if (p['area']==area or (p['area']=='regional' and place_connection(graph,p,area)['routable'])) and not listed_closed(p,selected_at)]
    sig=(_signature(context),size,pet,shade,distance)
    if st.button('Build campus dashboard',type='primary',key='planner_build'):
        with st.spinner('Aggregating the network and calculating distances to listed indoor sites...'):
            segments=segment_frame(graph,places,area)
            grid=planning_grid(segments,cell_size_m=size,pet_threshold_c=pet,shade_threshold_pct=shade,access_threshold_m=distance)
            st.session_state.planner_result=(sig,segments,grid)
    stored=st.session_state.get('planner_result')
    if not stored or stored[0]!=sig:
        st.info('Build the dashboard to apply these settings. A new profile, hour or threshold requires rebuilding.')
        return
    segments,grid=stored[1:];total=float(grid.network_km.sum())
    cols=st.columns(4)
    cols[0].metric('Unique network represented',f'{total:.2f} km')
    cols[1].metric('Above selected PET threshold',f'{100*grid.hot_network_km.sum()/total:.1f}%')
    cols[2].metric('Hot + low shade',f'{grid.hot_low_shade_km.sum():.2f} km')
    cols[3].metric('Hot + indoor-site distance gap',f'{grid.hot_refuge_gap_km.sum():.2f} km')
    unknown=float(grid.unknown_refuge_access_km.sum())
    st.caption(f'{len(grid)} occupied cells. {unknown:.2f} km has unknown/unreachable indoor-site access and is reported separately from a measured distance gap.')
    metric=st.selectbox('Planning map layer',['mean_pet_c','mean_shade_pct','hot_low_shade_km','mean_refuge_distance_m'],format_func=lambda s:{'mean_pet_c':'Mean modeled PET (°C)','mean_shade_pct':'Modeled shade (%)','hot_low_shade_km':'Hot + low-shade network (km)','mean_refuge_distance_m':'Network distance to listed indoor site (m)'}[s],key='planner_metric')
    st_folium(build_planning_map(grid,places,metric),height=470,use_container_width=True,returned_objects=[],key='planner_map')
    table=grid.drop(columns='geometry')
    st.markdown('#### Cells for shade-investment screening')
    st.caption('Rank: descending hot + low-shade network length, then mean PET. Reverse directions are grouped by endpoint pair, geometry and length. Segments are clipped to grid cells. Cell means use segment-length weights.')
    st.dataframe(table.head(20),hide_index=True)
    export=grid.assign(**context)
    st.download_button('Download planning grid (CSV)',export.drop(columns='geometry').to_csv(index=False).encode(),file_name='heatwise_planning_grid.csv',mime='text/csv',on_click='ignore')
    st.download_button('Download planning grid (GeoJSON)',export.to_crs(4326).to_json(),file_name='heatwise_planning_grid.geojson',mime='application/geo+json',on_click='ignore')
    st.markdown('#### Shade-only intervention sensitivity')
    cell=st.selectbox('Planning cell',grid.cell_id.tolist(),key='scenario_cell')
    extra=st.slider('Additional shade (percentage points)',0,80,30,5,key='scenario_extra')
    st.caption('Recalculate PET after increasing shade in this cell. Air temperature, wind, SVF and albedo remain fixed. This is an isolated model sensitivity, not a forecast of tree planting or measured cooling.')
    scenario_sig=(sig,cell,extra)
    if st.button('Evaluate shade scenario',key='scenario_run'):
        with st.spinner('Recalculating the selected cell...'):
            result=shade_scenario(graph,segments,cell,size,extra,profile,environment)
            st.session_state.shade_scenario=(scenario_sig,{**result,**context})
    scenario=st.session_state.get('shade_scenario')
    if scenario and scenario[0]==scenario_sig:
        result=scenario[1]
        st.dataframe(pd.DataFrame([result]),hide_index=True)
        st.download_button('Download shade scenario (CSV)',pd.DataFrame([result]).to_csv(index=False).encode(),file_name='heatwise_shade_scenario.csv',mime='text/csv',on_click='ignore')
