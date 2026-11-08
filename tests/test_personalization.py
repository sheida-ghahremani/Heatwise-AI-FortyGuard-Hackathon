import json
from unittest.mock import patch, Mock
import networkx as nx
import pytest
import requests
from heatwise.models import UserProfile, AgeGroup, Activity, Clothing
from heatwise.routing import prepare_graph, calculate_routes, route_temperature_profile, segment_pet_c
from heatwise.assistant import answer_question, _context_text


def graph():
    g = nx.MultiDiGraph()
    for n in range(4):
        g.add_node(n, x=-96.34 + n*.001, y=30.61)
    for a,b,t in [(0,1,32),(1,3,32),(0,2,35),(2,3,35)]:
        g.add_edge(a,b,length=100,temperature_c=t,shade_fraction=.2)
    return g


@pytest.mark.parametrize('profile', [UserProfile(AgeGroup.OLDER), UserProfile(activity=Activity.JOGGING), UserProfile(clothing=Clothing.PROTECTIVE)])
def test_profile_cost_changes_path_and_does_not_mutate_shared_graph(profile):
    # Controlled PET response tests that changed profile costs reach the solver.
    base=graph()
    def pet(t,m,w,h,p):
        return 31 if (t==32)==(p==UserProfile()) else 40
    with patch('heatwise.routing.segment_pet_c', side_effect=pet):
        g1=prepare_graph(base,UserProfile(),44,2)
        g2=prepare_graph(base,profile,44,2)
    r1=calculate_routes(g1,0,3)[1];r2=calculate_routes(g2,0,3)[1]
    assert r1.nodes==[0,1,3] and r2.nodes==[0,2,3]
    assert 'heat_cost' not in base[0][1][0]
    for g,r in [(g1,r1),(g2,r2)]:
        assert r.heat_score==pytest.approx(r.pet_exposure_load)
        assert r.pet_exposure_load==pytest.approx(sum(g[a][b][0]['pet_heat_cost'] for a,b in zip(r.nodes,r.nodes[1:])))


def test_parallel_edge_profile_uses_same_objective_as_route():
    g=graph();g.add_edge(0,1,length=110,temperature_c=28,shade_fraction=.9)
    p=UserProfile();g=prepare_graph(g,p,44,2)
    route=calculate_routes(g,0,1)[1]
    expected=min(g[0][1].values(),key=lambda e:e['heat_cost'])
    assert route_temperature_profile(g,route)[0]['temperature_c']==expected['temperature_c']


def test_real_pet_uses_each_physiological_input():
    profiles=[UserProfile(),UserProfile(AgeGroup.OLDER),UserProfile(activity=Activity.JOGGING),UserProfile(clothing=Clothing.PROTECTIVE)]
    values=[segment_pet_c(36,55,1.3,44,p,exact=True) for p in profiles]
    assert all(v!=values[0] for v in values[1:])


@pytest.mark.parametrize('failure',['length','network','table'])
def test_incomplete_online_reply_falls_back_to_all_three_routes(failure):
    p=UserProfile(clothing=Clothing.PROTECTIVE);g=prepare_graph(graph(),p,44,2);routes=calculate_routes(g,0,3)
    body={'choices':[{'finish_reason':'length' if failure=='length' else 'stop','message':{'content':'| broken table' if failure=='table' else 'incomplete'}}]}
    response=Mock(ok=True);response.json.return_value=body
    with patch.dict('os.environ',{'GROQ_API_KEY':'test'},clear=True),patch('heatwise.assistant.requests.post',side_effect=requests.Timeout() if failure=='network' else None,return_value=response):
        answer,mode=answer_question('Compare routes',routes,p,'A','B',44,2)
    assert mode.startswith('Local fallback')
    assert all(r.name in answer for r in routes)
    context=json.loads(_context_text(routes,p,'A','B',44,2))
    assert context['profile']['clothing_clo']==1
    assert context['routes'][0]['pet_load_above_29_c_min']==round(routes[0].pet_exposure_load,3)


def test_online_explainer_preserves_authoritative_numbers():
    p=UserProfile();g=prepare_graph(graph(),p,44,2);routes=calculate_routes(g,0,3)
    response=Mock(ok=True)
    response.json.return_value={'choices':[{'finish_reason':'stop','message':{'content':'Fastest prioritizes travel time. Lowest Heat Risk minimizes modeled exposure.'}}]}
    with patch.dict('os.environ',{'GROQ_API_KEY':'test'},clear=True),patch('heatwise.assistant.requests.post',return_value=response):
        answer,mode=answer_question('Compare',routes,p,'A','B',44,2)
    assert mode.startswith('Groq')
    for r in routes:
        assert f'{r.pet_exposure_load:.1f} C-min' in answer
    response.json.return_value['choices'][0]['message']['content']='PET is 999 degrees.'
    with patch.dict('os.environ',{'GROQ_API_KEY':'test'},clear=True),patch('heatwise.assistant.requests.post',return_value=response):
        answer,mode=answer_question('Compare',routes,p,'A','B',44,2)
    assert mode.startswith('Local fallback') and '999' not in answer
