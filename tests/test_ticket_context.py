"""Coordinator test of durable references after whole-exchange archival."""
import copy
import json
from local_data_agent.context_budget import prepare_history, estimate_tokens


def test_archival_keeps_latest_ticket_ids_and_real_scope_without_repeating_quotes():
    searches = {f'search_{i}': {'source_id':'tickets','filters':{'month':'2026-09'},
                 'source_ticket_count':3000,'population_ticket_count':128,
                 'matched_ticket_count':23,'returned_ticket_count':6,
                 'evidence':[{'evidence_id':f'ticket:tickets:T{i}:0-9','text':'NEVER_REINSERT_THIS_QUOTE'*1000}]}
                for i in range(5)}
    state = {'question':'explain','plan_version':1,'plan':[],'evidence':[],
             'artifacts':{},'observed_facts':[],'open_questions':[],'ticket_searches':searches}
    history = [{'role':'system','content':'agent'},{'role':'user','content':'explain'}]
    for i in range(6):
        history.extend([{'type':'function_call','call_id':f'c{i}','name':'inspect_artifact','arguments':'{}'},
                        {'type':'function_call_output','call_id':f'c{i}','output':json.dumps({'note':'x'*2000})}])
    before = copy.deepcopy((history,state))
    assert estimate_tokens(history)>2500
    prepared=prepare_history(history,maximum=2500,state=state)
    summary=next(i['content'] for i in prepared if i.get('role')=='user' and '{"question":' in i.get('content',''))
    payload=json.loads(summary[summary.index('{'):])
    refs=payload['ticket_searches']
    assert [r['search_id'] for r in refs]==['search_1','search_2','search_3','search_4']
    assert all(r['population_ticket_count']==128 and r['matched_ticket_count']==23 for r in refs)
    assert refs[-1]['evidence_ids']==['ticket:tickets:T4:0-9']
    assert 'NEVER_REINSERT_THIS_QUOTE' not in summary
    assert estimate_tokens(prepared)<=2500
    assert (history,state)==before
    assert {i['call_id'] for i in prepared if i.get('type')=='function_call'} == {
        i['call_id'] for i in prepared if i.get('type')=='function_call_output'}
