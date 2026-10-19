"""Exercise actual create_agent, native tool nodes and SQLite checkpoints offline."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from local_data_agent.artifacts import file_hash
from local_data_agent.fixtures import bootstrap
from local_data_agent.replay import base_steps, select
from langchain_data_agent.runner import LangChainRunner


class ScriptedChat(BaseChatModel):
    responses:list=Field(default_factory=list)
    observed:list=Field(default_factory=list)
    calls:int=0

    @property
    def _llm_type(self):return 'independent-scripted-chat'

    def bind_tools(self,tools,*,tool_choice=None,**kwargs):return self

    def _generate(self,messages,stop=None,run_manager=None,**kwargs):
        self.observed.append(deepcopy(messages))
        response=self.responses[self.calls];self.calls+=1
        if isinstance(response,Exception):raise response
        return ChatResult(generations=[ChatGeneration(message=deepcopy(response))])


def call(name,args,identifier=None):
    return AIMessage(content='',tool_calls=[{'name':name,'args':args,'id':identifier or name}])


def financial_sequence():
    return [call('bind_metric',{'metric_id':'CASH_NET_v1'}),
            call('propose_plan',{'steps_json':json.dumps(base_steps()),'reason':'independent fixture'}),
            call('execute_step',{'plan_version':1,'step_id':'monthly_total'}),
            call('publish_verified_tables',{'plan_version':1,'artifact_ids':['monthly_total'],'open_questions':[]})]


@pytest.fixture
def root(tmp_path):bootstrap(tmp_path);return tmp_path


def test_framework_performs_real_sql_publication_and_stops_without_extra_model_call(root):
    model=ScriptedChat(responses=financial_sequence())
    runner=LangChainRunner(root,model=model)
    result=runner.run('按现金净额比较八月和九月',mode='live')
    assert result['status']=='SUCCEEDED',result
    assert model.calls==4 and result['model_requests']==4
    assert result['framework']=='langchain.create_agent'
    assert result['narrative_verification']=='host_deterministic_tables'
    proof=json.loads((Path(result['report_path']).parent/'table_narrative.json').read_text(encoding='utf-8'))
    assert proof['groups'][0]['delta_cents']==-7000
    assert (root/'var/langchain_checkpoints.sqlite').is_file()
    assert result['langchain_checkpoint']['messages']>=10
    assert (Path(result['run_dir'])/'langchain_graph.mmd').is_file()


def test_multiple_tool_calls_are_paired_and_committed(root):
    batch=AIMessage(content='',tool_calls=[{'name':'inspect_source_checks','args':{},'id':'checks'},
        {'name':'search_context','args':{'query':'现金净额','source_ids':[],'types':[]},'id':'search'}])
    model=ScriptedChat(responses=[batch]+financial_sequence())
    runner=LangChainRunner(root,model=model)
    result=runner.run('现金净额比较',mode='live')
    assert result['status']=='SUCCEEDED',result
    history=runner.store.load(result['run_id'])['history']
    assert [i['call_id'] for i in history if i.get('type')=='function_call_output'][:2]==['checks','search']
    for messages in model.observed:
        requests={c['id'] for m in messages if isinstance(m,AIMessage) for c in m.tool_calls}
        replies={m.tool_call_id for m in messages if m.type=='tool'}
        assert requests==replies


def test_free_text_cannot_publish_and_model_budget_is_exact(root):
    model=ScriptedChat(responses=[AIMessage(content='已经算好了。')]*5)
    result=LangChainRunner(root,model=model,max_requests=2).run('现金净额',mode='live')
    assert result['status']=='FAILED' and result['stop_reason']=='model_request_budget_exceeded'
    assert model.calls==2 and result['model_requests']==2 and not result['report_path']


def test_pause_and_resume_in_fresh_runner_keeps_history_and_snapshot(root):
    model=ScriptedChat(responses=[call('ask_user',{'question':'请选择口径','reason':'收入不明确','choices':['现金净额','其他']})])
    first=LangChainRunner(root,model=model)
    result=first.run('分析收入',mode='live')
    assert result['status']=='WAITING_INPUT' and model.calls==1
    before=first.store.load(result['run_id']);source_hashes={s['path']:file_hash(s['path']) for s in before['manifest']['sources']}
    second=LangChainRunner(root,model=ScriptedChat(responses=financial_sequence()))
    result=second.resume(result['run_id'],answer='我选择现金净额')
    assert result['status']=='SUCCEEDED',result
    after=second.store.load(result['run_id'])
    assert after['history'][:len(before['history'])]==before['history']
    assert all(file_hash(path)==h for path,h in source_hashes.items())
    from langgraph.checkpoint.sqlite import SqliteSaver
    with SqliteSaver.from_conn_string(str(root/'var/langchain_checkpoints.sqlite')) as saver:
        saved=saver.get({'configurable':{'thread_id':result['run_id']}})
    ids=[m.id for m in saved['channel_values']['messages']]
    assert len(ids)==len(set(ids))


def test_followup_patch_reuses_artifacts_and_preserves_old_publication(root):
    first=LangChainRunner(root,model=ScriptedChat(responses=financial_sequence()))
    result=first.run('现金净额比较',mode='live');assert result['status']=='SUCCEEDED',result
    before=first.store.load(result['run_id'])
    old=Path(result['report_path']).parent;old_hashes={p.name:file_hash(p) for p in old.iterdir() if p.is_file()}
    new_step=select('channel_totals',['cash_events'],'SELECT month,channel_code,CAST(SUM(signed_cents) AS BIGINT) AS net_cents FROM cash_events GROUP BY month,channel_code')
    sequence=[call('patch_plan',{'base_plan_version':1,'upsert_steps_json':json.dumps([new_step]),'remove_step_ids':[],'reason':'add channels'},'patch2'),
              call('execute_step',{'plan_version':2,'step_id':'channel_totals'},'exec2'),
              call('publish_verified_tables',{'plan_version':2,'artifact_ids':['monthly_total','channel_totals'],'open_questions':[]},'pub2')]
    second=LangChainRunner(root,model=ScriptedChat(responses=sequence))
    result=second.resume(result['run_id'],question='补充全部渠道净额，继续沿用现金净额')
    assert result['status']=='SUCCEEDED',result
    after=second.store.load(result['run_id'])
    assert after['plan_version']==2 and len(after['publications'])==2
    assert all(after['artifacts'][k]['artifact_id']==v['artifact_id'] for k,v in before['artifacts'].items())
    assert all(file_hash(old/n)==h for n,h in old_hashes.items())


def test_failed_provider_attempt_can_resume_without_reexecuting_committed_tool(root):
    first=LangChainRunner(root,model=ScriptedChat(responses=[financial_sequence()[0],ConnectionError('synthetic network failure')]))
    result=first.run('现金净额',mode='live')
    assert result['status']=='FAILED' and result['model_requests']==2
    assert result['langchain_checkpoint']['checkpoint_id']
    assert result['langchain_checkpoint']['messages']>=3
    before=first.store.load(result['run_id'])
    second=LangChainRunner(root,model=ScriptedChat(responses=financial_sequence()[1:]))
    result=second.resume(result['run_id'])
    assert result['status']=='SUCCEEDED',result
    after=second.store.load(result['run_id'])
    calls=[i for i in after['history'] if i.get('type')=='function_call_output' and i['call_id']=='bind_metric']
    assert len(calls)==1 and after['history'][:len(before['history'])]==before['history']


def test_unknown_tool_has_persisted_reply_and_model_can_recover(root):
    model=ScriptedChat(responses=[call('unregistered_tool',{},'unknown')]+financial_sequence())
    runner=LangChainRunner(root,model=model)
    result=runner.run('现金净额',mode='live')
    assert result['status']=='SUCCEEDED',result
    history=runner.store.load(result['run_id'])['history']
    output=next(i for i in history if i.get('type')=='function_call_output' and i['call_id']=='unknown')
    assert 'Unknown tool' in output['output']


def test_terminal_publication_skips_remaining_batch_tool(root):
    last=financial_sequence()[-1]
    last.tool_calls.append({'name':'execute_step','args':{'plan_version':1,'step_id':'monthly_total'},'id':'after_done','type':'tool_call'})
    model=ScriptedChat(responses=financial_sequence()[:-1]+[last])
    runner=LangChainRunner(root,model=model)
    result=runner.run('现金净额',mode='live')
    assert result['status']=='SUCCEEDED' and model.calls==4,result
    history=runner.store.load(result['run_id'])['history']
    output=next(i for i in history if i.get('call_id')=='after_done')
    assert json.loads(output['output'])['status']=='SKIPPED'


def test_truncated_model_response_never_executes_its_tool_call(root):
    message=financial_sequence()[0];message.response_metadata={'finish_reason':'length'}
    model=ScriptedChat(responses=[message])
    runner=LangChainRunner(root,model=model)
    result=runner.run('现金净额',mode='live')
    state=runner.store.load(result['run_id'])
    assert result['status']=='FAILED' and result['stop_reason']=='model_response_incomplete'
    assert state['metric_binding'] is None and model.calls==1
    output=next(i for i in state['history'] if i.get('type')=='function_call_output')
    assert json.loads(output['output'])['status']=='SKIPPED'


def test_new_runner_never_enters_legacy_loop(root,monkeypatch):
    import local_data_agent.runner as legacy
    def forbidden(*args,**kwargs):raise AssertionError('Legacy model loop must not run')
    monkeypatch.setattr(legacy,'run_loop',forbidden)
    message=call('ask_user',{'question':'选择指标','reason':'待澄清','choices':['现金净额','其他']})
    result=LangChainRunner(root,model=ScriptedChat(responses=[message])).run('分析收入',mode='live')
    assert result['status']=='WAITING_INPUT'


def test_semantic_operator_model_uses_native_langchain_json_mode(root):
    from langchain_data_agent.model import ModelGateway
    message=AIMessage(content='{"items":[{"ticket_id":"T1","label":"其他","evidence_id":"e1","start_line":1,"end_line":1}]}')
    model=ScriptedChat(responses=[message]);gateway=ModelGateway(root,chat_model=model)
    result=gateway.semantic([{'ticket_id':'T1','text':'参考文本','evidence_id':'e1'}])
    assert result['items'][0]['ticket_id']=='T1' and model.calls==1
    assert model.observed[0][0].type=='system'


def test_context_preparation_failure_does_not_consume_request_budget(root,monkeypatch):
    from langchain_data_agent.model import ModelGateway
    import langchain_data_agent.runtime as runtime

    class Budget:
        attempts=0
        def before_request(self):self.attempts+=1

    def fail(*args,**kwargs):raise ValueError('Synthetic invalid provider context')

    budget=Budget();model=ScriptedChat(responses=financial_sequence())
    gateway=ModelGateway(root,chat_model=model);gateway.budget=budget
    monkeypatch.setattr(runtime,'prepare_history',fail)
    result=LangChainRunner(root,model=gateway).run('现金净额',mode='live')
    assert result['status']=='FAILED' and result['model_requests']==0
    assert budget.attempts==0 and model.calls==0
    assert result['langchain_checkpoint']['checkpoint_id']


def test_checkpoint_inspection_failure_keeps_original_provider_error(root,monkeypatch):
    from langgraph.graph.state import CompiledStateGraph

    def fail(*args,**kwargs):raise OSError('Synthetic checkpoint inspection failure')

    monkeypatch.setattr(CompiledStateGraph,'get_state',fail)
    runner=LangChainRunner(root,model=ScriptedChat(responses=[ConnectionError('Synthetic provider failure')]))
    result=runner.run('现金净额',mode='live')
    state=runner.store.load(result['run_id'])
    assert result['status']=='FAILED' and result['model_requests']==1
    assert 'LangChain model request failed: ConnectionError' in str(state['error'])
    assert state['checkpoint_inspection_error']=='OSError'


def test_cancel_before_first_request_never_calls_model(root):
    model=ScriptedChat(responses=financial_sequence())
    result=LangChainRunner(root,model=model,cancel_check=lambda:True).run('现金净额',mode='live')
    assert result['status']=='CANCELLED' and result['stop_reason']=='user_cancelled'
    assert result['model_requests']==0 and model.calls==0 and not result['artifacts']


def test_cancel_during_model_response_pairs_but_skips_tool_and_can_resume(root):
    import threading
    stop=threading.Event()

    class CancelDuringResponse(ScriptedChat):
        def _generate(self,*args,**kwargs):
            response=super()._generate(*args,**kwargs);stop.set();return response

    first=LangChainRunner(root,model=CancelDuringResponse(responses=[call('bind_metric',{'metric_id':'CASH_NET_v1'},'cancelled_bind')]),cancel_check=stop.is_set)
    result=first.run('现金净额',mode='live');before=first.store.load(result['run_id'])
    assert result['status']=='CANCELLED' and before['metric_binding'] is None
    replies=[i for i in before['history'] if i.get('type')=='function_call_output']
    assert len(replies)==1 and json.loads(replies[0]['output'])['status']=='SKIPPED'
    second=LangChainRunner(root,model=ScriptedChat(responses=financial_sequence()))
    result=second.resume(result['run_id']);after=second.store.load(result['run_id'])
    assert result['status']=='SUCCEEDED',result
    assert after['history'][:len(before['history'])]==before['history']


def test_cancelled_provider_error_is_cancelled_and_still_counts_attempt(root):
    import threading
    stop=threading.Event()
    class CancelWithFailure(ScriptedChat):
        def _generate(self,*args,**kwargs):stop.set();raise ConnectionError('Synthetic interrupted request')
    result=LangChainRunner(root,model=CancelWithFailure(),cancel_check=stop.is_set).run('现金净额',mode='live')
    assert result['status']=='CANCELLED' and result['model_requests']==1
    assert not result['artifacts'] and result['stop_reason']=='user_cancelled'


def test_resume_records_changed_provider_without_rewriting_history(root):
    from langchain_data_agent.model import ModelGateway
    first=ModelGateway(root,chat_model=ScriptedChat(responses=[call('ask_user',{'question':'选择指标','reason':'待澄清','choices':['现金净额','其他']})]))
    first._metadata.update(provider='deepseek',model='first-provider-model')
    runner=LangChainRunner(root,model=first);result=runner.run('分析收入',mode='live')
    before=runner.store.load(result['run_id'])
    second=ModelGateway(root,chat_model=ScriptedChat(responses=financial_sequence()))
    second._metadata.update(provider='mimo',model='second-provider-model')
    runner=LangChainRunner(root,model=second);result=runner.resume(result['run_id'],answer='现金净额')
    after=runner.store.load(result['run_id'])
    assert result['status']=='SUCCEEDED' and result['model']['provider']=='mimo'
    assert [s['model']['provider'] for s in result['model_sessions']]==['deepseek','mimo']
    assert after['history'][:len(before['history'])]==before['history']
