"""Coordinator integration acceptance of the actual public tool contracts."""
import json
from pathlib import Path
import pytest
from local_data_agent.fixtures import bootstrap
from local_data_agent.connectors import snapshot_sources
from local_data_agent.state import StateStore
from local_data_agent.artifacts import checked_run_dir
from local_data_agent.tools import ToolContext
from local_data_agent.replay import base_steps, select

@pytest.fixture
def ctx(tmp_path):
    bootstrap(tmp_path);store=StateStore(tmp_path)
    state=store.create('cash net','live','cash-net-v1','cash-net-v1');run=checked_run_dir(tmp_path,state['run_id'])
    manifest=snapshot_sources(tmp_path,run);state['manifest']=manifest
    c=ToolContext(tmp_path,run,state,manifest,store,None)
    try:yield c
    finally:c.retriever.close()

def plan(c,scale=1):
    columns=f"CAST(SUM(CASE WHEN event_type='payment' THEN signed_cents ELSE 0 END)*{scale} AS BIGINT) AS paid_cents,CAST(-SUM(CASE WHEN event_type='refund' THEN signed_cents ELSE 0 END)*{scale} AS BIGINT) AS refund_cents,CAST(SUM(signed_cents)*{scale} AS BIGINT) AS net_cents"
    steps=base_steps()+[select('totals',['cash_events'],f'SELECT month,{columns} FROM cash_events GROUP BY month'),select('by_channel',['cash_events'],f'SELECT month,channel_code,{columns} FROM cash_events GROUP BY month,channel_code')]
    c.dispatch('bind_metric',{'metric_id':'CASH_NET_v1'})
    c.dispatch('propose_plan',{'steps_json':json.dumps(steps),'reason':'period comparison'})
    for step in ('totals','by_channel'):c.dispatch('execute_step',{'plan_version':1,'step_id':step})
    return dict(plan_version=1,month_artifact='totals',channel_artifact='by_channel',base_month='2026-08',current_month='2026-09',ticket_search_id='',open_questions=[])

def search(c):
    return c.dispatch('search_tickets',dict(query='',source_id='tickets',month='',channel_code='',limit=2))

def test_new_publication_quotes_and_checked_analysis(ctx):
    args=plan(ctx);s=search(ctx);args['ticket_search_id']=s['search_id']
    r=ctx.dispatch('publish_analysis',args);assert r['status']=='SUCCEEDED'
    pub=Path(r['report_path']).parent;analysis=json.loads((pub/'analysis.json').read_text(encoding='utf-8'))
    evidence=json.loads((pub/'evidence.json').read_text(encoding='utf-8'))
    assert analysis['ticket_evidence']==s['evidence']==evidence
    assert analysis['verification_scope']['derived_event_months']=='passed'
    assert analysis['ticket_scope']['population_ticket_count']==s['population_ticket_count']
    assert next(x for x in analysis['statements'] if x['kind']=='month_delta')['values']['net_delta_cents']==-7000
    assert ctx.state['publications'][-1]['narrative_verification']=='host_deterministic_evidence'
    assert all(e['evidence_id'] in r['answer'] for e in evidence)

@pytest.mark.parametrize('bad',['missing','tamper','source_change','stale_version'])
def test_invalid_saved_evidence_or_version_never_publishes(ctx,bad):
    args=plan(ctx);s=search(ctx);args['ticket_search_id']=s['search_id']
    if bad=='missing':args['ticket_search_id']='ticket_search_missing'
    elif bad=='tamper':ctx.state['ticket_searches'][s['search_id']]['evidence'][0]['text']='fabricated'
    elif bad=='stale_version':args['plan_version']=0
    else:
        source=next(s for s in ctx.manifest['sources'] if s['source_id']=='tickets')
        with open(source['path'],'ab') as f:f.write(b'\n')
    with pytest.raises(ValueError):ctx.dispatch('publish_analysis',args)
    assert not(ctx.run_dir/'report.md').exists()
    assert not ctx.state.get('publications')

def test_self_consistent_wrong_finance_still_fails_source_ledger(ctx):
    args=plan(ctx,scale=2)
    r=ctx.dispatch('publish_analysis',args)
    assert r.get('error')=='publication_validation_failed'
    assert ctx.state['status']!='SUCCEEDED'
    assert not(ctx.run_dir/'report.md').exists()

def test_legacy_followup_keeps_old_files_without_stale_analysis(ctx):
    args=plan(ctx);s=search(ctx);args['ticket_search_id']=s['search_id']
    first=ctx.dispatch('publish_analysis',args);old=Path(first['report_path']).parent
    before={p.name:p.read_bytes() for p in old.iterdir() if p.is_file()}
    legacy=ctx.dispatch('publish_tables',dict(plan_version=1,artifact_ids=['totals','by_channel'],summary='Source-bound facts; business reasons remain unverified.',open_questions=[]))
    assert legacy['status']=='SUCCEEDED'
    assert not(Path(legacy['report_path']).parent/'analysis.json').exists()
    assert {p.name:p.read_bytes() for p in old.iterdir() if p.is_file()}==before
    assert ctx.state['publications'][-1]['narrative_verification']=='model_prose_requires_review'
