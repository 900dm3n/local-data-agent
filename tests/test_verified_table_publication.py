"""Coordinator acceptance of deterministic publication, not model intelligence."""
import json
from pathlib import Path

import pytest

from local_data_agent.artifacts import checked_run_dir, file_hash
from local_data_agent.connectors import snapshot_sources
from local_data_agent.fixtures import bootstrap
from local_data_agent.replay import base_steps, select
from local_data_agent.state import StateStore
from local_data_agent.tools import ToolContext


@pytest.fixture
def ctx(tmp_path):
    bootstrap(tmp_path)
    store=StateStore(tmp_path)
    state=store.create('cash net','live','cash-net-v1','cash-net-v1')
    run=checked_run_dir(tmp_path,state['run_id']);manifest=snapshot_sources(tmp_path,run);state['manifest']=manifest
    c=ToolContext(tmp_path,run,state,manifest,store,None)
    try:yield c
    finally:c.retriever.close()


def prepare(ctx, scale=1):
    fields=f"CAST(SUM(CASE WHEN event_type='payment' THEN signed_cents ELSE 0 END)*{scale} AS BIGINT) AS paid_cents, CAST(-SUM(CASE WHEN event_type='refund' THEN signed_cents ELSE 0 END)*{scale} AS BIGINT) AS refund_cents, CAST(SUM(signed_cents)*{scale} AS BIGINT) AS net_cents"
    steps=base_steps()+[select('totals',['cash_events'],f'SELECT month,{fields} FROM cash_events GROUP BY month'),
                        select('by_channel',['cash_events'],f'SELECT month,channel_code,{fields} FROM cash_events GROUP BY month,channel_code')]
    ctx.dispatch('bind_metric',{'metric_id':'CASH_NET_v1'})
    ctx.dispatch('propose_plan',{'steps_json':json.dumps(steps),'reason':'deterministic publication'})
    for name in ('totals','by_channel'):ctx.dispatch('execute_step',{'plan_version':1,'step_id':name})


def test_verified_publication_has_immutable_table_proof_and_no_model_summary(ctx):
    prepare(ctx)
    result=ctx.dispatch('publish_verified_tables',{'plan_version':1,'artifact_ids':['totals','by_channel'],'open_questions':[]})
    assert ctx.state['status']=='SUCCEEDED' and ctx.state['narrative_verification']=='host_deterministic_tables'
    folder=Path(ctx.state['report_path']).parent
    proof=json.loads((folder/'table_narrative.json').read_text(encoding='utf-8'))
    assert proof['source_artifact_ids']
    assert len(proof['groups'])==3
    assert '业务原因尚未证明' in Path(ctx.state['report_path']).read_text(encoding='utf-8')
    assert all(g['delta_cents']<=0 for g in proof['groups'])
    with pytest.raises(ValueError):
        ctx.dispatch('publish_verified_tables',{'plan_version':1,'artifact_ids':['totals'],'open_questions':[],'summary':'伪造说明'})


def test_wrong_source_amounts_cannot_be_published_with_verified_marker(ctx):
    prepare(ctx,scale=2)
    result=ctx.dispatch('publish_verified_tables',{'plan_version':1,'artifact_ids':['totals','by_channel'],'open_questions':[]})
    assert result.get('error')=='publication_validation_failed'
    assert ctx.state['status']!='SUCCEEDED'
    assert not (ctx.run_dir/'report.md').exists()


def test_legacy_followup_does_not_copy_stale_table_narrative_proof(ctx):
    prepare(ctx)
    ctx.dispatch('publish_verified_tables',{'plan_version':1,'artifact_ids':['totals','by_channel'],'open_questions':[]})
    old=Path(ctx.state['report_path']).parent
    hashes={p.name:file_hash(p) for p in old.iterdir() if p.is_file()}
    ctx.state['status']='RUNNING'
    result=ctx.dispatch('publish_tables',{'plan_version':1,'artifact_ids':['totals'],'summary':'仅展示已核对的月度金额，业务原因未验证。','open_questions':[]})
    new=Path(ctx.state['report_path']).parent
    assert new!=old and not(new/'table_narrative.json').exists()
    assert ctx.state['narrative_verification']=='model_prose_requires_review'
    assert all(file_hash(old/name)==h for name,h in hashes.items())


def test_stale_version_and_duplicate_artifact_rejected_before_publication(ctx):
    prepare(ctx)
    with pytest.raises(ValueError):ctx.dispatch('publish_verified_tables',{'plan_version':0,'artifact_ids':['totals'],'open_questions':[]})
    with pytest.raises(ValueError):ctx.dispatch('publish_verified_tables',{'plan_version':1,'artifact_ids':['totals','totals'],'open_questions':[]})
    assert ctx.state['status']!='SUCCEEDED' and not(ctx.run_dir/'report.md').exists()


def test_integral_decimal_aggregates_are_accepted_without_float_conversion(ctx):
    ctx.dispatch('bind_metric',{'metric_id':'CASH_NET_v1'})
    steps=base_steps()+[select('totals',['cash_events'],"SELECT month,SUM(CASE WHEN event_type='payment' THEN signed_cents ELSE 0 END) AS paid_cents,-SUM(CASE WHEN event_type='refund' THEN signed_cents ELSE 0 END) AS refund_cents,SUM(signed_cents) AS net_cents FROM cash_events GROUP BY month")]
    ctx.dispatch('propose_plan',{'steps_json':json.dumps(steps),'reason':'exact Decimal sum'})
    ctx.dispatch('execute_step',{'plan_version':1,'step_id':'totals'})
    result=ctx.dispatch('publish_verified_tables',{'plan_version':1,'artifact_ids':['totals'],'open_questions':[]})
    assert result['status']=='SUCCEEDED'
    proof=json.loads((Path(result['report_path']).parent/'table_narrative.json').read_text(encoding='utf-8'))
    assert all(type(g['delta_cents']) is int for g in proof['groups'])


def test_large_table_is_rejected_before_materialization(ctx,monkeypatch):
    from types import SimpleNamespace
    import local_data_agent.tools as tools
    prepare(ctx)
    monkeypatch.setattr(tools.pq,'ParquetFile',lambda path:SimpleNamespace(metadata=SimpleNamespace(num_rows=21)))
    def forbidden_read(*args,**kwargs):raise AssertionError('Oversized table must not be read')
    monkeypatch.setattr(tools.pq,'read_table',forbidden_read)
    with pytest.raises(ValueError,match='1..20 rows'):
        ctx.dispatch('publish_verified_tables',{'plan_version':1,'artifact_ids':['totals'],'open_questions':[]})
