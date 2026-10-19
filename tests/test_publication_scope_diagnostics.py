"""Coordinator acceptance with real tools, SQLite and Parquet."""
import json
from pathlib import Path

import pytest

from local_data_agent.agent_loop import execute_calls
from local_data_agent.artifacts import checked_run_dir, file_hash
from local_data_agent.connectors import snapshot_sources
from local_data_agent.fixtures import bootstrap
from local_data_agent.state import StateStore
from local_data_agent.tools import ToolContext


@pytest.fixture
def ctx(tmp_path):
    bootstrap(tmp_path)
    store = StateStore(tmp_path)
    state = store.create('cash net', 'live', 'cash-net-v1', 'cash-net-v1')
    run = checked_run_dir(tmp_path, state['run_id'])
    manifest = snapshot_sources(tmp_path, run)
    state['manifest'] = manifest
    context = ToolContext(tmp_path, run, state, manifest, store, None)
    context.dispatch('bind_metric', {'metric_id':'CASH_NET_v1'})
    try: yield context
    finally: context.retriever.close()


def load(identifier, table, sql=None):
    result = dict(step_id=identifier, type='LOAD', output_name=identifier, source_id='business', table=table)
    if sql is not None: result['sql'] = sql
    return result


def select(identifier, deps, sql):
    return dict(step_id=identifier,type='SELECT',output_name=identifier,depends_on=deps,sql=sql)


def plan(ctx, steps):
    ctx.dispatch('propose_plan', {'steps_json':json.dumps(steps),'reason':'coordinator acceptance'})


def test_actual_upstream_limit_diagnostic_preserves_partial_results_and_replay(ctx):
    sql = 'SELECT o.order_id FROM orders o CROSS JOIN orders a CROSS JOIN orders b CROSS JOIN orders c CROSS JOIN orders d CROSS JOIN orders e'
    plan(ctx, [load('refunds','refunds'),load('bad_orders','orders',sql),
               select('aggregate',['refunds','bad_orders'],'SELECT COUNT(*) AS order_count FROM bad_orders')])
    hashes = {s['path']:file_hash(s['path']) for s in ctx.manifest['sources']}
    ctx.state['status'] = 'RUNNING'
    call = {'call_id':'actual_limit','name':'execute_step','arguments':{'plan_version':1,'step_id':'aggregate'}}
    execute_calls(ctx,[call])
    output = json.loads(ctx.state['history'][-1]['output'])
    assert output['error']=='ValueError' and 'row limit' in output['message'].lower()
    assert output['failed_step']==dict(step_id='bad_orders',type='LOAD',output_name='bad_orders',source_id='business',table='orders')
    assert output['executed_outputs']==['refunds']
    preserved = dict(ctx.state['artifacts']['refunds'])
    assert ctx.state['repair_counts']=={'aggregate':1}
    execute_calls(ctx,[call])
    assert json.loads(ctx.state['history'][-1]['output'])==output
    assert ctx.state['artifacts']['refunds']==preserved
    assert ctx.state['repair_counts']=={'aggregate':1}
    assert all(file_hash(p)==h for p,h in hashes.items())


def test_missing_comparison_scope_names_reports_schema_without_amounts(ctx):
    plan(ctx,[load('orders','orders'),select('bad',['orders'],"SELECT '2026-08' AS label_month,CAST(SUM(paid_cents) AS BIGINT) AS base_net_cents FROM orders WHERE status='paid'")])
    ctx.dispatch('execute_step',{'plan_version':1,'step_id':'bad'})
    result = ctx.reconcile({'bad':ctx.state['artifacts']['bad']})
    assert not result['passed']
    details = result['checks'][0]['details']
    assert details['required_scope_columns']==['base_month','current_month']
    assert details['actual_columns']==['base_net_cents','label_month']
    assert details['counterexamples']==[{'group':'unsupported','error':'missing_scope_columns'}]
    assert 'expected' not in details and result['verified_columns']=={}


@pytest.mark.parametrize('table,id_col,amount,status',[('orders','order_id','paid_cents','paid'),('refunds','refund_id','refund_cents','succeeded')])
def test_registered_record_details_remain_verified_without_month(ctx,table,id_col,amount,status):
    plan(ctx,[load('records',table),select('detail',['records'],f"SELECT {id_col},{amount} FROM records WHERE status='{status}'")])
    ctx.dispatch('execute_step',{'plan_version':1,'step_id':'detail'})
    result = ctx.reconcile({'detail':ctx.state['artifacts']['detail']},publication=True)
    assert result['passed'] and amount in result['verified_columns']['detail']
