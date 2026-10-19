"""Regression cases discovered during persistence/integration review."""
import json
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from local_data_agent.agent_loop import execute_calls, pending_calls
from local_data_agent.artifacts import checked_run_dir
from local_data_agent.artifacts import file_hash
from local_data_agent.connectors import snapshot_sources
from local_data_agent.fixtures import bootstrap
from local_data_agent.replay import ReplayModelClient
from local_data_agent.runner import AgentRunner
from local_data_agent.state import StateStore
from local_data_agent.tools import ToolContext

@pytest.fixture
def context(tmp_path):
    bootstrap(tmp_path)
    store = StateStore(tmp_path)
    state = store.create('按现金净额口径分析', 'replay', 'cash-net-v1', 'cash-net-v1')
    directory = checked_run_dir(tmp_path, state['run_id'])
    manifest = snapshot_sources(tmp_path, directory)
    state['manifest'] = manifest
    state['status'] = 'RUNNING'
    store.save(state)
    return ToolContext(tmp_path, directory, state, manifest, store, ReplayModelClient())

def test_resume_contention_preserves_current_owners_state(tmp_path):
    runner = AgentRunner(tmp_path, model=ReplayModelClient())
    state = runner.store.create('question', 'replay', 'cash-net-v1', 'cash-net-v1')
    state['manifest'] = {'sources': [], 'documents': [], 'metric_contracts': []}
    runner.store.save(state)
    with runner.store.acquire(state['run_id']):
        before = runner.store.load(state['run_id'])
        try:
            runner.resume(state['run_id'])
        except RuntimeError:
            pass  # Contention may be raised or returned, but must never rewrite the run.
        after = runner.store.load(state['run_id'])
        assert after == before

def test_resume_client_configuration_failure_does_not_overwrite_active_owner(tmp_path, monkeypatch):
    runner = AgentRunner(tmp_path, model=ReplayModelClient())
    state = runner.store.create('question', 'live', 'cash-net-v1', 'cash-net-v1')
    state['manifest'] = {'sources': [], 'documents': [], 'metric_contracts': []}
    runner.store.save(state)
    def unavailable_client(_state):
        raise ValueError('simulated credential configuration missing in second caller')
    monkeypatch.setattr(runner, '_client', unavailable_client)
    with runner.store.acquire(state['run_id']):
        before = runner.store.load(state['run_id'])
        try:
            runner.resume(state['run_id'])
        except (RuntimeError, ValueError):
            pass
        assert runner.store.load(state['run_id']) == before

def test_tool_commit_crash_preserves_effect_and_complete_protocol_pair(context, monkeypatch):
    call = {'call_id': 'atomic_bind', 'name': 'bind_metric', 'arguments': {'metric_id': 'CASH_NET_v1'}}
    context.state['history'] = [{'type': 'function_call', **call, 'arguments': json.dumps(call['arguments'])}]
    context.store.save(context.state)
    original = context.store.finish_call
    def crash_after_commit(*args, **kwargs):
        original(*args, **kwargs)
        raise SystemExit('simulated process loss after durable call completion')
    monkeypatch.setattr(context.store, 'finish_call', crash_after_commit)
    with pytest.raises(SystemExit):
        execute_calls(context, [call])
    recovered = context.store.load(context.state['run_id'])
    assert recovered['metric_binding'] == 'CASH_NET_v1'
    assert pending_calls(recovered['history']) == []
    assert sum(item.get('type') == 'function_call_output' for item in recovered['history']) == 1

def test_bound_metric_uses_approved_version_even_with_stale_same_id(context):
    approved = context.manifest['metric_contracts'][0]
    stale = {**approved, 'status': 'STALE', 'timezone': 'Pacific/Honolulu', 'version': 0}
    context.manifest['metric_contracts'].insert(0, stale)
    context.dispatch('bind_metric', {'metric_id': 'CASH_NET_v1'})
    assert context.metric() == approved

def test_paused_tool_crash_recovery_pairs_output_before_user_answer(context, monkeypatch):
    arguments = {'question': '哪个口径？', 'reason': '需要明确业务定义', 'choices': ['现金净额', '其他']}
    call = {'call_id': 'atomic_pause', 'name': 'ask_user', 'arguments': arguments}
    context.state['history'] = [{'type': 'function_call', **call, 'arguments': json.dumps(arguments)}]
    context.store.save(context.state)
    original = context.store.finish_call
    def crash_after_commit(*args, **kwargs):
        original(*args, **kwargs)
        raise SystemExit('simulated loss after pause result commit')
    monkeypatch.setattr(context.store, 'finish_call', crash_after_commit)
    with pytest.raises(SystemExit):
        execute_calls(context, [call])
    monkeypatch.setattr(context.store, 'finish_call', original)
    class PairingModel:
        def respond(self, history, tools):
            output_index = next(index for index, item in enumerate(history)
                                if item.get('type') == 'function_call_output' and item.get('call_id') == 'atomic_pause')
            answer_index = next(index for index, item in enumerate(history)
                                if item.get('role') == 'user' and item.get('content') == '现金净额')
            assert output_index < answer_index, 'Tool output must precede resumed user input'
            next_call = {'type': 'function_call', 'call_id': 'next_pause', 'name': 'ask_user',
                         'arguments': json.dumps({'question': '测试暂停', 'reason': '结束协议测试', 'choices': []})}
            return {'protocol_items': [next_call], 'tool_calls': [next_call], 'status': 'completed', 'usage': {}}
    result = AgentRunner(context.root, model=PairingModel()).resume(context.state['run_id'], answer='现金净额')
    assert result['status'] == 'WAITING_INPUT'

def test_currency_claim_without_registered_business_verifier_cannot_publish(context):
    context.dispatch('bind_metric', {'metric_id': 'CASH_NET_v1'})
    steps = [
        {'step_id': 'orders', 'output_name': 'orders', 'type': 'LOAD', 'source_id': 'business', 'table': 'orders'},
        {'step_id': 'fabricated', 'output_name': 'fabricated', 'type': 'SELECT', 'depends_on': ['orders'],
         'sql': 'SELECT CAST(99900 AS BIGINT) AS fabricated_cents FROM orders LIMIT 1'}]
    context.dispatch('propose_plan', {'steps_json': json.dumps(steps), 'reason': 'test bounded but unsupported metric derivation'})
    context.dispatch('execute_step', {'plan_version': 1, 'step_id': 'fabricated'})
    claims = [{'claim_id': 'fake', 'artifact_id': 'fabricated', 'column': 'fabricated_cents', 'filter': {},
               'aggregation': 'single_row', 'display_unit': 'CNY_yuan', 'label': '现金净额'}]
    try:
        result = context.dispatch('finalize_run', {'plan_version': 1, 'claims_json': json.dumps(claims),
                                                 'summary': '现金净额来自计算结果。', 'open_questions': []})
    except ValueError:
        return  # Rejecting unsupported business derivations is an acceptable bounded failure.
    assert result.get('status') != 'SUCCEEDED', 'A bound scalar alone does not establish approved metric correctness'
    assert context.state['status'] != 'SUCCEEDED'

@pytest.mark.parametrize('delta_cents,percentage_offset,expected_pass', [
    (-7000, 0.0, True), (-7000, 0.00105263157895, True),
    (-7000, 0.004999, True), (-7000, 0.005001, False),
    (-7001, 0.0, False), (-6999, 0.0, False),
])
def test_period_verifier_money_is_exact_and_percentage_rounding_is_bounded(context, delta_cents, percentage_offset, expected_pass):
    context.dispatch('bind_metric', {'metric_id': 'CASH_NET_v1'})
    row = {'base_month': '2026-08', 'current_month': '2026-09',
           'base_net_cents': 38000, 'current_net_cents': 31000,
           'base_paid_cents': 40000, 'current_paid_cents': 40000,
           'base_refund_cents': 2000, 'current_refund_cents': 9000,
           'paid_delta_cents': 0, 'refund_delta_cents': 7000,
           'delta_cents': delta_cents, 'change_pct': -7000 / 38000 * 100 + percentage_offset}
    folder = context.run_dir / 'artifacts'
    folder.mkdir(exist_ok=True)
    path = folder / 'period_review.parquet'
    table = pa.Table.from_pylist([row])
    pq.write_table(table, path)
    artifact = {'artifact_id': 'artifact_period_review', 'step_id': 'period_review',
                'output_name': 'period_review', 'kind': 'table', 'path': str(path),
                'sha256': file_hash(path), 'row_count': 1,
                'schema': [{'name': field.name, 'type': str(field.type)} for field in table.schema]}
    result = context.reconcile({'period_review': artifact})
    assert result['passed'] is expected_pass
    if expected_pass:
        assert 'change_pct' in result['verified_columns']['period_review']
    else:
        assert result['checks'][0]['details']['counterexamples']
