"""Provider-view compaction must preserve protocol and persisted evidence."""
import copy
import json
import pytest
from local_data_agent.context_budget import prepare_history, check_history, estimate_tokens

def responses_exchange(index, padding=1600):
    return [
        {'type': 'reasoning', 'id': f'reason_{index}', 'summary': [{'type': 'summary_text', 'text': f'exchange {index}'}]},
        {'type': 'function_call', 'call_id': f'call_{index}', 'name': 'search_context', 'arguments': '{}'},
        {'type': 'function_call_output', 'call_id': f'call_{index}', 'output': json.dumps({'note': 'q' * padding})}]

def assert_pairs(history):
    requests, outputs = [], []
    for item in history:
        if item.get('type') == 'function_call':
            requests.append(item['call_id'])
        if item.get('type') == 'chat_message':
            requests.extend(call['id'] for call in item['message'].get('tool_calls', []))
        if item.get('type') == 'function_call_output':
            outputs.append(item['call_id'])
    assert sorted(requests) == sorted(outputs)
    assert len(requests) == len(set(requests))
    assert len(outputs) == len(set(outputs))

def test_compact_result_view_preserves_arguments_and_original_history():
    history = [
        {'role': 'system', 'content': 'agent'},
        {'type': 'function_call', 'call_id': 'one', 'name': 'execute_step', 'arguments': '{"plan_version":1,"step_id":"orders"}'},
        {'type': 'function_call_output', 'call_id': 'one', 'output': json.dumps({'artifact_id': 'a1', 'schema': [{'name': 'net_cents'}], 'preview': [{'net_cents': 31000}], 'sql': 'SELECT SUM(x)', 'explain': ['large'], 'path': 'local-only', 'lineage': {'sources': ['s1']}})}]
    original = copy.deepcopy(history)
    compacted = prepare_history(history)
    assert history == original
    assert compacted[1] == history[1]
    payload = json.loads(compacted[2]['output'])
    assert payload['preview'] == [{'net_cents': 31000}]
    assert payload['schema'] == [{'name': 'net_cents'}]
    assert not {'path', 'lineage', 'explain', 'sql'}.intersection(payload)
    assert_pairs(compacted)

def test_responses_compaction_keeps_reasoning_with_its_retained_tool_exchange():
    history = [{'role': 'system', 'content': 'agent'}, {'role': 'user', 'content': 'question'}]
    for index in range(5):
        history.extend(responses_exchange(index))
    original = copy.deepcopy(history)
    prepared = prepare_history(history, maximum=1500)
    assert estimate_tokens(prepared) <= 1500
    assert history == original
    assert_pairs(prepared)
    retained_calls = [item for item in prepared if item.get('type') == 'function_call']
    assert retained_calls
    for call in retained_calls:
        index = call['call_id'].split('_')[1]
        assert any(item.get('type') == 'reasoning' and item.get('id') == f'reason_{index}' for item in prepared)

def test_chat_multi_tool_exchange_is_never_split():
    history = [{'role': 'system', 'content': 'agent'}, {'role': 'user', 'content': 'question'}]
    for index in range(5):
        calls = [{'id': f'{index}_{suffix}', 'type': 'function', 'function': {'name': 'inspect_schema', 'arguments': '{}'}} for suffix in ('a', 'b')]
        history.append({'type': 'chat_message', 'message': {'role': 'assistant', 'content': None, 'tool_calls': calls}})
        history.extend({'type': 'function_call_output', 'call_id': call['id'], 'output': json.dumps({'note': 'r' * 800})} for call in calls)
    original = copy.deepcopy(history)
    prepared = prepare_history(history, maximum=1800)
    assert estimate_tokens(prepared) <= 1800
    assert history == original
    assert_pairs(prepared)
    last = next(item for item in reversed(prepared) if item.get('type') == 'chat_message')
    assert {call['id'] for call in last['message']['tool_calls']} == {'4_a', '4_b'}

@pytest.mark.parametrize('protocol', ['responses', 'chat'])
def test_pending_calls_are_retained_or_budget_stop_never_silently_dropped(protocol):
    history = [{'role': 'system', 'content': 'agent'}]
    history.extend(responses_exchange(0, padding=2000))
    if protocol == 'responses':
        pending = {'type': 'function_call', 'call_id': 'pending', 'name': 'inspect_schema', 'arguments': '{}'}
    else:
        pending = {'type': 'chat_message', 'message': {'role': 'assistant', 'tool_calls': [
            {'id': 'pending', 'type': 'function', 'function': {'name': 'inspect_schema', 'arguments': '{}'}}]}}
    history.append(pending)
    prepared = prepare_history(history, maximum=200)
    assert pending in prepared
    with pytest.raises(RuntimeError, match='context_budget_exceeded'):
        check_history(prepared, maximum=200)

def test_persisted_current_plan_and_evidence_references_survive_archival():
    history = [{'role': 'system', 'content': 'agent'}, {'role': 'user', 'content': 'old question'}]
    for index in range(5):
        history.extend(responses_exchange(index))
    contract = {'metric_id': 'CASH_NET_v1', 'version': 1, 'timezone': 'Australia/Sydney'}
    plan = [{'step_id': 'net', 'type': 'SELECT', 'output_name': 'net', 'depends_on': ['events'], 'sql': 'SELECT month,SUM(signed_cents) AS net_cents FROM events GROUP BY month'}]
    reference = {'evidence_id': 'doc:approved-metric', 'doc_hash': 'specific-hash', 'start_line': 3, 'end_line': 9, 'text': 'approved event month definition'}
    state = {'question': '只看官网', 'bound_contract': contract, 'plan_version': 3, 'plan': plan,
             'observed_facts': [{'artifact_id': 'artifact_net', 'description': 'result inspected'}],
             'open_questions': ['需要官网明细'], 'artifacts': {},
             'evidence': [{'query': '指标口径', 'evidence': [reference]}]}
    original_state = copy.deepcopy(state)
    prepared = prepare_history(history, maximum=2100, state=state)
    summaries = [item['content'].split('：', 1)[1] for item in prepared
                 if item.get('role') == 'user' and item.get('content', '').startswith('持久化状态摘要：')]
    assert summaries
    summary = json.loads(summaries[0])
    assert summary['question'] == state['question']
    assert summary['plan'] == plan
    assert summary['plan_version'] == 3
    assert summary['metric_contract'] == contract
    assert summary['open_questions'] == state['open_questions']
    encoded = json.dumps(summary, ensure_ascii=False)
    assert 'doc:approved-metric' in encoded and 'specific-hash' in encoded
    assert 'start_line' in encoded and 'end_line' in encoded
    assert state == original_state
    assert_pairs(prepared)
