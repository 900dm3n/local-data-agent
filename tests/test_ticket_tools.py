"""Integration tests for search_tickets and inspect_source_checks."""

import json

import pytest

from local_data_agent.fixtures import bootstrap
from local_data_agent.connectors import snapshot_sources
from local_data_agent.state import StateStore
from local_data_agent.artifacts import checked_run_dir
from local_data_agent.tools import ToolContext, validate_arguments


@pytest.fixture()
def ctx(tmp_path):
    bootstrap(tmp_path)
    store = StateStore(tmp_path)
    state = store.create('cash net', 'live', 'cash-net-v1', 'cash-net-v1')
    run = checked_run_dir(tmp_path, state['run_id'])
    manifest = snapshot_sources(tmp_path, run)
    state['manifest'] = manifest
    context = ToolContext(tmp_path, run, state, manifest, store, None)
    yield context
    context.retriever.close()


def _source_jsonl(tmp_path, manifest, source_id):
    source = [item for item in manifest['sources'] if item['source_id'] == source_id][0]
    path = source['path'] if isinstance(source['path'], str) and source['path'].startswith(str(tmp_path)) else str(tmp_path / source['path'])
    with open(path, 'r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_search_tickets_persists_result(ctx):
    result = ctx.dispatch('search_tickets', {'query': '', 'source_id': 'tickets', 'month': '', 'channel_code': '', 'limit': 2})
    search_id = result['search_id']
    assert search_id.startswith('ticket_search_')
    stored = ctx.state['ticket_searches'][search_id]
    assert result['returned_ticket_count'] == len(result['evidence'])
    assert result['matched_ticket_count'] >= result['returned_ticket_count']
    assert result['population_ticket_count'] >= result['matched_ticket_count']
    assert result['source_ticket_count'] >= result['population_ticket_count']
    rows = _source_jsonl(ctx.root, ctx.manifest, 'tickets')
    assert result['source_ticket_count'] == len(rows)
    assert result['population_ticket_count'] == len(rows)
    assert result['returned_ticket_count'] == 2
    for quote in result['evidence']:
        assert quote['source_hash'] and quote['record_hash']
    persisted = json.loads((ctx.run_dir / 'ticket_searches' / f'{search_id}.json').read_text(encoding='utf-8'))
    assert persisted == stored


def test_search_tickets_validation(ctx):
    with pytest.raises(ValueError):
        ctx.dispatch('search_tickets', {'query': '', 'source_id': 'tickets', 'month': '2024-13', 'channel_code': '', 'limit': 1})
    with pytest.raises(ValueError):
        ctx.dispatch('search_tickets', {'query': '', 'source_id': 'tickets', 'month': '', 'channel_code': '', 'limit': True})
    with pytest.raises(ValueError):
        ctx.dispatch('search_tickets', {'query': '', 'source_id': 'tickets', 'month': '', 'channel_code': '', 'limit': 9})
    with pytest.raises(ValueError):
        ctx.dispatch('search_tickets', {'query': '', 'source_id': 'missing-source', 'month': '', 'channel_code': '', 'limit': 1})
    ctx.state.setdefault('ticket_searches', {})
    for index in range(8):
        ctx.state['ticket_searches'][f'seed_{index}'] = {}
    with pytest.raises(ValueError):
        ctx.dispatch('search_tickets', {'query': '', 'source_id': 'tickets', 'month': '', 'channel_code': '', 'limit': 1})


def test_inspect_source_checks_detects_change(ctx):
    first = ctx.dispatch('inspect_source_checks', {})
    assert first['passed'] is True
    assert first['checks']
    assert not first.get('failed_details')
    source = ctx.manifest['sources'][0]
    with open(source['path'], 'ab') as handle:
        handle.write(b'\n')
    second = ctx.dispatch('inspect_source_checks', {})
    assert second['passed'] is False
    assert second['failed_details']
