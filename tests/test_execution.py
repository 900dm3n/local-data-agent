import csv
import json
import sqlite3
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from local_data_agent.connectors import snapshot_sources, sha256
from local_data_agent.executor import Executor, guard_sql
from local_data_agent.plan import validate_plan, affected_steps
from local_data_agent.lineage import build_lineage

@pytest.fixture
def workspace(tmp_path):
    base = tmp_path / 'var' / 'sources' / 'cash-net-v1'
    base.mkdir(parents=True)
    database = base / 'business.db'
    with sqlite3.connect(database) as db:
        db.executescript('CREATE TABLE orders(order_id TEXT, paid_at TEXT, channel_code TEXT, paid_cents INTEGER, status TEXT); CREATE TABLE refunds(refund_id TEXT, order_id TEXT, refunded_at TEXT, refund_cents INTEGER, status TEXT);')
        db.executemany('INSERT INTO orders VALUES (?,?,?,?,?)', [
            ('O1', '2026-08-05T00:00:00Z', 'web', 10000, 'success'),
            ('O2', '2026-08-20T00:00:00Z', 'app', 20000, 'success'),
            ('O3', '2026-08-28T00:00:00Z', 'web', 10000, 'success'),
            ('O4', '2026-09-05T00:00:00Z', 'web', 12000, 'success'),
            ('O5', '2026-09-15T00:00:00Z', 'app', 18000, 'success'),
            ('O6', '2026-09-28T00:00:00Z', 'web', 10000, 'success')])
        db.executemany('INSERT INTO refunds VALUES (?,?,?,?,?)', [
            ('R1', 'O2', '2026-08-25T00:00:00Z', 2000, 'success'),
            ('R2', 'O4', '2026-09-10T00:00:00Z', 3000, 'success'),
            ('R3', 'O4', '2026-09-12T00:00:00Z', 1000, 'success'),
            ('R4', 'O3', '2026-09-06T00:00:00Z', 5000, 'success')])
    channels = base / 'channels.csv'
    channels.write_text('channel_code,channel_name\nweb,官网\napp,移动端\n', encoding='utf-8')
    tickets = base / 'tickets.jsonl'
    tickets.write_text(json.dumps({'ticket_id': 'T1', 'text': '等待退款'}, ensure_ascii=False) + '\n', encoding='utf-8')
    document = base / 'metric.md'
    document.write_text('CASH_NET_v1: success events, Australia/Sydney', encoding='utf-8')
    registry = {'fixture': 'cash-net-v1', 'sources': [
        {'source_id': 'business', 'kind': 'sqlite', 'path': str(database)},
        {'source_id': 'channels', 'kind': 'csv', 'path': str(channels)},
        {'source_id': 'tickets', 'kind': 'jsonl', 'path': str(tickets)}],
        'documents': [{'document_id': 'metric', 'path': str(document)}], 'metric_contracts': []}
    (base / 'registry.json').write_text(json.dumps(registry), encoding='utf-8')
    run = tmp_path / 'var' / 'workspaces' / 'run1'
    run.mkdir(parents=True)
    manifest = snapshot_sources(tmp_path, run)
    return tmp_path, run, manifest

def load(step_id, source, table, sql=None):
    result = {'step_id': step_id, 'type': 'LOAD', 'output_name': step_id,
              'source_id': source, 'table': table}
    if sql:
        result['sql'] = sql
    return result

def test_plan_cycles_unknown_fields_and_transitive_invalidation():
    first = load('orders', 'business', 'orders')
    second = {'step_id': 'total', 'type': 'SELECT', 'output_name': 'total', 'depends_on': ['orders'], 'sql': 'SELECT SUM(paid_cents) AS cents FROM orders'}
    third = {'step_id': 'save', 'type': 'SAVE', 'output_name': 'saved', 'depends_on': ['total']}
    assert len(validate_plan([first, second, third], ['business'])) == 3
    with pytest.raises(ValueError):
        validate_plan([{**first, 'shell': 'bad'}], ['business'])
    with pytest.raises(ValueError):
        validate_plan([{**second, 'depends_on': ['total']}], ['business'])
    changed = {**first, 'sql': "SELECT * FROM orders WHERE status='success'"}
    assert affected_steps([first, second, third], [changed, second, third]) == {'orders', 'total', 'save'}

@pytest.mark.parametrize('sql', [
    "SELECT * FROM read_csv_auto('secret.csv')", "SELECT * FROM read_parquet('secret.parquet')",
    "SELECT SUM(CAST(read_text('secret') AS INTEGER)) FROM orders",
    'SELECT * FROM orders; DROP TABLE orders', 'DELETE FROM orders',
    "SELECT load_extension('x') FROM orders", 'SELECT missing FROM orders',
    'SELECT * FROM unexpected', "WITH x AS (SELECT * FROM read_json('x')) SELECT * FROM x",
    "SELECT query('SELECT 1')", "COPY orders TO 'x.csv'", "SELECT * FROM main.orders",
    'SELECT o.missing FROM orders AS o',
    'WITH x AS (SELECT missing FROM orders) SELECT * FROM x',
    'SELECT orders.missing FROM orders',
])
def test_sql_guard_rejects_external_mutation_and_unknown_fields(sql):
    with pytest.raises(Exception):
        guard_sql(sql, {'orders': [{'name': 'paid_cents', 'type': 'int64'}]})

def test_snapshot_is_immutable_and_hashes_are_checked(workspace):
    project, run, manifest = workspace
    original = project / 'var' / 'sources' / 'cash-net-v1' / 'business.db'
    with sqlite3.connect(original) as db:
        db.execute("UPDATE orders SET paid_cents=99999 WHERE order_id='O1'")
    artifact = Executor(run, manifest).execute(load('orders', 'business', 'orders'), {})
    assert artifact['preview'][0]['paid_cents'] == 10000
    Path(manifest['sources'][0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        Executor(run, manifest).execute(load('other', 'business', 'orders'), {})

def test_cash_event_ledger_and_multi_source_join(workspace):
    _, run, manifest = workspace
    executor = Executor(run, manifest)
    steps = [load('orders', 'business', 'orders', "SELECT * FROM orders WHERE status='success'"),
             load('refunds', 'business', 'refunds', "SELECT * FROM refunds WHERE status='success'"),
             load('channels', 'channels', 'channels')]
    artifacts = {}
    for step in steps:
        artifacts[step['output_name']] = executor.execute(step, artifacts)
    step = {'step_id': 'monthly', 'type': 'SELECT', 'output_name': 'monthly',
            'depends_on': ['orders', 'refunds', 'channels'], 'sql': """
        WITH events AS (
          SELECT strftime(timezone('Australia/Sydney', CAST(paid_at AS TIMESTAMPTZ)), '%Y-%m') AS month,
                 channel_code, paid_cents AS cents FROM orders
          UNION ALL
          SELECT strftime(timezone('Australia/Sydney', CAST(r.refunded_at AS TIMESTAMPTZ)), '%Y-%m'),
                 o.channel_code, -r.refund_cents FROM refunds r JOIN orders o ON r.order_id=o.order_id
        ) SELECT e.month, c.channel_name, SUM(e.cents) AS net_cents
          FROM events e JOIN channels c ON e.channel_code=c.channel_code
          GROUP BY e.month,c.channel_name ORDER BY e.month,c.channel_name
        """}
    artifact = executor.execute(step, artifacts)
    artifacts['monthly'] = artifact
    values = {(row['month'], row['channel_name']): row['net_cents'] for row in artifact['preview']}
    assert values == {('2026-08', '官网'): 20000, ('2026-08', '移动端'): 18000,
                      ('2026-09', '官网'): 13000, ('2026-09', '移动端'): 18000}
    assert artifact['sha256'] == sha256(Path(artifact['path']))
    assert len(build_lineage([*steps, step], artifacts, manifest)['columns']) > 0

def test_timezone_boundary_and_unknown_dependency(workspace):
    _, run, manifest = workspace
    executor = Executor(run, manifest)
    artifact = executor.execute(load('orders', 'business', 'orders'), {})
    step = {'step_id': 'boundary', 'type': 'SELECT', 'output_name': 'boundary', 'depends_on': ['orders'],
            'sql': "SELECT strftime(timezone('Australia/Sydney', CAST('2026-08-31T14:30:00Z' AS TIMESTAMPTZ)), '%Y-%m') AS local_month, strftime(timezone('Australia/Sydney', CAST('2026-10-31T13:30:00Z' AS TIMESTAMPTZ)), '%Y-%m') AS dst_month FROM orders LIMIT 1"}
    result = executor.execute(step, {'orders': artifact})
    assert result['preview'] == [{'local_month': '2026-09', 'dst_month': '2026-11'}]
    with pytest.raises(ValueError, match='dependency'):
        executor.execute({**step, 'depends_on': ['missing']}, {'orders': artifact})

def test_limits_and_jsonl(workspace):
    _, run, manifest = workspace
    with pytest.raises(ValueError, match='Output row limit'):
        Executor(run, manifest, max_output_rows=2).execute(load('orders', 'business', 'orders'), {})
    with pytest.raises(ValueError, match='Input row limit'):
        Executor(run, manifest, max_input_rows=2).execute(load('orders', 'business', 'orders'), {})
    result = Executor(run, manifest).execute(load('tickets', 'tickets', 'tickets'), {})
    assert result['preview'] == [{'ticket_id': 'T1', 'text': '等待退款'}]
    with pytest.raises(RuntimeError, match='timed out'):
        Executor(run, manifest, timeout=0.001).execute(load('orders', 'business', 'orders'), {})

def test_registry_rejects_path_escape(tmp_path):
    folder = tmp_path / 'var' / 'sources' / 'cash-net-v1'
    folder.mkdir(parents=True)
    outside = tmp_path / 'private.csv'
    outside.write_text('x\n1\n')
    (folder / 'registry.json').write_text(json.dumps({'sources': [{'source_id': 'outside', 'kind': 'csv', 'path': str(outside)}]}))
    run = tmp_path / 'run'
    run.mkdir()
    with pytest.raises(ValueError, match='outside'):
        snapshot_sources(tmp_path, run)

def test_parquet_source_schema_empty_results_and_cte(workspace):
    project, run, manifest = workspace
    source = run / 'inputs' / 'reference.parquet'
    pq.write_table(pa.table({'code': ['web', 'app'], 'weight': [2, 3]}), source)
    manifest['sources'].append({'source_id': 'reference', 'kind': 'parquet', 'path': str(source),
                               'sha256': sha256(source), 'schema': {'reference': [{'name': 'code', 'type': 'string'}, {'name': 'weight', 'type': 'int64'}]}})
    executor = Executor(run, manifest)
    artifact = executor.execute(load('reference', 'reference', 'reference'), {})
    assert artifact['preview'][1]['weight'] == 3
    step = {'step_id': 'filtered', 'type': 'SELECT', 'output_name': 'filtered', 'depends_on': ['reference'],
            'sql': "WITH selected AS (SELECT code,weight FROM reference WHERE weight<0) SELECT substr(code,1,3) AS prefix,SUM(weight) AS total FROM selected GROUP BY code"}
    result = executor.execute(step, {'reference': artifact})
    assert result['row_count'] == 0
    assert [f['name'] for f in result['schema']] == ['prefix', 'total']

def test_sum_currency_is_int64_without_model_cast_and_fraction_is_rejected(workspace):
    _, run, manifest = workspace
    executor = Executor(run, manifest)
    orders = executor.execute(load('orders', 'business', 'orders'), {})
    step = {'step_id': 'total', 'type': 'SELECT', 'output_name': 'total', 'depends_on': ['orders'],
            'sql': 'SELECT SUM(paid_cents) AS total_cents FROM orders'}
    result = executor.execute(step, {'orders': orders})
    full = pq.read_table(result['path'])
    assert full.schema.field('total_cents').type == pa.int64()
    assert full.to_pylist() == [{'total_cents': 80000}]
    assert isinstance(full.to_pylist()[0]['total_cents'], int)
    with pytest.raises(ValueError, match='exact integer cents'):
        executor.execute({**step, 'sql': 'SELECT CAST(1.25 AS DECIMAL(10,2)) AS total_cents FROM orders LIMIT 1'}, {'orders': orders})
    with pytest.raises(ValueError, match='outside int64'):
        executor.execute({**step, 'sql': 'SELECT CAST(9223372036854775808 AS DECIMAL(38,0)) AS total_cents FROM orders LIMIT 1'}, {'orders': orders})

def test_boolean_date_ranges_and_ordinary_predicates_execute(workspace):
    _, run, manifest = workspace
    executor = Executor(run, manifest)
    orders = executor.execute(load('orders', 'business', 'orders',
                                   "SELECT * FROM orders WHERE paid_cents>=10000 AND paid_cents<20000"), {})
    assert orders['row_count'] == 5
    step = {'step_id': 'september', 'type': 'SELECT', 'output_name': 'september', 'depends_on': ['orders'],
            'sql': "SELECT SUM(paid_cents) AS total_cents FROM orders WHERE paid_at>='2026-09-01T00:00:00Z' AND paid_at<'2026-10-01T00:00:00Z' AND (channel_code IN ('web','app') OR channel_code IS NULL) AND NOT paid_cents BETWEEN 0 AND 9999"}
    result = executor.execute(step, {'orders': orders})
    assert result['preview'] == [{'total_cents': 40000}]
