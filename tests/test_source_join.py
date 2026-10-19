"""Bounded JOIN/aggregation over ONE registered SQLite source snapshot.

The source-side refund aggregate needs orders.channel_code; the same physical file
registers both tables, so the guard validates the query against that full registered
schema while the input budget still counts distinct referenced tables.

Business vocabulary (matches the approved CASH_NET_v1 contract, ``fixtures.bootstrap``,
``fixtures.compute_gold``, ``tools.reconcile`` and ``validation.validate_sources``):
orders are ``paid`` / ``failed``, refunds are ``succeeded`` / ``failed``. A fixture row
carrying orders status ``success`` would be excluded by every real consumer of the
venue, so the fixture writes the real business statuses and the SQL filters match them.
"""
import json
import sqlite3
from pathlib import Path

import pytest

from local_data_agent.connectors import snapshot_sources, sha256
from local_data_agent.executor import Executor

CHANNEL_AMOUNT = ("SELECT o.channel_code AS channel_code, SUM(r.refund_cents) AS refunded_cents "
                  "FROM refunds r JOIN orders o ON o.order_id=r.order_id "
                  "WHERE r.status='succeeded' GROUP BY o.channel_code ORDER BY o.channel_code")

def load(step_id, source, table, sql=None):
    step = {'step_id': step_id, 'type': 'LOAD', 'output_name': step_id, 'source_id': source, 'table': table}
    if sql:
        step['sql'] = sql
    return step

@pytest.fixture
def source(tmp_path):
    """One readonly SQLite snapshot: orders (11), refunds (4), marketing_clicks (2).

    ``orders`` is deliberately bigger than ``refunds`` so a summed distinct-table budget
    can be distinguished from a nominated-table-only budget. R2 carries refunds status
    ``failed``, so the ``succeeded`` predicate is provably doing work rather than matching
    every row.
    """
    base = tmp_path / 'var' / 'sources' / 'join-v1'
    base.mkdir(parents=True, exist_ok=False)
    database = base / 'business.db'
    orders = [(f'O{i}', '2026-09-01T00:00:00Z', 'web' if i % 2 else 'app', 1000 * i, 'paid') for i in range(1, 12)]
    refunds = [('R1', 'O1', '2026-09-02T00:00:00Z', 500, 'succeeded'),
               ('R2', 'O1', '2026-09-03T00:00:00Z', 100, 'failed'),
               ('R3', 'O2', '2026-09-04T00:00:00Z', 700, 'succeeded'),
               ('R4', 'O3', '2026-09-05T00:00:00Z', 200, 'succeeded')]
    with sqlite3.connect(database) as db:
        db.executescript('CREATE TABLE orders(order_id TEXT, paid_at TEXT, channel_code TEXT, paid_cents INTEGER, status TEXT); CREATE TABLE refunds(refund_id TEXT, order_id TEXT, refunded_at TEXT, refund_cents INTEGER, status TEXT); CREATE TABLE marketing_clicks(campaign TEXT, clicked_at TEXT, click_count INTEGER);')
        db.executemany('INSERT INTO orders VALUES (?,?,?,?,?)', orders)
        db.executemany('INSERT INTO refunds VALUES (?,?,?,?,?)', refunds)
        db.executemany('INSERT INTO marketing_clicks VALUES (?,?,?)', [('summer', '2026-09-01', 5), ('autumn', '2026-09-02', 7)])
    registry = {'fixture': 'join-v1', 'sources': [{'source_id': 'business', 'kind': 'sqlite', 'path': str(database)}],
                'documents': [], 'metric_contracts': []}
    (base / 'registry.json').write_text(json.dumps(registry), encoding='utf-8')
    run = tmp_path / 'var' / 'workspaces' / 'run1'
    run.mkdir(parents=True)
    manifest = snapshot_sources(tmp_path, run, fixture='join-v1')
    return run, manifest, next(source for source in manifest['sources'] if source['source_id'] == 'business')

def read_rows(artifact):
    import pyarrow.parquet as pq
    return pq.read_table(artifact['path']).to_pylist()

def test_refunds_join_orders_aggregates_channel_amount_and_original_channel(source):
    run, manifest, business = source
    artifact = Executor(run, manifest).execute(load('refunds_by_channel', 'business', 'refunds', CHANNEL_AMOUNT), {})
    rows = read_rows(artifact)
    # O1 is odd -> web (500), O2 even -> app (700), O3 odd -> web (200); R2's 100 is failed.
    assert {(row['channel_code'], row['refunded_cents']) for row in rows} == {('web', 700), ('app', 700)}
    assert artifact['row_count'] == 2
    # The failing refund is excluded by the source-side predicate, and every aggregated
    # row carries the channel of its ORIGINAL order, not a refund-side attribute.
    assert sum(row['refunded_cents'] for row in rows) == 1400
    assert artifact['input_hashes'] == {'business': business['sha256']}
    assert artifact['sha256'] == sha256(Path(artifact['path']))

def test_source_only_legacy_load_still_works(source):
    run, manifest, _ = source
    executor = Executor(run, manifest)
    plain = executor.execute(load('orders', 'business', 'orders'), {})
    # A bare LOAD is normalized by the guard (`qualify(..., identify=True)`): the star is
    # expanded to the registered columns and every name is quoted, so pin that shape
    # instead of a literal source-string comparison.
    assert plain['row_count'] == 11
    assert plain['sql'].startswith('SELECT ')
    assert 'FROM "orders"' in plain['sql']
    assert '*' not in plain['sql']
    assert [field['name'] for field in plain['schema']] == ['order_id', 'paid_at', 'channel_code', 'paid_cents', 'status']
    preview = {row['order_id']: row for row in plain['preview']}
    assert preview['O1'] == {'order_id': 'O1', 'paid_at': '2026-09-01T00:00:00Z', 'channel_code': 'web', 'paid_cents': 1000, 'status': 'paid'}
    assert preview['O3']['paid_cents'] == 3000
    # The expanding LOAD also comes back registered as a real source read, not a constant.
    assert plain['lineage']['source_id'] == 'business'
    filtered = executor.execute(load('refunds', 'business', 'refunds', "SELECT refund_id FROM refunds WHERE status='failed'"), {})
    assert filtered['row_count'] == 1
    # The guard legitimately qualifies/quotes the filtered query, so pin the rows and the
    # projected schema (R2 is the only `failed` refund) rather than any literal SQL string.
    assert filtered['preview'] == [{'refund_id': 'R2'}]
    assert [field['name'] for field in filtered['schema']] == ['refund_id']

def test_constant_query_without_referenced_table_is_not_a_source_load(source):
    run, manifest, _ = source
    with pytest.raises(ValueError, match='not referenced'):
        Executor(run, manifest).execute(load('orders', 'business', 'orders', 'SELECT 1 AS one'), {})

# The guard rejects each of these for a concrete, already-verified reason; the literals
# below are the messages the executor actually raises, not invented categories.
@pytest.mark.parametrize('sql,message', [
    # Names that are neither registered nor CTE aliases fail the guard's table check.
    ("SELECT * FROM other_source_orders", 'Unknown table'),
    ("SELECT SUM(paid_cents) FROM orders JOIN unexpected ON 1=1", 'Unknown table'),
    ("SELECT system('ls') FROM orders", 'Function is not allowed'),
    # A table function is refused by the guard's table-function/qualified-table check, not by
    # the scalar function allow-list, so this case pins the `forbidden` category it raises.
    ("SELECT * FROM readfile('secret')", 'forbidden'),
    ("SELECT * FROM orders; DROP TABLE orders", 'Only one SELECT'),
    ("DELETE FROM refunds", 'Only one SELECT'),
    ("PRAGMA table_info(orders)", 'Only one SELECT'),
    ("ATTACH DATABASE 'other.db' AS other", 'Only one SELECT'),
    ("SELECT SUM(missing_cents) FROM refunds JOIN orders ON orders.order_id=refunds.order_id", 'validation failed'),
    ("WITH x AS (SELECT * FROM sqlite_master) SELECT * FROM x", 'Unknown table'),
])
def test_unregistered_tables_files_writes_and_unsupported_expressions_refused(source, sql, message):
    run, manifest, _ = source
    with pytest.raises(ValueError, match=message):
        Executor(run, manifest).execute(load('orders', 'business', 'orders', sql), {})

@pytest.mark.parametrize('sql', [
    "SELECT load_extension('x') FROM orders",
    "SELECT read_text('secret') AS payload FROM orders",
])
def test_extension_and_file_read_functions_refused(source, sql):
    run, manifest, _ = source
    # `load_extension` is refused up front by the statement check; `read_text` reaches the
    # function allow-list. Either way the query is refused and never touches the file, so
    # only the refusal itself is pinned, not a message category.
    with pytest.raises(ValueError):
        Executor(run, manifest).execute(load('orders', 'business', 'orders', sql), {})

def test_unknown_nominated_or_per_source_table_refused(source):
    run, manifest, _ = source
    executor = Executor(run, manifest)
    with pytest.raises(ValueError, match='Unknown source table'):
        executor.execute(load('ghost', 'business', 'ghost'), {})
    # `marketing_clicks` is registered here but never selectable through the other source.
    with pytest.raises(ValueError, match='Unknown source_id'):
        executor.execute(load('other', 'business_archive', 'marketing_clicks'), {})

def test_summed_input_row_budget_uses_distinct_referenced_tables(source):
    run, manifest, _ = source
    # orders alone has 11 rows and refunds alone 4, yet the JOIN reads 15 physical rows.
    with pytest.raises(ValueError, match='Input row limit exceeded'):
        Executor(run, manifest, max_input_rows=14).execute(load('refunds_by_channel', 'business', 'refunds', CHANNEL_AMOUNT), {})
    # Nominated table `refunds` is small; the small-nomination count must not hide orders.
    with pytest.raises(ValueError, match='Input row limit exceeded'):
        Executor(run, manifest, max_input_rows=4).execute(load('refunds_by_channel', 'business', 'refunds', CHANNEL_AMOUNT), {})
    assert Executor(run, manifest, max_input_rows=15).execute(
        load('refunds_by_channel', 'business', 'refunds', CHANNEL_AMOUNT), {})['row_count'] == 2

def test_self_join_counts_the_physical_table_once(source):
    run, manifest, _ = source
    pairs = ("SELECT COUNT(*) AS pair_count FROM orders a JOIN orders b ON a.order_id=b.order_id")
    assert Executor(run, manifest, max_input_rows=11).execute(load('orders', 'business', 'orders', pairs), {})['preview'] == [{'pair_count': 11}]
    with pytest.raises(ValueError, match='Input row limit exceeded'):
        Executor(run, manifest, max_input_rows=10).execute(load('orders', 'business', 'orders', pairs), {})

def test_cte_query_resolves_physical_inputs_and_budget(source):
    run, manifest, _ = source
    cte = ("WITH paid AS (SELECT order_id, channel_code FROM orders WHERE status='paid'), "
           "successful AS (SELECT order_id, refund_cents FROM refunds WHERE status='succeeded') "
           "SELECT p.channel_code AS channel_code, SUM(s.refund_cents) AS refunded_cents "
           "FROM successful s JOIN paid p ON p.order_id=s.order_id GROUP BY p.channel_code ORDER BY p.channel_code")
    assert read_rows(Executor(run, manifest).execute(load('refunds', 'business', 'refunds', cte), {})) == [
        {'channel_code': 'app', 'refunded_cents': 700}, {'channel_code': 'web', 'refunded_cents': 700}]
    # A CTE alias is not a physical table: orders(11) + refunds(4) are still both counted.
    with pytest.raises(ValueError, match='Input row limit exceeded'):
        Executor(run, manifest, max_input_rows=14).execute(load('refunds', 'business', 'refunds', cte), {})

def test_raw_source_output_cap_still_enforced_by_executor(source):
    run, manifest, _ = source
    with pytest.raises(ValueError, match='Output row limit'):
        Executor(run, manifest, max_output_rows=5).execute(load('orders', 'business', 'orders'), {})
    # Aggregating the same registered rows down to a small result is allowed.
    artifact = Executor(run, manifest, max_output_rows=2).execute(load('refunds', 'business', 'refunds', CHANNEL_AMOUNT), {})
    assert artifact['row_count'] == 2

# A valid scope shadow: the outer query reads a physical `orders o` and joins a derived
# table whose own CTE is *also* named `orders`. `WITH orders AS (SELECT ... FROM orders)`
# would be circular in SQLite, so the shadow lives inside the derived subquery instead.
SCOPE_SHADOW = ("SELECT COUNT(*) AS pair_count FROM orders o "
                "JOIN (WITH orders AS (SELECT refund_id FROM refunds) SELECT refund_id FROM orders) q ON 1=1")
# Independent case: the ONLY read of the shadowed name is the CTE itself, so nominating
# the physical table `orders` is not a load of it.
SCOPE_SHADOW_ONLY = "WITH orders AS (SELECT refund_id FROM refunds) SELECT refund_id FROM orders"

def test_scope_shadow_still_counts_both_physical_tables(source):
    run, manifest, _ = source
    executor = Executor(run, manifest)
    # The CTE named `orders` must not hide the outer physical `orders`: orders(11) + refunds(4)
    # = 15 rows are charged, so the shadow is scoped rather than subtracted globally.
    artifact = executor.execute(load('orders', 'business', 'orders', SCOPE_SHADOW), {})
    assert artifact['preview'] == [{'pair_count': 44}]  # 11 outer orders x 4 refund rows
    assert Executor(run, manifest, max_input_rows=15).execute(load('orders', 'business', 'orders', SCOPE_SHADOW), {})['row_count'] == 1
    with pytest.raises(ValueError, match='Input row limit exceeded'):
        Executor(run, manifest, max_input_rows=14).execute(load('orders', 'business', 'orders', SCOPE_SHADOW), {})
    # The same 15 physical rows are charged regardless of which physical table is nominated.
    assert Executor(run, manifest, max_input_rows=15).execute(load('refunds', 'business', 'refunds', SCOPE_SHADOW), {})['row_count'] == 1
    with pytest.raises(ValueError, match='Input row limit exceeded'):
        Executor(run, manifest, max_input_rows=14).execute(load('refunds', 'business', 'refunds', SCOPE_SHADOW), {})

def test_cte_shadowing_registered_name_is_not_a_physical_orders_read(source):
    run, manifest, _ = source
    executor = Executor(run, manifest)
    # This query reads only physical `refunds`; the CTE named `orders` shadows it, so this is
    # not a load of table orders: it must be refused as not referenced instead of being
    # permitted (and budgeted) as if physical orders had been read.
    with pytest.raises(ValueError, match='not referenced'):
        executor.execute(load('orders', 'business', 'orders', SCOPE_SHADOW_ONLY), {})
    # Nominated as `refunds` the same query is a legitimate read of one physical table.
    artifact = executor.execute(load('refunds', 'business', 'refunds', SCOPE_SHADOW_ONLY), {})
    assert {row['refund_id'] for row in read_rows(artifact)} == {'R1', 'R2', 'R3', 'R4'}
    # Budget charges exactly the physical refunds(4) rows: the shadowing CTE name adds nothing.
    assert Executor(run, manifest, max_input_rows=4).execute(load('refunds', 'business', 'refunds', SCOPE_SHADOW_ONLY), {})['row_count'] == 4
    with pytest.raises(ValueError, match='Input row limit exceeded'):
        Executor(run, manifest, max_input_rows=3).execute(load('refunds', 'business', 'refunds', SCOPE_SHADOW_ONLY), {})

def test_cte_aggregation_reads_refunds_and_orders_with_summed_distinct_budget(source):
    run, manifest, _ = source
    # Aggregation through two CTEs over BOTH physical tables still reads orders(11) + refunds(4),
    # so the summed distinct physical-table budget of 15 applies to the aggregate result too.
    cte = ("WITH paid AS (SELECT order_id, channel_code FROM orders WHERE status='paid'), "
           "successful AS (SELECT order_id, refund_cents FROM refunds WHERE status='succeeded') "
           "SELECT p.channel_code AS channel_code, SUM(s.refund_cents) AS refunded_cents "
           "FROM successful s JOIN paid p ON p.order_id=s.order_id GROUP BY p.channel_code ORDER BY p.channel_code")
    assert read_rows(Executor(run, manifest, max_input_rows=15).execute(load('refunds', 'business', 'refunds', cte), {})) == [
        {'channel_code': 'app', 'refunded_cents': 700}, {'channel_code': 'web', 'refunded_cents': 700}]
    # One row below the summed physical total is refused even though the aggregate is tiny.
    with pytest.raises(ValueError, match='Input row limit exceeded'):
        Executor(run, manifest, max_input_rows=14).execute(load('refunds', 'business', 'refunds', cte), {})
    # Nominating either physical table is valid, and both charge the same 15 physical rows.
    assert Executor(run, manifest, max_input_rows=15).execute(load('orders', 'business', 'orders', cte), {})['row_count'] == 2

def test_fixture_uses_real_business_statuses_and_filters(source):
    run, manifest, _ = source
    executor = Executor(run, manifest)
    # The fixture must not invent a `success` status: the venue's business vocabulary is
    # orders paid/failed and refunds succeeded/failed, and only the successful side counts.
    ordered = executor.execute(load('orders', 'business', 'orders', 'SELECT status, COUNT(*) AS n FROM orders GROUP BY status'), {})
    assert {row['status'] for row in read_rows(ordered)} == {'paid'}
    refunded = executor.execute(load('refunds', 'business', 'refunds', 'SELECT status, COUNT(*) AS n FROM refunds GROUP BY status'), {})
    assert {(row['status'], row['n']) for row in read_rows(refunded)} == {('succeeded', 3), ('failed', 1)}
    # Re-labelling the three successful refunds must empty (and therefore fail) the aggregate,
    # which proves the `succeeded` predicate is real rather than satisfied by every row.
    with sqlite3.connect(Path(manifest['sources'][0]['path'])) as db:
        assert db.execute("SELECT COUNT(*) FROM refunds WHERE status NOT IN ('succeeded','failed')").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM orders WHERE status <> 'paid'").fetchone()[0] == 0
