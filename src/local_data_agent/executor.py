"""Bounded, read-only SQL in killable spawned processes.

This is a restricted SQL executor, not a sandbox for arbitrary Python code.
"""
import json
import multiprocessing as mp
import os
import queue
import uuid
from decimal import Decimal
from contextlib import closing
from pathlib import Path
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope
from .connectors import contained, file_table, readonly_sqlite, sha256
from .models import PlanStep

SAFE_FUNCTIONS = {
    'SUM', 'COUNT', 'AVG', 'MIN', 'MAX', 'COALESCE', 'NULLIF', 'ROUND', 'ABS',
    'CAST', 'TRY_CAST', 'SUBSTRING', 'STRFTIME', 'TIME_TO_STR', 'STR_TO_TIME',
    'DATE_TRUNC', 'TIMESTAMP_TRUNC', 'TIMEZONE', 'EXTRACT', 'YEAR', 'MONTH',
    'LOWER', 'UPPER', 'LENGTH', 'TRIM', 'LTRIM', 'RTRIM', 'REPLACE', 'CONCAT',
    'CONCAT_WS', 'IF', 'CASE', 'GREATEST', 'LEAST', 'FLOOR', 'CEIL', 'CEILING',
    'ROW_NUMBER', 'RANK', 'DENSE_RANK', 'LAG', 'LEAD', 'FIRST_VALUE', 'LAST_VALUE',
    'DATE', 'DATETIME', 'TIMESTAMP', 'TS_OR_DS_TO_DATE', 'TS_OR_DS_TO_TIMESTAMP',
    # SQLGlot models some ordinary predicates as Func subclasses.
    'AND', 'OR', 'NOT', 'IN', 'BETWEEN', 'IS',
}
SQLITE_FUNCTIONS = {'sum', 'count', 'avg', 'min', 'max', 'coalesce', 'nullif',
                    'round', 'abs', 'substr', 'substring', 'strftime', 'lower',
                    'upper', 'length', 'trim', 'ltrim', 'rtrim', 'replace',
                    'date', 'datetime', 'ifnull', 'iif', 'total'}

def _sql_type(value):
    value = value.lower()
    if any(word in value for word in ['int', 'long']):
        return 'BIGINT'
    if any(word in value for word in ['float', 'double', 'real', 'decimal']):
        return 'DOUBLE'
    if 'bool' in value:
        return 'BOOLEAN'
    if 'timestamp' in value:
        return 'TIMESTAMP'
    if value == 'date':
        return 'DATE'
    return 'VARCHAR'

def referenced_tables(sql: str, schemas: dict, dialect='duckdb') -> set[str]:
    """Physical tables of THIS source that the guard-approved query actually reads.

    Resolution is scope-aware, not name-based. A `exp.Table` node in the syntax tree is
    not proof of a physical read: `WITH orders AS (SELECT refund_id FROM refunds)
    SELECT refund_id FROM orders` mentions the registered name `orders`, but the FROM
    source resolves to the CTE's Scope, so no physical `orders` is read. Conversely a CTE
    alias elsewhere must not veto a real table of the same name, so CTE names are never
    subtracted globally.

    Scopes are therefore walked and only sources that sqlglot resolves to an actual
    `exp.Table` (a physical FROM/JOIN source) are collected, across every child scope
    (CTEs, derived tables, subqueries, set-operation branches). A self-JOIN still counts
    its physical table once, because the result is a set of names.
    """
    names = set()
    for scope in traverse_scope(sqlglot.parse_one(sql, read=dialect)):
        sources = {**scope.sources, **scope.lateral_sources}
        for source in sources.values():
            # A Scope source is a CTE/derived-table reference, never a physical read.
            if not isinstance(source, exp.Table):
                continue
            if source.name in schemas:
                names.add(source.name)
    return names

def guard_sql(sql: str, schemas: dict, dialect='duckdb') -> str:
    if not isinstance(sql, str) or not sql.strip() or len(sql) > 32768:
        raise ValueError('SQL must be nonempty and at most 32768 characters')
    statements = sqlglot.parse(sql, read=dialect)
    if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise ValueError('Only one SELECT/CTE/set query is allowed')
    tree = statements[0]
    prohibited = {'Into', 'Lock', 'Command', 'Pragma', 'Attach', 'Copy', 'Insert',
                  'Delete', 'Update', 'Create', 'Drop', 'Alter', 'Transaction',
                  'Parameter', 'Placeholder', 'SessionParameter', 'TableSample', 'Pivot', 'Unpivot'}
    for node in tree.walk():
        if type(node).__name__ in prohibited:
            raise ValueError(f'Unsupported SQL expression: {type(node).__name__}')
        if isinstance(node, exp.Table):
            if not isinstance(node.this, exp.Identifier) or node.db or node.catalog:
                raise ValueError('Table functions and qualified external tables are forbidden')
        if isinstance(node, exp.Func):
            name = node.name if isinstance(node, exp.Anonymous) else node.sql_name()
            if name.upper() not in SAFE_FUNCTIONS:
                raise ValueError(f'Function is not allowed: {name}')
    normalized_schema = {table: {field['name']: _sql_type(field['type']) for field in columns}
                         for table, columns in schemas.items()}
    # Qualification validates table/column resolution including CTEs and aliases.
    try:
        qualified = qualify(tree, dialect=dialect, schema=normalized_schema,
                            validate_qualify_columns=True, identify=True)
    except Exception as exc:
        raise ValueError(f'SQL schema validation failed: {exc}') from exc
    ctes = {cte.alias for cte in qualified.find_all(exp.CTE)}
    for table in qualified.find_all(exp.Table):
        if table.name not in schemas and table.name not in ctes:
            raise ValueError(f'Unknown table: {table.name}')
    return qualified.sql(dialect=dialect)

def _authorizer(action, first, second, database, trigger):
    import sqlite3
    if action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE}:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and (second or first or '').lower() in SQLITE_FUNCTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY

def _sqlite_table(source: dict, table: str, sql: str | None, max_rows: int):
    schemas = source['schema']
    if table not in schemas:
        raise ValueError(f'Unknown source table: {table}')
    # The guard validates against the FULL registered schema of this ONE source: the
    # registry of other sources is never an escape hatch.
    query = guard_sql(sql or f'SELECT * FROM "{table}"', schemas, 'sqlite')
    # `table` stays meaningful as a declared load: constant queries reference nothing.
    referenced = referenced_tables(query, schemas, 'sqlite')
    if table not in referenced:
        raise ValueError(f'Declared table is not referenced by the query: {table}')
    with closing(readonly_sqlite(Path(source['path']))) as connection:
        connection.set_authorizer(_authorizer)
        # Budget counts distinct physical tables: a self-JOIN reads its table once.
        total_rows = 0
        for name in sorted(referenced):
            quoted = name.replace('"', '""')
            total_rows += connection.execute(f'SELECT count(*) FROM "{quoted}"').fetchone()[0]
            if total_rows > max_rows:
                raise ValueError('Input row limit exceeded')
        cursor = connection.execute(query)
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise ValueError('Input row limit exceeded')
        names = [description[0] for description in cursor.description]
        if len(names) != len(set(names)):
            raise ValueError('Query output contains duplicate column names')
        data = pa.Table.from_pylist([dict(zip(names, row)) for row in rows]) if rows else pa.table({name: pa.array([], type=pa.string()) for name in names})
        explain = [list(row) for row in connection.execute('EXPLAIN QUERY PLAN ' + query)]
    return data, query, explain

def _normalize_currency(data: pa.Table) -> pa.Table:
    """DuckDB SUM(BIGINT) is HUGEINT/Arrow decimal: keep integer cents exact.

    Normalization is a host-enforced unit contract, never a lossy rounding step.
    Non-currency decimals retain their original precision and scale.
    """
    for index, field in enumerate(data.schema):
        if not field.name.endswith('_cents') or not pa.types.is_decimal(field.type):
            continue
        values = data.column(index).to_pylist()
        normalized = []
        for value in values:
            if value is None:
                normalized.append(None)
                continue
            if not value.is_finite() or value != value.to_integral_value():
                raise ValueError(f'Currency unit violation: {field.name} must contain exact integer cents')
            integer = int(value)
            if not -(2 ** 63) <= integer < 2 ** 63:
                raise ValueError(f'Currency overflow: {field.name} is outside int64 range')
            normalized.append(integer)
        replacement = pa.field(field.name, pa.int64(), nullable=field.nullable, metadata=field.metadata)
        data = data.set_column(index, replacement, pa.array(normalized, type=pa.int64()))
    return data

def _output_limit_error(observed: int, limit: int, streamed: bool = False, source_sql: bool = True) -> str:
    """Actionable limit error: the legacy phrase stays first for compatibility.

    The advice must only name capabilities that actually exist for this step: a
    registered SQLite LOAD can push work into the source, a dependent SELECT can
    reduce named inputs, but a raw file LOAD has neither until it is loaded.
    """
    observed_text = (f'observed at least {observed} rows (streamed prefix, not the exact total)'
                     if streamed else f'observed {observed} rows')
    if source_sql:
        advice = ("Aggregate, filter or JOIN inside the registered SQLite LOAD, or compute a dependent SELECT "
                  "that returns few named rows.")
    else:
        advice = ("This file LOAD accepts no SQL and no dependent SELECT can read the file until it is loaded, "
                  "so a bounded file/partition loading strategy is required: load a bounded slice of the file, "
                  "or register it as SQLite, before reducing it with SQL.")
    return "Output row limit exceeded: " + observed_text + f", configured limit {limit} rows. " + advice

def _query_child(payload: dict, results):
    try:
        step = payload['step']
        connection = duckdb.connect(':memory:', config={'memory_limit': '512MB', 'threads': '2', 'enable_external_access': 'false'})
        if step['type'] == 'LOAD':
            source = payload['source']
            if source['kind'] == 'sqlite':
                data, query, explain = _sqlite_table(source, step['table'], step.get('sql'), payload['max_input_rows'])
            else:
                if step.get('sql'):
                    raise ValueError('File LOAD does not accept SQL; use a dependent SELECT')
                if step['table'] not in source['schema']:
                    raise ValueError('Unknown file source table')
                data = file_table(Path(source['path']), source['kind'], payload['max_input_rows'])
                query, explain = None, {'kind': 'trusted_file_load', 'format': source['kind']}
        else:
            schemas, total_rows = {}, 0
            for name, artifact in payload['inputs'].items():
                metadata = pq.read_metadata(artifact['path'])
                total_rows += metadata.num_rows
                if total_rows > payload['max_input_rows']:
                    raise ValueError('Input row limit exceeded')
                table = pq.read_table(artifact['path'])
                schemas[name] = [{'name': field.name, 'type': str(field.type)} for field in table.schema]
                connection.register(name, table)
            query = guard_sql(step['sql'], schemas)
            explain = [list(row) for row in connection.execute('EXPLAIN ' + query).fetchall()]
            reader = connection.execute(query).fetch_record_batch(rows_per_batch=1024)
            batches, total = [], 0
            for batch in reader:
                total += batch.num_rows
                if total > payload['max_output_rows']:
                    raise ValueError(_output_limit_error(total, payload['max_output_rows'], streamed=True))
                batches.append(batch)
            data = pa.Table.from_batches(batches, schema=reader.schema)
        if data.num_rows > payload['max_output_rows']:
            raise ValueError(_output_limit_error(data.num_rows, payload['max_output_rows'],
                                                 source_sql=step['type'] == 'LOAD' and payload['source']['kind'] == 'sqlite'))
        if len(data.column_names) != len(set(data.column_names)):
            raise ValueError('Query output contains duplicate column names')
        data = _normalize_currency(data)
        pq.write_table(data, payload['stage'])
        connection.close()
        results.put({'ok': True, 'sql': query, 'explain': explain,
                     'row_count': data.num_rows,
                     'schema': [{'name': field.name, 'type': str(field.type)} for field in data.schema]})
    except Exception as exc:
        results.put({'ok': False, 'error': f'{type(exc).__name__}: {exc}'})

def _preview_value(value):
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {key: _preview_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_preview_value(item) for item in value]
    if value is not None and not isinstance(value, (str, int, float, bool)):
        return str(value)
    return value

class Executor:
    def __init__(self, run_dir: Path, manifest: dict, timeout=30, max_input_rows=100000, max_output_rows=10000):
        self.run_dir = Path(run_dir).resolve()
        self.manifest = manifest
        self.timeout = float(timeout)
        self.max_input_rows, self.max_output_rows = int(max_input_rows), int(max_output_rows)
        if self.timeout <= 0 or min(self.max_input_rows, self.max_output_rows) < 1:
            raise ValueError('Execution budgets must be positive')

    def execute(self, step: dict, artifacts: dict) -> dict:
        step = PlanStep.model_validate(step).model_dump(exclude_none=True)
        if step['type'] not in {'LOAD', 'SELECT'}:
            raise ValueError('Executor only handles LOAD and SELECT')
        payload = {'step': step, 'max_input_rows': self.max_input_rows, 'max_output_rows': self.max_output_rows}
        input_hashes, inputs = {}, {}
        if step['type'] == 'LOAD':
            source = next((source for source in self.manifest['sources'] if source['source_id'] == step['source_id']), None)
            if source is None:
                raise ValueError('Unknown source_id')
            path = contained(Path(source['path']), self.run_dir / 'inputs')
            if sha256(path) != source['sha256']:
                raise ValueError('Source snapshot hash mismatch')
            payload['source'] = source
            input_hashes[source['source_id']] = source['sha256']
        else:
            for dep in step['depends_on']:
                matched = [(name, value) for name, value in artifacts.items() if value['step_id'] == dep]
                if len(matched) != 1:
                    raise ValueError(f'Missing or ambiguous completed dependency: {dep}')
                name, artifact = matched[0]
                path = contained(Path(artifact['path']), self.run_dir / 'artifacts')
                if sha256(path) != artifact['sha256']:
                    raise ValueError('Input artifact hash mismatch')
                inputs[name] = artifact
                input_hashes[name] = artifact['sha256']
            payload['inputs'] = inputs
            guard_sql(step['sql'], {name: artifact['schema'] for name, artifact in inputs.items()})
        directory = self.run_dir / 'artifacts'
        directory.mkdir(parents=True, exist_ok=True)
        artifact_id = 'artifact_' + uuid.uuid4().hex
        stage = directory / ('.' + artifact_id + '.parquet')
        target = directory / (artifact_id + '.parquet')
        payload['stage'] = str(stage)
        context = mp.get_context('spawn')
        result_queue = context.Queue()
        process = context.Process(target=_query_child, args=(payload, result_queue))
        process.start()
        try:
            # Drain before join: a large EXPLAIN message can otherwise fill Queue's pipe.
            try:
                result = result_queue.get(timeout=self.timeout)
            except queue.Empty:
                process.terminate()
                raise RuntimeError('Query timed out or child exited without a result')
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
            if not result['ok']:
                raise ValueError(result['error'])
            os.replace(stage, target)
            table = pq.read_table(target)
            preview = [_preview_value(row) for row in table.slice(0, 50).to_pylist()]
            artifact = {'artifact_id': artifact_id, 'step_id': step['step_id'],
                        'output_name': step['output_name'], 'kind': 'table', 'path': str(target),
                        'sha256': sha256(target), 'row_count': result['row_count'],
                        'schema': result['schema'], 'preview': preview,
                        'sql': result['sql'], 'explain': result['explain'],
                        'input_hashes': input_hashes,
                        'lineage': {'source_id': step.get('source_id'), 'depends_on': step['depends_on'],
                                    'column_derivation': 'unknown; use supported SQL lineage where available'}}
            (directory / (artifact_id + '.json')).write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding='utf-8')
            if result['sql']:
                (directory / (artifact_id + '.sql')).write_text(result['sql'], encoding='utf-8')
            return artifact
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
            result_queue.close()
            if stage.exists():
                stage.unlink()
