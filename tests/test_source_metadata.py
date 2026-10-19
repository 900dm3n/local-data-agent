import pytest
from local_data_agent.connectors import sha256, snapshot_sources
from local_data_agent.executor import Executor
from local_data_agent.fixtures import bootstrap
from local_data_agent.retrieval import ContextRetriever


@pytest.fixture
def retriever(tmp_path):
    bootstrap(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    instance = ContextRetriever(run, snapshot_sources(tmp_path, run))
    yield instance
    instance.close()


def test_sqlite_count_is_exact_and_metadata_preserved(retriever):
    result = retriever.inspect_schema("business", "orders")
    assert result["row_count"] == 6
    assert result["source_kind"] == "sqlite"
    assert result["query_dialect"] == "sqlite"
    assert result["default_load_output_rows"] == 10000
    assert result["raw_load_exceeds_default_output_limit"] is False
    assert result["source_id"] == "business" and result["table"] == "orders"
    assert [field["name"] for field in result["schema"]][:2] == ["order_id", "paid_at"]
    assert result["relationships"] == [{"column": "channel_code", "references": "channels.channels.channel_code", "cardinality": "many-to-one"}]
    assert result["schema_hash"] == retriever.inspect_schema("business", "orders")["schema_hash"]
    assert "path" not in result


def test_file_source_count_and_dialect(retriever):
    csv_row = retriever.inspect_schema("channels", "channels")
    assert (csv_row["row_count"], csv_row["source_kind"]) == (2, "csv")
    # A file LOAD accepts no SQL: the file advertises the artifact dialect, denies source SQL.
    assert csv_row["query_dialect"] is None
    assert csv_row["artifact_query_dialect"] == "duckdb"
    assert csv_row["source_sql_allowed"] is False
    csv_hash = csv_row["schema_hash"]
    tickets = retriever.inspect_schema("tickets", "tickets")
    assert (tickets["row_count"], tickets["source_kind"]) == (5, "jsonl")
    assert tickets["query_dialect"] is None
    assert tickets["source_sql_allowed"] is False
    assert retriever.inspect_schema("channels", "channels")["schema_hash"] == csv_hash


def test_sqlite_source_dialect_allows_source_sql(retriever):
    result = retriever.inspect_schema("business", "orders")
    assert result["query_dialect"] == "sqlite"
    assert result["artifact_query_dialect"] == "duckdb"
    assert result["source_sql_allowed"] is True


def test_registered_quoted_table_name_counts_readonly(tmp_path):
    import sqlite3
    from pathlib import Path
    path = tmp_path / "quoted.sqlite"
    weird = 'odd"table'
    with sqlite3.connect(path) as db:
        db.execute(f'CREATE TABLE "{weird.replace(chr(34), chr(34) * 2)}"(v INTEGER)')
        db.executemany(f'INSERT INTO "{weird.replace(chr(34), chr(34) * 2)}" VALUES(?)', [(i,) for i in range(3)])
    manifest = {"sources": [{"source_id": "quoted", "kind": "sqlite", "path": str(Path(path)),
                             "schema": {weird: [{"name": "v", "type": "INTEGER"}]}}], "documents": []}
    instance = ContextRetriever(tmp_path / "run", manifest)
    result = instance.inspect_schema("quoted", weird)
    assert result["row_count"] == 3
    assert result["query_dialect"] == "sqlite"
    assert result["table"] == weird
    with pytest.raises(ValueError, match="Unknown source or table"):
        instance.inspect_schema("quoted", 'odd"table; DROP TABLE x')
    instance.close()


def test_unknown_table_and_source_still_rejected(retriever):
    with pytest.raises(ValueError, match="Unknown source or table"):
        retriever.inspect_schema("business", "does_not_exist")
    with pytest.raises(ValueError, match="Unknown source or table"):
        retriever.inspect_schema("missing", "orders")


def test_parquet_metadata_count(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    (tmp_path / "inputs").mkdir()
    pq.write_table(pa.table({"code": ["web", "app", "web"]}), tmp_path / "inputs" / "reference.parquet")
    manifest = {"sources": [{"source_id": "reference", "kind": "parquet",
                             "path": str(tmp_path / "inputs" / "reference.parquet"),
                             "schema": {"reference": [{"name": "code", "type": "string"}]}}], "documents": []}
    instance = ContextRetriever(tmp_path / "run", manifest)
    result = instance.inspect_schema("reference", "reference")
    assert (result["row_count"], result["query_dialect"], result["artifact_query_dialect"]) == (3, None, "duckdb")
    assert result["source_sql_allowed"] is False
    instance.close()


def test_large_table_reports_exceeding_default_limit(tmp_path):
    import sqlite3
    from pathlib import Path
    path = tmp_path / "big.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE events(event_id INTEGER, kind TEXT)")
        db.executemany("INSERT INTO events VALUES(?,?)", [(i, "pay") for i in range(20001)])
    manifest = {"sources": [{"source_id": "big", "kind": "sqlite", "path": str(Path(path)),
                             "schema": {"events": [{"name": "event_id", "type": "INTEGER"}]}}], "documents": []}
    instance = ContextRetriever(tmp_path / "run", manifest)
    result = instance.inspect_schema("big", "events")
    assert result["row_count"] == 20001
    assert result["raw_load_exceeds_default_output_limit"] is True
    instance.close()


def test_output_limit_error_is_actionable_and_keeps_phrase(tmp_path):
    import json
    run = tmp_path / "run"
    run.mkdir()
    (run / "inputs").mkdir()
    source = run / "inputs" / "tickets.jsonl"
    source.write_text("".join(json.dumps({"ticket_id": "T%d" % i}) + "\n" for i in range(4)), encoding="utf-8")
    manifest = {"sources": [{"source_id": "tickets", "kind": "jsonl", "path": str(source), "sha256": sha256(source),
                             "schema": {"tickets": [{"name": "ticket_id", "type": "string"}]}}]}
    with pytest.raises(ValueError) as raised:
        Executor(run, manifest, max_output_rows=2).execute(
            {"step_id": "tickets", "type": "LOAD", "output_name": "tickets", "source_id": "tickets", "table": "tickets"}, {})
    message = str(raised.value)
    assert "Output row limit exceeded" in message
    assert "observed 4 rows" in message
    assert "configured limit 2 rows" in message
    assert "bounded file/partition" in message


def test_file_load_limit_advice_does_not_suggest_sql_on_file(tmp_path):
    import json
    run = tmp_path / "run"
    run.mkdir()
    (run / "inputs").mkdir()
    source = run / "inputs" / "events.jsonl"
    source.write_text("".join(json.dumps({"event_id": i}) + "\n" for i in range(4)), encoding="utf-8")
    manifest = {"sources": [{"source_id": "events", "kind": "jsonl", "path": str(source), "sha256": sha256(source),
                             "schema": {"events": [{"name": "event_id", "type": "bigint"}]}}]}
    with pytest.raises(ValueError) as raised:
        Executor(run, manifest, max_output_rows=2).execute(
            {"step_id": "events", "type": "LOAD", "output_name": "events", "source_id": "events", "table": "events"}, {})
    message = str(raised.value)
    assert "Output row limit exceeded" in message
    assert "observed 4 rows" in message and "configured limit 2 rows" in message
    assert "bounded file/partition loading strategy is required" in message
    assert "registered SQLite LOAD" not in message
    assert "no dependent SELECT can read the file until it is loaded" in message
