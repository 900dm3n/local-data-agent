"""Incremental patch_plan integration tests against the real fixture, store and dispatch."""

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from local_data_agent.agent_loop import execute_calls, pending_calls
from local_data_agent.connectors import snapshot_sources
from local_data_agent.artifacts import checked_run_dir
from local_data_agent.fixtures import bootstrap
from local_data_agent.state import StateStore
from local_data_agent.tools import ToolContext

SQL_TOTAL = "SELECT sum(paid_cents) AS paid_cents FROM orders"

ORDERS = {"step_id": "orders", "type": "LOAD", "output_name": "orders",
          "depends_on": [], "source_id": "business", "table": "orders"}
TOTAL = {"step_id": "total", "type": "SELECT", "output_name": "total",
         "depends_on": ["orders"], "sql": SQL_TOTAL}
REFUNDS = {"step_id": "refunds", "type": "LOAD", "output_name": "refunds",
           "depends_on": [], "source_id": "business", "table": "refunds"}
FILTERED = {"step_id": "filtered", "type": "SELECT", "output_name": "filtered",
            "depends_on": ["total"],
            "sql": "SELECT paid_cents FROM total WHERE paid_cents > 0"}


def _steps_json(*steps):
    return json.dumps([dict(step) for step in steps], ensure_ascii=False)


def _patch_args(version, upserts=(), removals=(), reason="patch"):
    return {"base_plan_version": version, "upsert_steps_json": _steps_json(*upserts),
            "remove_step_ids": list(removals), "reason": reason}


def _files(base):
    return sorted(str(path.relative_to(base)) for path in Path(base).rglob("*") if path.is_file())


def _snapshot(base):
    return {name: (os.stat(Path(base) / name).st_mtime_ns, (Path(base) / name).read_bytes())
            for name in _files(base)}


@pytest.fixture
def ctx(tmp_path):
    bootstrap(tmp_path)
    store = StateStore(tmp_path)
    state = store.create("cash net", "live", "cash-net-v1", "cash-net-v1")
    run_dir = checked_run_dir(tmp_path, state["run_id"])
    manifest = snapshot_sources(tmp_path, run_dir)
    state["manifest"] = manifest
    context = ToolContext(tmp_path, run_dir, state, manifest, store, None)
    try:
        yield context
    finally:
        context.retriever.close()


def _bound(context):
    context.dispatch("bind_metric", {"metric_id": "CASH_NET_v1"})
    return context


def _planned(context, *steps, reason="initial plan"):
    _bound(context)
    context.dispatch("propose_plan", {"steps_json": _steps_json(*steps), "reason": reason})
    return context


def _default_plan(context):
    return _planned(context, ORDERS, TOTAL, REFUNDS, FILTERED)


def _row(artifact):
    import pyarrow.parquet as pq

    return pq.read_table(artifact["path"]).to_pylist()


def _execute(context, step_id):
    context.dispatch("execute_step", {"plan_version": context.state["plan_version"], "step_id": step_id})


def _artifact(context, output_name):
    return context.state["artifacts"][output_name]


def test_patch_adds_steps_and_advances_version(ctx):
    _planned(ctx, ORDERS, TOTAL)
    result = ctx.dispatch("patch_plan", _patch_args(
        ctx.state["plan_version"], upserts=[REFUNDS, {"step_id": "refund_total", "type": "SELECT",
                                                      "output_name": "refund_total",
                                                      "depends_on": ["refunds"],
                                                      "sql": "SELECT sum(refund_cents) AS refund_cents FROM refunds"}],
        reason="add refund branch"))
    assert result["changed"] is True
    assert result["plan_version"] == 2
    assert sorted(result["upserted"]) == ["refund_total", "refunds"]
    assert result["removed"] == []
    assert sorted(result["affected"]) == ["refund_total", "refunds"]
    assert result["invalidated"] == []
    assert result["retained"] == []
    assert set(result) == {"plan_version", "changed", "upserted", "removed", "affected", "invalidated", "retained"}
    assert "steps" not in result and "sql" not in json.dumps(result)
    written = json.loads((ctx.run_dir / "plans" / "plan_2.json").read_text(encoding="utf-8"))
    assert written["version"] == 2 and written["base_version"] == 1 and written["patch"] is True
    assert written["reason"] == "add refund branch"
    assert sorted(written["upserts"]) == ["refund_total", "refunds"]
    assert written["removals"] == [] and written["invalidated"] == []
    assert [step["step_id"] for step in ctx.state["plan"]] == ["orders", "total", "refunds", "refund_total"]


def test_patch_replacement_invalidates_only_affected_and_repairs_downstream(ctx):
    _default_plan(ctx)
    _execute(ctx, "orders")
    _execute(ctx, "total")
    _execute(ctx, "refunds")
    _execute(ctx, "filtered")
    before = deepcopy(ctx.state["artifacts"])
    orders_path = Path(before["orders"]["path"])
    orders_stat = orders_path.stat().st_mtime_ns
    orders_sha = before["orders"]["sha256"]
    refunds_path = Path(before["refunds"]["path"])
    refunds_stat = refunds_path.stat().st_mtime_ns
    refunds_sha = before["refunds"]["sha256"]
    total_output = dict(_row(before["total"])[0])
    replacement = dict(TOTAL, sql="SELECT count(*) AS order_count FROM orders")
    result = ctx.dispatch("patch_plan", _patch_args(ctx.state["plan_version"], upserts=[replacement],
                                                    reason="count orders instead"))
    assert result["changed"] is True
    assert sorted(result["affected"]) == ["filtered", "total"]
    assert sorted(result["invalidated"]) == ["filtered", "total"]
    assert sorted(result["retained"]) == ["orders", "refunds"]
    assert set(ctx.state["artifacts"]) == {"orders", "refunds"}
    assert ctx.state["artifacts"]["orders"] == before["orders"]
    assert ctx.state["artifacts"]["refunds"] == before["refunds"]
    assert orders_path.is_file() and refunds_path.is_file()
    assert orders_path.stat().st_mtime_ns == orders_stat and refunds_path.stat().st_mtime_ns == refunds_stat
    assert before["orders"]["sha256"] == orders_sha and before["refunds"]["sha256"] == refunds_sha
    assert ctx.state["plan_version"] == 2
    _execute(ctx, "total")
    assert _row(_artifact(ctx, "total"))[0] != total_output


def test_patch_removal_requires_dependency_repair(ctx):
    _default_plan(ctx)
    with pytest.raises(ValueError):
        ctx.dispatch("patch_plan", _patch_args(ctx.state["plan_version"], removals=["total"], reason="drop total"))
    assert ctx.state["plan_version"] == 1
    assert not (ctx.run_dir / "plans" / "plan_2.json").exists()
    repaired = ctx.dispatch("patch_plan", _patch_args(
        ctx.state["plan_version"],
        upserts=[dict(FILTERED, depends_on=["orders"], sql="SELECT paid_cents FROM orders WHERE paid_cents>0")],
        removals=["total"], reason="drop total and repair dependents"))
    assert repaired["changed"] is True
    assert repaired["removed"] == ["total"]
    assert sorted(repaired["affected"]) == ["filtered", "total"]
    assert [step["step_id"] for step in ctx.state["plan"]] == ["orders", "refunds", "filtered"]


def test_empty_and_equivalent_patch_is_a_no_op(ctx):
    _default_plan(ctx)
    _execute(ctx, "orders")
    _execute(ctx, "total")
    ctx.state["claims"] = [{"claim_id": "c1"}]
    ctx.state["report_path"] = str(ctx.run_dir / "report.md")
    Path(ctx.state["report_path"]).write_text("previous report", encoding="utf-8")
    ctx.state["answer"] = "published"
    ctx.state["narrative_verification"] = "model_prose_requires_review"
    ctx.state["repair_counts"] = {"total": 2}
    before_state = deepcopy(ctx.state)
    files = _snapshot(ctx.run_dir)
    artifacts = deepcopy(ctx.state["artifacts"])

    empty = ctx.dispatch("patch_plan", _patch_args(ctx.state["plan_version"], reason="nothing to change"))
    assert empty["changed"] is False
    assert empty["plan_version"] == 1
    assert empty["upserted"] == [] and empty["removed"] == []
    assert empty["affected"] == [] and empty["invalidated"] == []
    assert sorted(empty["retained"]) == ["orders", "total"]
    assert set(empty) == {"plan_version", "changed", "upserted", "removed", "affected", "invalidated", "retained"}
    assert "steps" not in empty and "sql" not in json.dumps(empty)
    assert ctx.state["claims"] == [{"claim_id": "c1"}]
    assert ctx.state["report_path"] == before_state["report_path"]
    assert ctx.state["answer"] == "published"
    assert ctx.state["narrative_verification"] == "model_prose_requires_review"
    assert ctx.state["repair_counts"] == {"total": 2}
    assert ctx.state["plan_version"] == 1
    assert _files(ctx.run_dir) == list(files)
    assert _snapshot(ctx.run_dir) == files
    assert ctx.state["artifacts"] == artifacts

    equivalent = ctx.dispatch("patch_plan", _patch_args(
        ctx.state["plan_version"], upserts=[dict(TOTAL), dict(ORDERS)], reason="resubmit identical"))
    assert equivalent["changed"] is False
    assert equivalent["upserted"] == ["total", "orders"]
    assert equivalent["affected"] == [] and equivalent["invalidated"] == []
    assert not (ctx.run_dir / "plans" / "plan_2.json").exists()
    assert ctx.state["plan"] == before_state["plan"]
    assert ctx.state["claims"] == [{"claim_id": "c1"}] and ctx.state["narrative_verification"] == "model_prose_requires_review"
    assert _files(ctx.run_dir) == list(files)

    added = ctx.dispatch("patch_plan", _patch_args(
        ctx.state["plan_version"], upserts=[REFUNDS, {"step_id": "refund_total", "type": "SELECT",
                                                      "output_name": "refund_total",
                                                      "depends_on": ["refunds"],
                                                      "sql": "SELECT sum(refund_cents) AS refund_cents FROM refunds"}],
        reason="add refunds"))
    assert added["changed"] is True and added["plan_version"] == 2
    assert ctx.state["claims"] == []
    assert "report_path" not in ctx.state and "answer" not in ctx.state
    assert "narrative_verification" not in ctx.state
    assert ctx.state["repair_counts"] == {"total": 2}
    assert sorted(ctx.state["artifacts"]) == ["orders", "total"]
    assert Path(before_state["report_path"]).is_file()
    assert Path(before_state["artifacts"]["orders"]["path"]).is_file()


@pytest.mark.parametrize("mutate", [
    lambda args, state: args.update({"base_plan_version": state["plan_version"] + 1}),
    lambda args, state: args.update({"base_plan_version": True}),
    lambda args, state: args.update({"base_plan_version": 1.0}),
    lambda args, state: args.update({"upsert_steps_json": json.dumps([dict(TOTAL, depends_on=["missing"])])}),
    lambda args, state: args.update({"upsert_steps_json": "not json"}),
    lambda args, state: args.update({"upsert_steps_json": json.dumps([dict(TOTAL), dict(TOTAL)])}),
    lambda args, state: args.update({"remove_step_ids": ["unknown_step"]}),
])
def test_invalid_patch_is_rejected_without_side_effects(ctx, mutate):
    _default_plan(ctx)
    args = _patch_args(ctx.state["plan_version"], upserts=[REFUNDS], reason="invalid attempt")
    mutate(args, ctx.state)
    before_state = deepcopy(ctx.state)
    before_files = _snapshot(ctx.run_dir)

    with pytest.raises(ValueError):
        ctx.dispatch("patch_plan", args)

    assert ctx.state == before_state
    assert ctx.state["plan_version"] == 1
    assert ctx.state["artifacts"] == before_state["artifacts"]
    assert _files(ctx.run_dir) == sorted(before_files)
    assert _snapshot(ctx.run_dir) == before_files


def test_patch_without_plan_or_binding_is_rejected(ctx):
    state = deepcopy(ctx.state)
    with pytest.raises(ValueError):
        ctx.dispatch("patch_plan", _patch_args(0, upserts=[ORDERS]))
    assert ctx.state == state
    _bound(ctx)
    with pytest.raises(ValueError):
        ctx.dispatch("patch_plan", _patch_args(0, upserts=[ORDERS]))
    assert ctx.state["plan"] == []
    assert ctx.state["plan_version"] == 0
    assert list((ctx.run_dir / "plans").glob("*.json")) == []


def test_write_failure_leaves_memory_unchanged(ctx, monkeypatch):
    _default_plan(ctx)
    before_state = deepcopy(ctx.state)
    before_files = _snapshot(ctx.run_dir)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("local_data_agent.tools.write_json", boom)
    with pytest.raises(OSError):
        ctx.dispatch("patch_plan", _patch_args(ctx.state["plan_version"], upserts=[dict(TOTAL, sql="SELECT count(*) AS n FROM orders")], reason="io failure"))
    assert ctx.state == before_state
    assert _snapshot(ctx.run_dir) == before_files


def test_coordinator_alias_repair_reuses_recomputed_parent(ctx):
    _default_plan(ctx)
    _execute(ctx, "filtered")
    previous = deepcopy(ctx.state["artifacts"])
    ctx.dispatch("patch_plan", _patch_args(1, upserts=[dict(TOTAL, output_name="total_v2")]))
    assert set(ctx.state['artifacts']) == {'orders'}
    with pytest.raises(ValueError):
        _execute(ctx, 'filtered')  # Depends on correct ID, but SQL still names old alias.
    recomputed = deepcopy(ctx.state['artifacts']['total_v2'])
    assert recomputed['artifact_id'] != previous['total']['artifact_id']
    ctx.dispatch('patch_plan', _patch_args(2, upserts=[dict(FILTERED, sql='SELECT paid_cents FROM total_v2 WHERE paid_cents>0')]))
    output = ctx.dispatch('execute_step', {'plan_version': 3, 'step_id': 'filtered'})
    assert ctx.state['artifacts']['total_v2'] == recomputed
    assert ctx.state['artifacts']['orders'] == previous['orders']
    assert len(output['executed_outputs']) == 1
    assert _row(ctx.state['artifacts']['filtered']) == _row(previous['filtered'])
    assert Path(previous['filtered']['path']).exists()


def test_coordinator_patch_cannot_bypass_sql_guard(ctx):
    from local_data_agent.artifacts import file_hash
    _planned(ctx, ORDERS)
    _execute(ctx, 'orders')
    before = deepcopy(ctx.state['artifacts']['orders'])
    source_hashes = {s['path']: file_hash(s['path']) for s in ctx.manifest['sources']}
    bad = dict(TOTAL, sql='DELETE FROM orders')
    ctx.dispatch('patch_plan', _patch_args(1, upserts=[bad]))
    with pytest.raises(ValueError):
        _execute(ctx, 'total')
    assert ctx.state['artifacts'] == {'orders': before}
    assert all(file_hash(p) == h for p, h in source_hashes.items())
    with pytest.raises(ValueError, match='Stale plan version'):
        ctx.dispatch('execute_step', {'plan_version': 1, 'step_id': 'orders'})


def test_coordinator_remove_leaf_keeps_historical_artifact_file(ctx):
    from local_data_agent.artifacts import file_hash
    _default_plan(ctx)
    _execute(ctx, 'filtered')
    before = deepcopy(ctx.state['artifacts'])
    ctx.dispatch('patch_plan', _patch_args(1, removals=['filtered']))
    assert 'filtered' not in ctx.state['artifacts']
    for name in ('orders', 'total'):
        assert ctx.state['artifacts'][name] == before[name]
    assert file_hash(before['filtered']['path']) == before['filtered']['sha256']


def test_execute_calls_is_idempotent_for_patch(ctx):
    _planned(ctx, ORDERS, TOTAL)
    ctx.state["status"] = "RUNNING"
    args = _patch_args(ctx.state["plan_version"], upserts=[REFUNDS], reason="idempotent patch")
    call = {"call_id": "patch_unique", "name": "patch_plan", "arguments": args}
    execute_calls(ctx, [call])
    execute_calls(ctx, [call])

    replies = [item for item in ctx.state["history"] if item.get("type") == "function_call_output"]
    assert len(replies) == 2
    first = json.loads(replies[0]["output"])
    assert first["changed"] is True and first["plan_version"] == 2
    assert json.loads(replies[1]["output"]) == first
    assert [step["step_id"] for step in ctx.state["plan"]] == ["orders", "total", "refunds"]
    assert pending_calls(ctx.state["history"]) == []
    stored = ctx.store.load(ctx.state["run_id"])
    assert stored["plan_version"] == 2
    assert [step["step_id"] for step in stored["plan"]] == ["orders", "total", "refunds"]
