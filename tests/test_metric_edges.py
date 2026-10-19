"""Metric checks distinguish raw data, filtered results and empty calendar periods."""

import json
from pathlib import Path

from local_data_agent.artifacts import checked_run_dir
from local_data_agent.connectors import snapshot_sources
from local_data_agent.fixtures import bootstrap, compute_gold
from local_data_agent.replay import ReplayModelClient, base_steps, load, select
from local_data_agent.runner import AgentRunner
from local_data_agent.state import StateStore
from local_data_agent.tools import ToolContext
from local_data_agent.validation import evaluate_gold


def make_context(root, variant=None):
    bootstrap(root, variant=variant)
    store = StateStore(root)
    state = store.create("按现金净额口径核查", "replay", "cash-net-v1", "cash-net-v1")
    folder = checked_run_dir(root, state["run_id"])
    manifest = snapshot_sources(root, folder)
    state.update(manifest=manifest, status="RUNNING")
    context = ToolContext(root, folder, state, manifest, store, ReplayModelClient())
    context.dispatch("bind_metric", {"metric_id": "CASH_NET_v1"})
    return context


def test_raw_failed_event_can_be_inspected_but_not_published_as_cash(tmp_path):
    context = make_context(tmp_path, "failed-events")
    context.dispatch("propose_plan", {"steps_json": json.dumps([load("orders", "business", "orders")]), "reason": "核查原始记录"})
    context.dispatch("execute_step", {"plan_version": 1, "step_id": "orders"})
    selected = context.state["artifacts"]
    assert context.reconcile(selected)["passed"]
    assert not context.reconcile(selected, publication=True)["passed"]
    context.retriever.close()


def test_zero_calendar_month_and_null_change_rate_publish_without_infinity(tmp_path):
    registry = bootstrap(tmp_path, variant="zero-denominator")
    steps = base_steps()
    steps.append(select("comparison", ["monthly_total"], """
        SELECT '2026-08' AS base_month,'2026-09' AS current_month,
        CAST(COALESCE(MAX(CASE WHEN month='2026-08' THEN net_cents END),0) AS BIGINT) AS base_net_cents,
        CAST(COALESCE(MAX(CASE WHEN month='2026-09' THEN net_cents END),0) AS BIGINT) AS current_net_cents,
        CAST(COALESCE(MAX(CASE WHEN month='2026-09' THEN net_cents END),0)-COALESCE(MAX(CASE WHEN month='2026-08' THEN net_cents END),0) AS BIGINT) AS delta_cents,
        CAST(NULL AS DOUBLE) AS change_pct FROM monthly_total
    """))
    model = ReplayModelClient()
    model.actions = [
        ("bind_metric", {"metric_id": "CASH_NET_v1"}),
        ("propose_plan", {"steps_json": json.dumps(steps), "reason": "空月份补零"}),
        ("execute_step", {"plan_version": 1, "step_id": "comparison"}),
        ("publish_tables", {"plan_version": 1, "artifact_ids": ["comparison"], "summary": "上期无成功事件，变化率不可定义；未做业务因果推断。", "open_questions": []}),
    ]
    result = AgentRunner(tmp_path, model).run("按现金净额比较月份并处理零分母", mode="replay")
    assert result["status"] == "SUCCEEDED", result
    assert "不可定义" in result["answer"]
    claims = json.loads((Path(result["run_dir"]) / "claims.json").read_text(encoding="utf-8"))
    assert next(c["value"] for c in claims if c["column"] == "base_net_cents") == 0
    assert next(c["value"] for c in claims if c["column"] == "change_pct") is None
    assert evaluate_gold(result["artifacts"], compute_gold(registry))["passed"]


def test_unknown_channel_zero_is_not_a_valid_mapped_channel(tmp_path):
    context = make_context(tmp_path)
    steps = [load("orders", "business", "orders"),
             select("fake_channel", ["orders"], "SELECT '2026-09' AS month,'ghost' AS channel_code,CAST(0 AS BIGINT) AS net_cents FROM orders LIMIT 1")]
    context.dispatch("propose_plan", {"steps_json": json.dumps(steps), "reason": "检查未知维度"})
    context.dispatch("execute_step", {"plan_version": 1, "step_id": "fake_channel"})
    assert not context.reconcile({"fake_channel": context.state["artifacts"]["fake_channel"]})["passed"]
    context.retriever.close()
