"""Host-generated publication reduces protocol friction without weakening verification."""

import json
from pathlib import Path

from local_data_agent.fixtures import bootstrap
from local_data_agent.replay import ReplayModelClient, extended_steps, select
from local_data_agent.runner import AgentRunner


class Scripted(ReplayModelClient):
    def __init__(self, actions):
        self.case = "publisher-test-protocol"
        self.actions = actions


def actions(selected, steps=None):
    steps = steps or extended_steps()
    result = [
        ("bind_metric", {"metric_id": "CASH_NET_v1"}),
        ("propose_plan", {"steps_json": json.dumps(steps), "reason": "分析月度净额与渠道"}),
    ]
    result.extend(("execute_step", {"plan_version": 1, "step_id": step}) for step in selected)
    result.append(("publish_tables", {"plan_version": 1, "artifact_ids": selected,
                                      "summary": "按已批准 CASH_NET_v1 分析；数值以绑定结果为准，业务原因尚未核实。", "open_questions": []}))
    return result


def test_tables_generate_typed_money_claims_from_actual_rows(tmp_path):
    bootstrap(tmp_path)
    result = AgentRunner(tmp_path, Scripted(actions(["monthly_total", "monthly_channels"]))).run("按现金净额比较两月与渠道", mode="live")
    assert result["status"] == "SUCCEEDED", result
    claims = json.loads((Path(result["run_dir"]) / "claims.json").read_text(encoding="utf-8"))
    assert len(claims) == 6
    assert [claim["value"] for claim in claims[:2]] == [38000, 31000]
    assert all(claim["aggregation"] == "single_row" and isinstance(claim["filter"], dict) for claim in claims)
    assert all(claim["verified_sha256"] for claim in claims)


def test_refund_presentation_contains_bound_record_and_order_ids(tmp_path):
    bootstrap(tmp_path)
    result = AgentRunner(tmp_path, Scripted(actions(["refund_details"]))).run("按现金净额列出相关退款明细", mode="live")
    assert result["status"] == "SUCCEEDED", result
    claims = json.loads((Path(result["run_dir"]) / "claims.json").read_text(encoding="utf-8"))
    assert {c["value"] for c in claims if c["column"] == "refund_id"} == {"R1", "R2", "R3", "R4"}
    assert {c["value"] for c in claims if c["column"] == "order_id"} == {"O2", "O3", "O4"}


def test_correct_money_cannot_hide_fabricated_refund_channel(tmp_path):
    bootstrap(tmp_path)
    steps = extended_steps()
    steps.append(select("misassigned", ["refunds", "orders"],
                        "SELECT r.refund_id,r.order_id,'app' AS channel_code,r.refund_cents FROM refunds r JOIN orders o ON r.order_id=o.order_id WHERE r.refund_id='R2'"))
    result = AgentRunner(tmp_path, Scripted(actions(["misassigned"], steps))).run("按现金净额核验退款渠道", mode="live")
    assert result["status"] == "FAILED"
    assert result["report_path"] is None
    validation = json.loads((Path(result["run_dir"]) / "validation.json").read_text(encoding="utf-8"))
    assert not validation["passed"]


def test_raw_numeric_constant_has_no_business_verification(tmp_path):
    bootstrap(tmp_path)
    steps = extended_steps()
    steps.append(select("fabricated", ["orders"], "SELECT 999 AS refund_count,'2026-09' AS month FROM orders LIMIT 1"))
    result = AgentRunner(tmp_path, Scripted(actions(["fabricated"], steps))).run("按现金净额核验退款笔数", mode="live")
    assert result["status"] == "FAILED"
    assert result["report_path"] is None
