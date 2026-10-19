"""Metric approval and task intent are distinct authorization boundaries."""

import json
from pathlib import Path

from local_data_agent.fixtures import bootstrap
from local_data_agent.replay import ReplayModelClient, load, select
from local_data_agent.runner import AgentRunner


class Scripted(ReplayModelClient):
    def __init__(self, actions):
        self.case = "intent-test-protocol"
        self.actions = actions


def outputs(result):
    state = json.loads((Path(result["run_dir"]) / "checkpoint.json").read_text(encoding="utf-8"))
    return state, [json.loads(item["output"]) for item in state["history"] if item.get("type") == "function_call_output"]


def test_approved_definition_does_not_automatically_mean_income(tmp_path):
    bootstrap(tmp_path)
    model = Scripted([
        ("bind_metric", {"metric_id": "CASH_NET_v1"}),
        ("ask_user", {"question": "收入采用现金净额还是其他定义？", "reason": "指标含义未确认", "choices": ["现金净额", "其他定义"]}),
    ])
    result = AgentRunner(tmp_path, model).run("九月收入为什么下降？", mode="live")
    state, observed = outputs(result)
    assert result["status"] == "WAITING_INPUT"
    assert state["metric_binding"] is None
    assert "not uniquely confirmed" in observed[0]["message"]
    assert observed[1]["status"] == "WAITING_INPUT"


def test_nonfinite_query_result_does_not_poison_persistent_state(tmp_path):
    bootstrap(tmp_path)
    steps = [load("orders", "business", "orders"),
             select("bad", ["orders"], "SELECT CAST('Infinity' AS DOUBLE) AS value FROM orders LIMIT 1")]
    model = Scripted([
        ("bind_metric", {"metric_id": "CASH_NET_v1"}),
        ("propose_plan", {"steps_json": json.dumps(steps), "reason": "检查零分母保护"}),
        ("execute_step", {"plan_version": 1, "step_id": "bad"}),
        ("ask_user", {"question": "需要明确零分母处理规则", "reason": "非有限计算结果", "choices": ["不可定义", "其他规则"]}),
    ])
    result = AgentRunner(tmp_path, model).run("按现金净额检查变化率", mode="live")
    state, observed = outputs(result)
    assert result["status"] == "WAITING_INPUT"
    assert "bad" not in state["artifacts"]
    assert "Nonfinite" in observed[2]["message"]
