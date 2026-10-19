"""Explicit protocol fixtures. This module is never used by the live adapter."""

import json


def load(step_id, source_id, table, sql=None):
    value = {"step_id": step_id, "output_name": step_id, "type": "LOAD",
             "source_id": source_id, "table": table, "depends_on": []}
    if sql:
        value["sql"] = sql
    return value


def select(step_id, dependencies, sql):
    return {"step_id": step_id, "output_name": step_id, "type": "SELECT", "depends_on": dependencies, "sql": sql}


def base_steps():
    month = lambda field: f"strftime(timezone('Australia/Sydney', CAST({field} AS TIMESTAMPTZ)), '%Y-%m')"
    steps = [load("orders", "business", "orders", "SELECT order_id,paid_at,channel_code,paid_cents,status FROM orders"),
             load("refunds", "business", "refunds", "SELECT refund_id,order_id,refunded_at,refund_cents,status FROM refunds"),
             load("channels", "channels", "channels")]
    steps.append(select("cash_events", ["orders", "refunds"],
        f"SELECT {month('paid_at')} AS month,channel_code,paid_cents AS signed_cents,'payment' AS event_type FROM orders WHERE status='paid' "
        f"UNION ALL SELECT {month('r.refunded_at')} AS month,o.channel_code,-r.refund_cents AS signed_cents,'refund' AS event_type FROM refunds r JOIN orders o ON o.order_id=r.order_id WHERE r.status='succeeded'"))
    steps.append(select("monthly_total", ["cash_events"],
                        "SELECT month,CAST(SUM(signed_cents) AS BIGINT) AS net_cents FROM cash_events GROUP BY month ORDER BY month"))
    return steps


def extended_steps():
    steps = base_steps()
    steps.append(select("monthly_channels", ["cash_events", "channels"],
        "SELECT e.month,e.channel_code,c.channel_name,CAST(SUM(e.signed_cents) AS BIGINT) AS net_cents FROM cash_events e JOIN channels c ON e.channel_code=c.channel_code GROUP BY e.month,e.channel_code,c.channel_name ORDER BY e.month,e.channel_code"))
    steps.append(select("components", ["cash_events"],
        "SELECT month,CAST(SUM(CASE WHEN event_type='payment' THEN signed_cents ELSE 0 END) AS BIGINT) AS paid_cents,CAST(-SUM(CASE WHEN event_type='refund' THEN signed_cents ELSE 0 END) AS BIGINT) AS refund_cents FROM cash_events GROUP BY month ORDER BY month"))
    steps.append(select("refund_details", ["refunds", "orders"],
        "SELECT r.refund_id,r.order_id,r.refunded_at,o.channel_code,r.refund_cents FROM refunds r JOIN orders o ON o.order_id=r.order_id WHERE r.status='succeeded' ORDER BY r.refunded_at"))
    return steps


def demo_claims():
    return [
        {"claim_id": "aug_net", "artifact_id": "monthly_total", "column": "net_cents", "filter": {"month": "2026-08"}, "aggregation": "single_row", "display_unit": "CNY_yuan", "label": "八月现金净额"},
        {"claim_id": "sep_net", "artifact_id": "monthly_total", "column": "net_cents", "filter": {"month": "2026-09"}, "aggregation": "single_row", "display_unit": "CNY_yuan", "label": "九月现金净额"},
        {"claim_id": "sep_web", "artifact_id": "monthly_channels", "column": "net_cents", "filter": {"month": "2026-09", "channel_code": "web"}, "aggregation": "single_row", "display_unit": "CNY_yuan", "label": "九月官网现金净额"},
        {"claim_id": "sep_app", "artifact_id": "monthly_channels", "column": "net_cents", "filter": {"month": "2026-09", "channel_code": "app"}, "aggregation": "single_row", "display_unit": "CNY_yuan", "label": "九月移动端现金净额"},
    ]


class ReplayModelClient:
    mode = "replay"

    def __init__(self, case="cash-net-v1"):
        self.case = case
        self.actions = [
            ("search_context", {"query": "现金净额 支付 退款 渠道", "source_ids": [], "types": []}),
            ("bind_metric", {"metric_id": "CASH_NET_v1"}),
            ("propose_plan", {"steps_json": json.dumps(base_steps()), "reason": "先建立事件净额与月度总量"}),
            ("execute_step", {"plan_version": 1, "step_id": "monthly_total"}),
            ("record_observation", {"facts_json": json.dumps([{"artifact_id": "monthly_total", "description": "已得到月度净额"}]),
                                    "open_questions": ["支付与退款的分量如何变化？", "哪个渠道贡献差额？"], "next_action": "追加分量与渠道分析"}),
            ("revise_plan", {"base_plan_version": 1, "steps_json": json.dumps(extended_steps()), "reason": "总量查询成功，仍需分解变化"}),
            ("execute_step", {"plan_version": 2, "step_id": "monthly_channels"}),
            ("execute_step", {"plan_version": 2, "step_id": "components"}),
            ("execute_step", {"plan_version": 2, "step_id": "refund_details"}),
            ("check_result", {"artifact_ids": ["monthly_total", "monthly_channels"], "check_ids": ["integrity", "finite", "monthly_reconciliation"]}),
            ("finalize_run", {"plan_version": 2, "claims_json": json.dumps(demo_claims(), ensure_ascii=False),
                              "summary": "协议回放已形成月度总额、渠道分解、支付退款分量和明细产物；实际数值以绑定事实和产物为准。原因边界：工单缺少交易关联，不能据此证明客户退款原因。本次为预设协议检查。", "open_questions": []}),
        ]
        if case == "clarification":
            self.actions.insert(1, ("ask_user", {"question": "你要分析现金净额还是其他收入定义？", "reason": "收入口径未明确", "choices": ["现金净额", "其他口径"]}))
        elif case == "repair":
            bad = base_steps()
            bad[-1] = select("monthly_total", ["cash_events"], "SELECT month,SUM(missing_amount) AS net_cents FROM cash_events GROUP BY month")
            self.actions[2] = ("propose_plan", {"steps_json": json.dumps(bad), "reason": "演示错误字段反馈"})
            self.actions.insert(4, ("revise_plan", {"base_plan_version": 1, "steps_json": json.dumps(base_steps()), "reason": "依据未知字段错误修正"}))
            self.actions.insert(5, ("execute_step", {"plan_version": 2, "step_id": "monthly_total"}))
            for name, args in self.actions[7:]:
                if "base_plan_version" in args:
                    args["base_plan_version"] = 2
                elif "plan_version" in args:
                    args["plan_version"] = 3
        elif case == "budget":
            self.actions = [("search_context", {"query": "未找到的词", "source_ids": [], "types": []})] * 32
        elif case not in {"cash-net-v1", "exploration"}:
            raise ValueError("Unknown replay case; replay uses declared protocol fixtures only")

    def metadata(self):
        return {"mode": "replay", "model": "declared-protocol-fixture", "case": self.case, "autonomous": False}

    def respond(self, history, tools):
        cursor = sum(item.get("type") == "function_call_output" for item in history)
        if cursor >= len(self.actions):
            return {"protocol_items": [], "tool_calls": [], "text": "Replay protocol ended", "usage": {}, "status": "completed"}
        name, arguments = self.actions[cursor]
        call = {"type": "function_call", "call_id": f"replay_{cursor}", "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False)}
        return {"protocol_items": [call], "tool_calls": [{key: call[key] for key in ("call_id", "name", "arguments")}],
                "text": "", "usage": {"replay": True}, "status": "completed"}

    def semantic(self, texts, prompt_version="ticket-label-v1"):
        raise ValueError("Replay does not pretend to perform real semantic classification")
