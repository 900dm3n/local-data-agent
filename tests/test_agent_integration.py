"""Scripted behavior integration tests, explicitly NOT autonomous-model evidence."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from local_data_agent.fixtures import bootstrap, compute_gold
from local_data_agent.memory import MemoryStore
from local_data_agent.replay import ReplayModelClient
from local_data_agent.runner import AgentRunner
from local_data_agent.semantic_ops import run_operator
from local_data_agent.validation import evaluate_gold


class ScriptedClient(ReplayModelClient):
    def __init__(self, actions):
        self.actions = actions
        self.case = "test-injected-protocol"


def checkpoint(result):
    return json.loads((Path(result["run_dir"])/"checkpoint.json").read_text(encoding="utf-8"))


def tool_outputs(state):
    return [json.loads(item["output"]) for item in state["history"] if item.get("type") == "function_call_output"]


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp("agent-base")
    registry = bootstrap(root)
    runner = AgentRunner(root)
    result = runner.run("按现金净额解释2026年9月变化", mode="replay")
    assert result["status"].lower() == "succeeded", result
    return root, registry, runner, result, checkpoint(result)


def test_A01_base_numeric_gold(base):
    _, registry, _, result, _ = base
    assert evaluate_gold(result["artifacts"], compute_gold(registry))["passed"]


def test_A02_multisource_channel_csv_join(base):
    _, _, _, result, _ = base
    assert {r["channel_name"] for r in pq.read_table(result["artifacts"]["monthly_channels"]["path"]).to_pylist()} == {"官网","移动端"}


def test_A03_independent_refund_event_and_one_to_many(base):
    _, _, _, result, _ = base
    components = pq.read_table(result["artifacts"]["components"]["path"]).to_pylist()
    assert components == [{"month":"2026-08","paid_cents":40000,"refund_cents":2000},{"month":"2026-09","paid_cents":40000,"refund_cents":9000}]


def test_A04_protocol_call_output_pairing(base):
    history = base[4]["history"]
    calls = [i["call_id"] for i in history if i.get("type") == "function_call"]
    outputs = [i["call_id"] for i in history if i.get("type") == "function_call_output"]
    assert calls == outputs and len(calls) >= 10


def test_A05_snapshot_and_bound_metric_persisted(base):
    result, state = base[3], base[4]
    folder = Path(result["run_dir"])
    for name in ("input_manifest.json","metric_contract.json","retrieval.json","agent_state.json","plan.json","transcript.jsonl","validation.json","lineage.json","claims.json","report.md","manifest.json","artifacts.json"):
        assert (folder/name).exists(), name
    assert state["metric_binding"] == "CASH_NET_v1"


def test_A06_gold_not_in_model_context(base):
    history = base[4]["history"]
    assert all("reference_sql" not in json.dumps(item) and "monthly_cents" not in json.dumps(item) for item in history)
    first = next(json.loads(i["output"]) for i in history if i.get("type") == "function_call_output")
    assert "38000" not in json.dumps(first) and "31000" not in json.dumps(first)


def test_A07_normal_result_triggers_more_analysis(base):
    state = base[4]
    calls = [i["name"] for i in state["history"] if i.get("type") == "function_call"]
    assert calls.index("record_observation") > calls.index("execute_step")
    assert calls.index("revise_plan") > calls.index("record_observation")
    assert len([i for i in calls if i == "execute_step"]) >= 4
    assert state["observed_facts"] and not state["open_questions"]


def test_A08_revision_preserves_unchanged_inputs(base):
    state = base[4]
    revision = next(o for o in tool_outputs(state) if o.get("plan_version") == 2 and "retained" in o)
    assert {"orders","refunds","channels","cash_events","monthly_total"}.issuperset(set(revision["retained"]))
    assert {"orders","refunds","cash_events","monthly_total"}.issubset(set(revision["retained"]))
    assert revision["invalidated"] == []


def test_A09_headlines_have_artifact_bindings(base):
    state = base[4]
    assert len(state["claims"]) >= 4
    ids = {a["artifact_id"] for a in state["artifacts"].values()} | set(state["artifacts"])
    assert all(c["artifact_id"] in ids and c["column"] == "net_cents" for c in state["claims"])


def test_A10_publication_reconciled_with_frozen_events(base):
    validation = json.loads((Path(base[3]["run_dir"])/"validation.json").read_text(encoding="utf-8"))
    assert validation["passed"]
    assert any(c["name"].startswith("source_event_reconciliation:") for c in validation["checks"])


def test_A11_completed_run_creates_only_draft_memory(base):
    cards = MemoryStore(base[0]).list_cards()
    assert cards and all(c["status"] == "DRAFT" for c in cards)
    assert MemoryStore(base[0]).recall("", {"fixture":"cash-net-v1","metric":"CASH_NET_v1"}) == []


def test_A12_run_events_monotonic_and_inspectable(base):
    inspect = base[2].inspect(base[3]["run_id"])
    events = inspect["events"]
    assert events and [e["event_id"] for e in events] == sorted({e["event_id"] for e in events})
    assert inspect["plan_version"] == 2 and inspect["stop_reason"] == "validated_answer_published"


@pytest.fixture(scope="module")
def repaired(tmp_path_factory):
    root = tmp_path_factory.mktemp("agent-repair")
    registry = bootstrap(root)
    result = AgentRunner(root).run("修复字段后计算现金净额", case="repair")
    return root,registry,result,checkpoint(result)


def test_A13_bad_column_gets_feedback_and_repair(repaired):
    _,registry,result,state = repaired
    assert result["status"].lower() == "succeeded", result
    errors = [o for o in tool_outputs(state) if "error" in o]
    assert errors and any("missing_amount" in o.get("message", "") for o in errors)
    assert evaluate_gold(result["artifacts"],compute_gold(registry))["passed"]


def test_A14_error_and_corrected_plans_both_saved(repaired):
    folder = Path(repaired[2]["run_dir"])/"plans"
    p1,p2 = [json.loads((folder/f"plan_{n}.json").read_text(encoding="utf-8")) for n in (1,2)]
    assert "missing_amount" in json.dumps(p1)
    assert "missing_amount" not in json.dumps(p2)
    assert repaired[3]["plan_version"] == 3


def test_A15_repair_retains_successful_upstream_artifacts(repaired):
    output = next(o for o in tool_outputs(repaired[3]) if o.get("plan_version") == 2 and "retained" in o)
    assert {"orders","refunds","cash_events"}.issubset(output["retained"])
    assert repaired[3]["repair_counts"].get("monthly_total") == 1


def test_A16_clarification_and_resume_same_snapshot(tmp_path):
    registry = bootstrap(tmp_path)
    runner = AgentRunner(tmp_path)
    paused = runner.run("收入为什么下降", case="clarification")
    assert paused["status"].lower() == "waiting_input" and paused["question"]
    old = checkpoint(paused)["manifest"]
    # Mutating live source while paused must not change the frozen run's answer.
    import sqlite3
    with sqlite3.connect(registry["sources"][0]["path"]) as db:
        db.execute("UPDATE orders SET paid_cents=999999 WHERE order_id='O1'")
    completed = AgentRunner(tmp_path).resume(paused["run_id"],answer="现金净额，采用批准定义")
    assert completed["status"].lower() == "succeeded", completed
    assert checkpoint(completed)["manifest"] == old
    rows = pq.read_table(completed["artifacts"]["monthly_total"]["path"]).to_pylist()
    assert rows == [{"month":"2026-08","net_cents":38000},{"month":"2026-09","net_cents":31000}]


def test_A17_waiting_input_requires_user_answer(tmp_path):
    bootstrap(tmp_path)
    runner = AgentRunner(tmp_path)
    result = runner.run("收入变化", case="clarification")
    with pytest.raises(ValueError,match="answer"):
        runner.resume(result["run_id"])


def test_A18_budget_stops_without_publishing(tmp_path):
    bootstrap(tmp_path)
    result = AgentRunner(tmp_path).run("无限检索",case="budget")
    assert result["status"].lower() == "failed" and result["stop_reason"] == "model_request_budget_exceeded"
    assert result["model_requests"] == 16 and not result.get("report_path")


def test_A19_held_out_results_source_derived(tmp_path):
    registry = bootstrap(tmp_path,seed=73,variant="held-out")
    result = AgentRunner(tmp_path).run("按现金净额口径计算新金额的净额")
    assert result["status"].lower() == "succeeded",result
    gold = compute_gold(registry)
    assert evaluate_gold(result["artifacts"],gold)["passed"]
    assert gold["monthly_cents"] != {"2026-08":38000,"2026-09":31000}


def test_A20_missing_metric_can_ask_user(tmp_path):
    bootstrap(tmp_path,variant="no-metric")
    actions = [("search_context", {"query":"收入口径","source_ids":[],"types":[]}),
               ("ask_user", {"question":"没有批准定义，请明确要按哪个事件月口径计算？","reason":"missing_context","choices":["现金净额","其他"]})]
    result = AgentRunner(tmp_path, model=ScriptedClient(actions)).run("净收入是多少？")
    assert result["status"].lower() == "waiting_input" and not result["artifacts"]


def test_A21_invalid_source_stops_before_model(tmp_path):
    bootstrap(tmp_path,variant="duplicate-channels")
    client = ScriptedClient([])
    result = AgentRunner(tmp_path,model=client).run("收入")
    assert result["stop_reason"] == "source_integrity_failed" and result["model_requests"] == 0


def test_A22_free_text_cannot_bypass_tools(tmp_path):
    bootstrap(tmp_path)
    class TextOnly:
        def respond(self,history,tools):
            return {"protocol_items":[],"tool_calls":[],"text":"已经计算完毕，现金净额很好。","usage":{},"status":"completed"}
    result = AgentRunner(tmp_path,model=TextOnly()).run("分析")
    assert result["status"].lower() == "failed" and not result["artifacts"] and not result.get("report_path")


def _text_artifact(run_dir):
    folder = run_dir/"artifacts"
    folder.mkdir(parents=True)
    path = folder/"input.parquet"
    pq.write_table(pa.Table.from_pylist([{"ticket_id":"T1","text":"第一行\n第二行"}]),path)
    return {"artifact_id":"input", "path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


class SemanticClient:
    def __init__(self,start=1,end=2,evidence="ticket:T1"):
        self.start,self.end,self.evidence = start,end,evidence
    def semantic(self,texts):
        return {"items":[{"ticket_id":"T1","label":"不确定","evidence_id":self.evidence,"start_line":self.start,"end_line":self.end}],"usage":{},"prompt_version":"test-v1","model":"injected-semantic-test"}


def test_A23_semantic_materialization_preserves_evidence(tmp_path):
    artifact = _text_artifact(tmp_path)
    step = {"step_id":"labels","output_name":"labels","operator":"classify_tickets"}
    result = run_operator(step,{"tickets":artifact},tmp_path,SemanticClient(),"test")
    row = pq.read_table(result["path"]).to_pylist()[0]
    assert row["evidence_text"] == "第一行\n第二行" and row["needs_review"]
    assert result["lineage"]["internal_model_lineage"] == "unknown"


@pytest.mark.parametrize("client",[SemanticClient(end=3),SemanticClient(start=0),SemanticClient(start=True),SemanticClient(evidence="fabricated")])
def test_A24_semantic_false_evidence_rejected(tmp_path,client):
    artifact = _text_artifact(tmp_path)
    step = {"step_id":"labels","output_name":"labels","operator":"classify_tickets"}
    with pytest.raises(ValueError,match="evidence"):
        run_operator(step,{"tickets":artifact},tmp_path,client,"test")


def test_A25_duplicate_call_returns_committed_result(tmp_path):
    from local_data_agent.state import StateStore
    store = StateStore(tmp_path)
    state = store.create("test", "replay", "cash-net-v1", "cash-net-v1")
    args = {"query":"schema"}
    assert store.start_call(state["run_id"], "call-1", "search_context", args) is None
    store.finish_call(state["run_id"], "call-1", {"observed":"committed"})
    assert store.start_call(state["run_id"], "call-1", "search_context", args) == {"observed":"committed"}
    with pytest.raises(ValueError,match="different arguments"):
        store.start_call(state["run_id"], "call-1", "search_context", {"query":"changed"})


def test_A26_same_run_cannot_have_two_active_attempts(tmp_path):
    from local_data_agent.state import StateStore
    store = StateStore(tmp_path)
    state = store.create("test", "replay", "cash-net-v1", "cash-net-v1")
    with store.acquire(state["run_id"]):
        with pytest.raises(RuntimeError,match="already"):
            with StateStore(tmp_path).acquire(state["run_id"]):
                pytest.fail("Concurrent acquisition must fail")


def test_A27_tampered_result_rejected_before_followup(base):
    artifact = base[3]["artifacts"]["monthly_total"]
    path = Path(artifact["path"])
    original = path.read_bytes()
    try:
        path.write_bytes(original+b"tamper")
        with pytest.raises(ValueError,match="hash|changed|integrity|mismatch"):
            base[2].resume(base[3]["run_id"],question="只看官网")
    finally:
        path.write_bytes(original)


def test_A28_missing_metric_explicit_human_approval_then_resume(tmp_path):
    from local_data_agent.tools import TOOL_DEFINITIONS
    registry = bootstrap(tmp_path,variant="no-metric")
    runner = AgentRunner(tmp_path)
    paused = runner.run("收入口径尚未批准",case="clarification")
    assert paused["status"].lower() == "waiting_input"
    prior = checkpoint(paused)
    assert prior["metric_binding"] is None and prior["manifest"]["metric_contracts"] == []
    source_hashes = {s["source_id"]:s["sha256"] for s in prior["manifest"]["sources"]}
    assert "approve_metric" not in {d["name"] for d in TOOL_DEFINITIONS}
    with pytest.raises(ValueError,match="human"):
        runner.approve_metric(paused["run_id"],"CASH_NET_v1","model")
    approved = runner.approve_metric(paused["run_id"],"CASH_NET_v1","test-human")
    assert approved["metric_contract"]["approved_by"] == "test-human"
    approved_state = checkpoint(paused)
    assert approved_state["metric_binding"] is None
    assert source_hashes == {s["source_id"]:s["sha256"] for s in approved_state["manifest"]["sources"]}
    assert any(i.get("role") == "user" and "test-human" in i.get("content", "") for i in approved_state["history"])
    assert (Path(paused["run_dir"])/"metric_approval.json").exists()
    completed = AgentRunner(tmp_path).resume(paused["run_id"],answer="我批准 CASH_NET_v1，按照这个口径继续")
    assert completed["status"].lower() == "succeeded", completed
    state = checkpoint(completed)
    assert state["bound_contract"]["approved_by"] == "test-human" and state["bound_contract"]["status"] == "APPROVED"
    assert evaluate_gold(completed["artifacts"],compute_gold(registry))["passed"]


def test_A29_followup_publication_keeps_prior_report_immutable(tmp_path):
    bootstrap(tmp_path)
    replay = ReplayModelClient()
    actions = copy.deepcopy(replay.actions)
    actions.extend([
        ("inspect_artifact",{"artifact_id":"monthly_channels","view":"result"}),
        ("finalize_run",{"plan_version":2,"claims_json":json.dumps([{"claim_id":"web_followup","artifact_id":"monthly_channels","column":"net_cents","filter":{"month":"2026-09","channel_code":"web"},"aggregation":"single_row","display_unit":"CNY_yuan","label":"官网现金净额"}],ensure_ascii=False),"summary":"事实：官网现金净额按所选指标汇总；此前总量报告已保留。","open_questions":[]}),
    ])
    runner = AgentRunner(tmp_path,model=ScriptedClient(actions))
    initial = runner.run("按现金净额分析",mode="live")
    assert initial["status"].lower() == "succeeded",initial
    initial_path = Path(initial["report_path"])
    initial_content = initial_path.read_bytes()
    old_ids = {name:a["artifact_id"] for name,a in initial["artifacts"].items()}
    followup = runner.resume(initial["run_id"],question="只看官网")
    assert followup["status"].lower() == "succeeded",followup
    assert followup["run_id"] == initial["run_id"]
    assert Path(followup["report_path"]) != initial_path
    assert initial_path.read_bytes() == initial_content
    assert len(followup["publications"]) == 2
    assert old_ids == {name:a["artifact_id"] for name,a in followup["artifacts"].items()}
    assert "官网现金净额" in followup["answer"] and "130" in followup["answer"]
