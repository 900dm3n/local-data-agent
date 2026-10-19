import json
import sqlite3
import pytest
from local_data_agent.evaluation import deterministic_cases, evaluate, _manifest
from local_data_agent.fixtures import bootstrap, compute_gold
from local_data_agent.retrieval import ContextRetriever
from local_data_agent.validation import validate_sources, check_artifacts
from local_data_agent.memory import MemoryStore


def test_independent_knowledge_matrix(tmp_path):
    results = deterministic_cases(tmp_path)
    assert len(results) == 32
    assert all(r["status"] == "passed" for r in results), [r for r in results if r["status"] != "passed"]


def test_live_missing_configuration_not_replay(tmp_path, monkeypatch):
    for name in ("DATA_AGENT_MODEL", "DATA_AGENT_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    report = evaluate(tmp_path, "agent-live")
    assert report["status"] == "unverified"
    assert not report["model_intelligence_verified"]
    assert report["cases"] == []


def test_schema_retrieval_filtered_and_paths_not_exposed(tmp_path):
    registry = bootstrap(tmp_path)
    manifest = _manifest(registry)
    retrieval = ContextRetriever(tmp_path/"run", manifest)
    result = retrieval.search("退款", source_ids=["business"], types=["schema"])
    assert any(r["table"] == "refunds" for r in result["evidence"])
    assert all(r["type"] == "schema" and r["source_id"] == "business" for r in result["evidence"])
    assert not any("path" in row for row in result["evidence"])
    retrieval.close()


def test_snapshot_hash_check_and_timestamp_naive_rejection(tmp_path):
    import hashlib
    manifest = _manifest(bootstrap(tmp_path))
    business = manifest["sources"][0]
    business["sha256"] = hashlib.sha256(open(business["path"], "rb").read()).hexdigest()
    with sqlite3.connect(business["path"]) as db:
        db.execute("UPDATE orders SET paid_at='2026-08-05T12:00:00' WHERE order_id='O1'")
    result = validate_sources(manifest)
    failures = {r["name"] for r in result["checks"] if not r["passed"]}
    assert "aware_event_timestamps" in failures
    assert "snapshot_hash:business" in failures


def test_memory_conflicts_kept_separate(tmp_path):
    store = MemoryStore(tmp_path)
    scope = {"metric":"cash", "version":1}
    for month in ("payment", "refund"):
        card = store.draft("test", {"note": f"month {month}"}, scope)
        store.approve(card["memory_id"], "owner")
    cards = store.recall("month", scope)
    assert len(cards) == 2
    assert cards[0]["memory_id"] != cards[1]["memory_id"]


def test_semantic_evidence_valid_rows(tmp_path):
    rows = [{"evidence_id":"T1","start_line":1,"end_line":1,"line_count":1}]
    assert check_artifacts({"tickets":{"rows":rows}}, ["semantic_evidence"])["passed"]


@pytest.mark.parametrize("variant", ["orphan-refund", "null-amount", "negative-amount", "duplicate-order"])
def test_gold_does_not_silently_compute_invalid_inputs(tmp_path, variant):
    registry = bootstrap(tmp_path, variant=variant)
    with pytest.raises(ValueError):
        compute_gold(registry)


def test_unknown_fixture_variant_does_not_write(tmp_path):
    with pytest.raises(ValueError):
        bootstrap(tmp_path, variant="unapproved")
    assert not (tmp_path/"var"/"sources"/"cash-net-v1").exists()


@pytest.mark.parametrize("missing_month", [False, True])
def test_live_evaluator_uppercase_and_partial_totals(tmp_path,monkeypatch,missing_month):
    """No API: an injected runner exercises evaluation result handling only."""
    import local_data_agent.model_client as model_module
    import local_data_agent.runner as runner_module
    bootstrap(tmp_path)
    monkeypatch.setattr(model_module,"load_settings",lambda root: {"model":"injected-test-only","api_mode":"chat"})
    rows = [{"month":"2026-08","net_cents":38000}]
    if not missing_month:
        rows.append({"month":"2026-09","net_cents":31000})
    class FakeRunner:
        def __init__(self,root):
            self.root = root
        def run(self,question,mode,fixture):
            assert mode == "live" and "38000" not in question and "SELECT" not in question
            return {"run_id":"injected-evaluator-test","status":"SUCCEEDED","run_dir":str(tmp_path),"artifacts":{"monthly":{"rows":rows}}}
    monkeypatch.setattr(runner_module,"AgentRunner",FakeRunner)
    report = evaluate(tmp_path,"agent-live",cases=["L01"])
    assert len(report["cases"]) == 3
    assert len(report["unexecuted_case_ids"]) == 23
    assert all(c["status"] == ("failed" if missing_month else "passed") for c in report["cases"])
    assert not report["model_intelligence_verified"]
