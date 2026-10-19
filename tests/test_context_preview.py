"""Bounded execute_step artifact preview in the provider view only.

Synthetic records only: no real run directory, history or gold data is read.
The preview is a SAMPLE, so totals/counts are never derived from it and the
persisted rows on disk are never truncated.
"""
import copy
import json

import pytest

from local_data_agent.context_budget import (
    DEFAULT_PREVIEW_LIMIT,
    estimate_tokens,
    prepare_history,
)


def execute_step_pair(index, payload, name="execute_step"):
    return [
        {"type": "function_call", "call_id": f"call_{index}", "name": name,
         "arguments": json.dumps({"plan_version": 1, "step_id": "net"})},
        {"type": "function_call_output", "call_id": f"call_{index}", "output": json.dumps(payload)},
    ]


def read_output(history, call_id):
    item = next(item for item in history
                if item.get("type") == "function_call_output" and item["call_id"] == call_id)
    return json.loads(item["output"])


def artifact_payload(rows=50, row_count=None):
    preview = [{"month": f"2024-{row % 12 + 1:02d}", "net_cents": 1000 + row} for row in range(rows)]
    return {
        "artifact": {
            "artifact_id": "artifact_net",
            "output_name": "net",
            "row_count": rows if row_count is None else row_count,
            "schema": [{"name": "month", "type": "VARCHAR"}, {"name": "net_cents", "type": "BIGINT"}],
            "preview": preview,
        },
        "executed_outputs": ["net"],
        "mode": "offline",
    }


def artifact_state(rows=50, name="net"):
    payload = artifact_payload(rows)["artifact"]
    return {"artifacts": {name: {**payload, "path": "local-only", "sha256": "abc"}}}


def history_for(payload):
    return [{"role": "system", "content": "agent"},
            {"role": "user", "content": "net revenue?"},
            *execute_step_pair(0, payload)]


def archival_history(padding=2000):
    history = [{"role": "system", "content": "agent"}, {"role": "user", "content": "old question"}]
    for index in range(5):
        history.extend([
            {"type": "reasoning", "id": f"reason_{index}", "summary": [{"type": "summary_text", "text": f"e{index}"}]},
            *execute_step_pair(index, {"note": "q" * padding}),
        ])
    return history


def archival_summary(prepared):
    texts = [item["content"].split("：", 1)[1] for item in prepared
             if item.get("role") == "user" and item.get("content", "").startswith("持久化状态摘要：")]
    assert texts, "archival summary expected"
    return json.loads(texts[0])


def test_original_history_and_state_are_never_mutated():
    payload = artifact_payload()
    history = history_for(payload)
    state = {"question": "net?", "bound_contract": {"metric_id": "NET_v1"}, "plan_version": 1,
             "plan": [{"step_id": "net", "type": "SELECT"}], "observed_facts": [],
             "open_questions": [], "evidence": [], **artifact_state()}
    original_history = copy.deepcopy(history)
    original_payload = copy.deepcopy(payload)
    original_state = copy.deepcopy(state)

    prepared = prepare_history(history, maximum=1500, state=state, preview_limit=4)

    assert history == original_history
    assert payload == original_payload          # full 50 rows still in the input
    assert state == original_state              # persisted artifact untouched
    assert len(read_output(history, "call_0")["artifact"]["preview"]) == 50


def test_execute_step_preview_is_capped_and_flagged_with_the_full_row_count():
    prepared = prepare_history(history_for(artifact_payload(rows=50)), maximum=10 ** 9)
    artifact = read_output(prepared, "call_0")["artifact"]

    assert len(artifact["preview"]) == DEFAULT_PREVIEW_LIMIT
    assert artifact["preview_rows_shown"] == DEFAULT_PREVIEW_LIMIT
    assert artifact["preview_truncated"] is True
    # The metadata identity and the true total are preserved, not the sample size.
    assert artifact["artifact_id"] == "artifact_net"
    assert artifact["row_count"] == 50
    assert artifact["row_count"] != artifact["preview_rows_shown"]
    assert artifact["schema"] == [{"name": "month", "type": "VARCHAR"}, {"name": "net_cents", "type": "BIGINT"}]
    # The retained sample rows are the leading rows, unchanged.
    assert artifact["preview"][0] == {"month": "2024-01", "net_cents": 1000}
    assert artifact["preview"][-1] == {"month": "2024-08", "net_cents": 1007}
    assert read_output(prepared, "call_0")["executed_outputs"] == ["net"]
    assert read_output(prepared, "call_0")["mode"] == "offline"


def test_unknown_execute_step_artifact_payload_passes_unchanged():
    # Missing row_count and schema: not the registered metadata shape.
    payload = {"artifact": {"artifact_id": "unknown",
                            "preview": [{"i": index} for index in range(50)]}}
    original = copy.deepcopy(payload)
    prepared = prepare_history(history_for(payload), maximum=10 ** 9, preview_limit=4)

    # The unknown payload is not mistaken for artifact metadata: no cap, no
    # injected preview_rows_shown/preview_truncated keys.
    assert read_output(prepared, "call_0") == original
    assert len(read_output(prepared, "call_0")["artifact"]["preview"]) == 50
    assert "preview_rows_shown" not in read_output(prepared, "call_0")["artifact"]
    assert "preview_truncated" not in read_output(prepared, "call_0")["artifact"]
    assert payload == original


def test_configurable_limit_and_truthful_small_or_empty_previews():
    small = prepare_history(history_for(artifact_payload(rows=3, row_count=3)), maximum=10 ** 9, preview_limit=8)
    artifact = read_output(small, "call_0")["artifact"]
    assert len(artifact["preview"]) == 3
    assert artifact["preview_rows_shown"] == 3
    assert artifact["preview_truncated"] is False          # 3 rows total, 3 shown

    empty = prepare_history(history_for(artifact_payload(rows=0, row_count=0)), maximum=10 ** 9, preview_limit=8)
    empty_artifact = read_output(empty, "call_0")["artifact"]
    assert empty_artifact["preview"] == []
    assert (empty_artifact["preview_rows_shown"], empty_artifact["preview_truncated"]) == (0, False)
    assert empty_artifact["row_count"] == 0

    exact = prepare_history(history_for(artifact_payload(rows=50)), maximum=10 ** 9, preview_limit=50)
    exact_artifact = read_output(exact, "call_0")["artifact"]
    assert len(exact_artifact["preview"]) == 50
    assert exact_artifact["preview_truncated"] is False

    with pytest.raises(ValueError):
        prepare_history(history_for(artifact_payload()), maximum=10 ** 9, preview_limit=0)
    with pytest.raises(ValueError):
        prepare_history(history_for(artifact_payload()), maximum=10 ** 9, preview_limit=51)


def test_explicit_inspection_rows_and_failed_counterexamples_are_retained():
    inspection = {"result": [{"i": index} for index in range(50)]}
    failed_checks = {"checks": [{"check_id": "c9", "passed": False, "message": "mismatch",
                                 "details": {"counterexamples": [{"group": "2024-01", "error": "mismatch"}]}}]}
    history = [{"role": "system", "content": "agent"},
               *execute_step_pair(0, inspection, name="inspect_artifact"),
               *execute_step_pair(1, failed_checks, name="check_result")]
    prepared = prepare_history(history, maximum=10 ** 9)

    assert read_output(prepared, "call_0") == inspection
    assert len(read_output(prepared, "call_0")["result"]) == 50
    assert read_output(prepared, "call_1")["checks"][0]["details"]["counterexamples"] == [
        {"group": "2024-01", "error": "mismatch"}]


def test_archival_snapshot_caps_artifact_previews_and_keeps_plan_and_evidence():
    state = {"question": "net?", "bound_contract": {"metric_id": "NET_v1", "version": 2},
             "plan_version": 4,
             "plan": [{"step_id": "net", "type": "SELECT", "output_name": "net",
                       "depends_on": ["events"], "sql": "SELECT month,SUM(c) FROM events GROUP BY month"}],
             "observed_facts": [{"artifact_id": "artifact_net", "description": "inspected"}],
             "open_questions": ["refunds?"],
             "evidence": [{"query": "metric", "evidence": [
                 {"evidence_id": "doc:approved", "doc_hash": "hash-1", "start_line": 2, "end_line": 5,
                  "text": "approved definition"}]}],
             **artifact_state(rows=50, name="net")}
    original_state = copy.deepcopy(state)
    history = archival_history()
    maximum = 2000

    # The fixture genuinely exceeds the chosen budget, so the archival branch is
    # exercised; a bounded snapshot of it fits.
    assert estimate_tokens(history) > maximum

    prepared = prepare_history(history, maximum=maximum, state=state)
    summary = archival_summary(prepared)

    assert summary["plan_version"] == 4
    assert summary["plan"] == state["plan"]
    assert summary["metric_contract"] == {"metric_id": "NET_v1", "version": 2}
    assert summary["question"] == "net?" and summary["open_questions"] == ["refunds?"]
    refs = summary["context_refs"]
    assert refs[0]["evidence_id"] == "doc:approved" and refs[0]["doc_hash"] == "hash-1"
    assert refs[0]["text"] == "approved definition"
    assert (refs[0]["start_line"], refs[0]["end_line"]) == (2, 5)
    assert summary["observations"] == [{"artifact_id": "artifact_net", "description": "inspected"}]

    entry = summary["artifacts"]["net"]
    assert entry["artifact_id"] == "artifact_net" and entry["row_count"] == 50
    assert len(entry["preview"]) == DEFAULT_PREVIEW_LIMIT
    assert entry["preview_rows_shown"] == DEFAULT_PREVIEW_LIMIT
    assert entry["preview_truncated"] is True
    assert entry["schema"][0]["name"] == "month"
    assert entry["preview"][0]["net_cents"] == 1000
    assert state == original_state
    # The prepared snapshot is bounded by the same unchanged budget.
    assert estimate_tokens(prepared) <= maximum
