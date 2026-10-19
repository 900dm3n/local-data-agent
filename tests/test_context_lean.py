"""Lean provider projection: redundant result-view keys only, data is never pruned."""
import copy
import json

import pytest

from local_data_agent.context_budget import check_history, estimate_tokens, prepare_history


def responses_pair(index, name, payload):
    return [
        {"type": "function_call", "call_id": f"call_{index}", "name": name, "arguments": json.dumps({"i": index})},
        {"type": "function_call_output", "call_id": f"call_{index}", "output": json.dumps(payload)},
    ]


def chat_pair(index, name, payload):
    argument = json.dumps({"i": index})
    return [
        {"type": "chat_message", "message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"chat_{index}", "type": "function", "function": {"name": name, "arguments": argument}}]}},
        {"type": "function_call_output", "call_id": f"chat_{index}", "output": json.dumps(payload)},
    ]


def read_output(history, call_id):
    item = next(item for item in history if item.get("type") == "function_call_output" and item["call_id"] == call_id)
    return json.loads(item["output"])


def search_payload():
    return {
        "query": "revenue metric definition",
        "paths_exposed": ["docs/a.md", "docs/b.md"],
        "retrieval_mode": "hybrid",
        "missing_context": ["cash refunds"],
        "truncated": True,
        "metric_contracts": [{"metric_id": "NET_v1", "status": "approved", "version": 2,
                              "timezone": "Australia/Sydney", "unit": "cents"}],
        "evidence": [
            {"evidence_id": "doc:1", "doc_hash": "abc123", "start_line": 4, "end_line": 9,
             "type": "document", "source_id": "docs/a.md", "text": "net revenue excludes tax"},
            {"evidence_id": "doc:2", "doc_hash": "def456", "start_line": 1, "end_line": 3,
             "type": "schema", "source_id": "table:events", "table": "events",
             "schema": [{"name": "net_cents", "type": "INTEGER"}], "text": "materialized schema summary"},
            {"evidence_id": "doc:3", "type": "schema", "source_id": "table:orders", "text": "unresolved schema text"},
        ],
    }


def check_payload():
    return {"checks": [
        {"check_id": "c1", "passed": True, "details": {"counterexample": "row 7"}, "sql": "SELECT 1",
         "message": "row matches reconciled total", "evidence": {"row_hash": "sha9"},
         "error": None, "explanation": "totals agree", "detail": "summary", "diagnostics": ["ok"]},
        {"check_id": "c2", "passed": False, "details": {"row": 42, "reason": "negative net"}, "message": "failed"},
    ]}


def original_evidence(history, index, evidence_id):
    """The persisted evidence record, parsed from the mutating input for comparison."""
    payload = json.loads(history[index]["output"])
    return next(record for record in payload["evidence"] if record["evidence_id"] == evidence_id)


def test_original_history_is_never_mutated():
    history = [
        {"role": "system", "content": "agent"},
        {"role": "user", "content": "net revenue?"},
        *responses_pair(0, "search_context", search_payload()),
        *responses_pair(1, "check_result", check_payload()),
    ]
    original = copy.deepcopy(history)
    prepare_history(history, maximum=10)
    assert history == original


def test_document_text_survives_but_only_registered_schema_text_is_removed():
    history = [{"role": "system", "content": "agent"}, *responses_pair(0, "search_context", search_payload())]
    prepared = prepare_history(history, maximum=10)
    payload = read_output(prepared, "call_0")
    assert not {"query", "paths_exposed", "retrieval_mode"}.intersection(payload)
    assert payload["missing_context"] == ["cash refunds"]
    assert payload["truncated"] is True
    document, schema_record, unresolved = payload["evidence"]
    # Compare against the parsed original evidence record, not the whole output JSON string.
    assert document == original_evidence(history, 2, "doc:1")
    assert document["text"] == "net revenue excludes tax"
    assert document["evidence_id"] == "doc:1" and document["doc_hash"] == "abc123"
    assert (document["start_line"], document["end_line"]) == (4, 9)
    assert "text" not in schema_record and schema_record["schema"] == [{"name": "net_cents", "type": "INTEGER"}]
    assert schema_record["table"] == "events" and schema_record["source_id"] == "table:events"
    assert (schema_record["start_line"], schema_record["end_line"]) == (1, 3)
    assert schema_record["evidence_id"] == "doc:2" and schema_record["doc_hash"] == "def456"
    # No explicit schema list -> not an enriched record -> text retained.
    assert unresolved == original_evidence(history, 2, "doc:3")
    assert unresolved["text"] == "unresolved schema text"
    assert payload["metric_contracts"][0]["timezone"] == "Australia/Sydney"


def test_unknown_schema_types_and_types_are_never_stripped():
    """Only the registered `schema` type may lose text; invented types are unknown payloads."""
    variants = [
        {"evidence_id": "doc:4", "type": "synthesized_schema", "source_id": "table:events", "table": "events",
         "schema": [{"name": "net_cents"}], "text": "synthesized schema summary"},
        {"evidence_id": "doc:5", "type": "inferred_schema", "source_id": "table:events", "table": "events",
         "schema": [{"name": "net_cents"}], "text": "inferred schema summary"},
        {"evidence_id": "doc:6", "type": "schema_v2", "source_id": "table:events", "table": "events",
         "schema": [{"name": "net_cents"}], "text": "future schema summary"},
        {"evidence_id": "doc:7", "source_id": "table:events", "table": "events",
         "schema": [{"name": "net_cents"}], "text": "untyped schema summary"},
    ]
    payload = search_payload()
    payload["evidence"] = variants + [
        {"evidence_id": "doc:8", "type": "schema", "source_id": "table:events", "table": "events",
         "schema": [], "text": "empty schema list keeps text"},
        {"evidence_id": "doc:9", "type": "schema", "source_id": "  ", "table": "events",
         "schema": [{"name": "net_cents"}], "text": "blank source keeps text"},
        {"evidence_id": "doc:10", "type": "schema", "source_id": "table:events", "table": "",
         "schema": [{"name": "net_cents"}], "text": "blank table keeps text"},
        {"evidence_id": "doc:11", "type": "schema", "source_id": "table:events", "table": "events",
         "schema": "net_cents", "text": "non-list schema keeps text"},
    ]
    history = [{"role": "system", "content": "agent"}, *responses_pair(0, "search_context", payload)]
    prepared = prepare_history(history, maximum=10)
    projected = read_output(prepared, "call_0")["evidence"]
    assert len(projected) == len(payload["evidence"])
    for record in projected:
        # Every malformed or unknown-type record is returned unchanged, `text` included.
        assert "text" in record, record["evidence_id"]
        assert record == next(original for original in payload["evidence"]
                              if original["evidence_id"] == record["evidence_id"])


def test_failed_checks_keep_details_while_successful_checks_drop_only_details():
    history = [{"role": "system", "content": "agent"}, *responses_pair(0, "check_result", check_payload())]
    prepared = prepare_history(history, maximum=10)
    checks = read_output(prepared, "call_0")["checks"]
    passed, failed = checks
    assert passed["check_id"] == "c1" and passed["passed"] is True
    assert "details" not in passed
    assert passed["sql"] == "SELECT 1"
    # Only `details` is in scope for removal; every other field is retained.
    assert passed["message"] == "row matches reconciled total"
    assert passed["evidence"] == {"row_hash": "sha9"}
    assert "error" in passed and passed["error"] is None
    assert passed["explanation"] == "totals agree"
    assert passed["detail"] == "summary"
    assert passed["diagnostics"] == ["ok"]
    assert set(passed) == {"check_id", "passed", "sql", "message", "evidence", "error",
                           "explanation", "detail", "diagnostics"}
    assert failed["passed"] is False
    assert failed["details"] == {"row": 42, "reason": "negative net"}
    assert failed["message"] == "failed"
    assert failed == check_payload()["checks"][1]


def test_arbitrary_nested_data_is_not_pruned():
    payload = {"result": {"text": "keep me", "details": {"text": "deep"}, "items": [{"text": "nested", "details": {"x": 1}}]}}
    other = {"query": "keep", "retrieval_mode": "keep", "text": "keep", "details": "keep"}
    history = [{"role": "system", "content": "agent"},
               *responses_pair(0, "inspect_artifact", payload),
               *responses_pair(1, "unknown_tool", other),
               {"type": "function_call_output", "call_id": "dangling", "output": json.dumps({"query": "keep"})}]
    prepared = prepare_history(history, maximum=10)
    assert read_output(prepared, "call_0") == payload
    assert read_output(prepared, "call_1") == other
    assert read_output(prepared, "dangling") == {"query": "keep"}


def test_projection_of_old_items_is_stable_when_exchanges_are_appended():
    prefix = [{"role": "system", "content": "agent"}, {"role": "user", "content": "question"},
              *responses_pair(0, "search_context", search_payload()),
              *responses_pair(1, "check_result", check_payload())]
    extended = copy.deepcopy(prefix) + responses_pair(2, "search_context", search_payload())
    short = prepare_history(prefix, maximum=10)
    long = prepare_history(extended, maximum=10)
    assert short == long[:len(short)]


def test_chat_and_responses_call_mapping_use_the_same_projection():
    chat = [{"role": "system", "content": "agent"}, *chat_pair(0, "search_context", search_payload())]
    responses = [{"role": "system", "content": "agent"}, *responses_pair(0, "search_context", search_payload())]
    chat_payload = read_output(prepare_history(chat, maximum=10), "chat_0")
    responses_payload = read_output(prepare_history(responses, maximum=10), "call_0")
    assert chat_payload == responses_payload
    assert "paths_exposed" not in chat_payload
    assert chat[1]["message"]["tool_calls"][0]["function"]["arguments"] == '{"i": 0}'


def test_pending_chat_call_keeps_requests_outputs_and_budget_stop_is_explicit():
    """One Chat assistant with p_a/p_b calls, one output for p_a; p_b stays pending."""
    calls = [{"id": "p_a", "type": "function", "function": {"name": "search_context", "arguments": "{}"}},
             {"id": "p_b", "type": "function", "function": {"name": "search_context", "arguments": "{}"}}]
    history = [{"role": "system", "content": "agent"},
               {"type": "chat_message", "message": {"role": "assistant", "content": None, "tool_calls": calls}},
               {"type": "function_call_output", "call_id": "p_a", "output": json.dumps(search_payload())}]
    original = copy.deepcopy(history)
    prepared = prepare_history(history, maximum=10)

    chat_messages = [item for item in prepared if item.get("type") == "chat_message"]
    requests = [call["id"] for item in chat_messages for call in item["message"].get("tool_calls", [])]
    outputs = [item["call_id"] for item in prepared if item.get("type") == "function_call_output"]

    assert len(chat_messages) == 1
    assert sorted(requests) == ["p_a", "p_b"]              # both unique calls retained
    assert len(requests) == len(set(requests)) == 2
    assert outputs == ["p_a"]                               # exactly the one real output
    assert prepared[1] == original[1]                       # request item byte-identical
    assert [item["call_id"] for item in prepared if item.get("type") == "function_call_output"] == ["p_a"]
    assert not [item for item in prepared if item.get("type") == "function_call"]
    # The output that exists is still lean-projected; document evidence is verbatim.
    payload = read_output(prepared, "p_a")
    assert "query" not in payload
    assert payload["evidence"][0] == original_evidence(history, 2, "doc:1")
    assert payload["evidence"][0]["text"] == "net revenue excludes tax"

    # Budget exhaustion must be explicit, not a silent drop of the pending call.
    with pytest.raises(RuntimeError, match="context_budget_exceeded"):
        check_history(prepared, maximum=10)
    generous = prepare_history(history, maximum=10 ** 9)
    assert generous[1]["message"]["tool_calls"] == calls
    assert [item["call_id"] for item in generous if item.get("type") == "function_call_output"] == ["p_a"]


@pytest.mark.parametrize("builder", [responses_pair, chat_pair])
def test_representative_output_is_smaller_without_losing_protocol(builder):
    history = [{"role": "system", "content": "agent"},
               *builder(0, "search_context", search_payload()),
               *builder(1, "check_result", check_payload())]
    before = estimate_tokens(history)
    prepared = prepare_history(history, maximum=10 ** 9)
    after = estimate_tokens(prepared)
    assert after < before
    assert len(prepared) == len(history)
    assert [item.get("call_id") for item in prepared] == [item.get("call_id") for item in history]
