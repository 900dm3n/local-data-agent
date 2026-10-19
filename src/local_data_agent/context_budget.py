"""Conservative approximate budget; never split tool calls or their results."""

import copy
import json

from .artifacts import dumps


def estimate_tokens(value):
    # Not a model tokenizer. A deliberately conservative bound is explicit in logs.
    return len(dumps(value).encode("utf-8")) // 2 + 1


_RESULT_DROP_KEYS = ("query", "paths_exposed", "retrieval_mode")
# Only the actual registered evidence type is recognized. Invented variants such as
# "synthesized_schema" or "inferred_schema" are unknown payloads and pass unchanged.
_SCHEMA_EVIDENCE_TYPE = "schema"

# Bounded provider preview for `execute_step` artifact metadata only. This is a
# SAMPLE of the persisted result, never a complete result: totals or counts must
# never be derived from it. The full rows stay on disk/artifacts and in the
# transcript, reachable through inspect_artifact.
DEFAULT_PREVIEW_LIMIT = 8
MAX_PREVIEW_LIMIT = 50
# The registered metadata shape emitted by tools.execute_step -> visible()/artifact().
_ARTIFACT_METADATA_KEYS = ("artifact_id", "row_count", "schema", "preview")


def _resolve_preview_limit(preview_limit):
    """Positive int, capped at the existing provider preview size."""
    if isinstance(preview_limit, bool) or not isinstance(preview_limit, int):
        raise ValueError("preview_limit must be a positive int no greater than 50")
    if preview_limit < 1 or preview_limit > MAX_PREVIEW_LIMIT:
        raise ValueError("preview_limit must be a positive int no greater than 50")
    return preview_limit


def _is_artifact_metadata(value):
    """True only for the known artifact metadata object carrying ALL registered keys.

    The registered shape is exactly what tools.execute_step emits: every key in
    `_ARTIFACT_METADATA_KEYS` must be present, `artifact_id` must be a nonempty
    string, `row_count` must be a nonnegative int (bool excluded), and both
    `schema` and `preview` must be lists. Partial or invented objects (documents,
    check results, inspect_artifact views, arbitrary records) are unknown payloads
    and pass unchanged.
    """
    if not isinstance(value, dict):
        return False
    if not all(key in value for key in _ARTIFACT_METADATA_KEYS):
        return False
    if not _nonempty_text(value.get("artifact_id")):
        return False
    row_count = value.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        return False
    return isinstance(value.get("schema"), list) and isinstance(value.get("preview"), list)


def _cap_artifact_preview(value, preview_limit):
    """Cap only `preview`; every other metadata field is retained verbatim."""
    preview = value.get("preview")
    if not isinstance(preview, list):
        return value
    shown = preview[:preview_limit]
    output = dict(value)
    output["preview"] = [dict(row) if isinstance(row, dict) else row for row in shown]
    output["preview_rows_shown"] = len(shown)
    row_count = value.get("row_count")
    # `preview_truncated` is only a truthful boolean when the full row count is known.
    if isinstance(row_count, int) and not isinstance(row_count, bool):
        output["preview_truncated"] = row_count > len(shown)
    else:
        output["preview_truncated"] = None
    return output


def _cap_execute_step_result(payload, preview_limit):
    """Cap the known `artifact` metadata object of an execute_step result.

    `executed_outputs`/`mode` and any other field are retained unchanged, and the
    projection is applied only where the registered metadata shape is present.
    """
    if not isinstance(payload, dict):
        return payload
    artifact = payload.get("artifact")
    if not _is_artifact_metadata(artifact):
        return payload
    output = dict(payload)
    output["artifact"] = _cap_artifact_preview(artifact, preview_limit)
    return output


def _project_artifact_metadata(value, preview_limit):
    """Apply the same bounded preview projection to snapshot artifact entries."""
    if not _is_artifact_metadata(value):
        return value
    return _cap_artifact_preview(value, preview_limit)


def _tool_names_by_call_id(history):
    """Map every call_id to its tool name from Responses calls and Chat tool_calls."""
    names = {}
    for item in history:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call" and "call_id" in item:
            names[item["call_id"]] = item.get("name")
        if item.get("type") == "chat_message":
            message = item.get("message")
            calls = message.get("tool_calls") if isinstance(message, dict) else None
            if isinstance(calls, list):
                for call in calls:
                    if not isinstance(call, dict) or "id" not in call:
                        continue
                    function = call.get("function")
                    names[call["id"]] = function.get("name") if isinstance(function, dict) else None
    return names


def _lean_search_context(payload):
    """Drop only redundant result-view keys; evidence, hashes and text are data."""
    result = {key: value for key, value in payload.items() if key not in _RESULT_DROP_KEYS}
    evidence = payload.get("evidence")
    if isinstance(evidence, list):
        result["evidence"] = [_lean_evidence(record) for record in evidence]
    return result


def _lean_evidence(record):
    """Drop `text` only for a registered `schema` record that already carries the data.

    Unknown evidence types, and records without an explicit nonempty schema list plus
    source/table identifiers, are unknown payloads: they are returned unchanged.
    """
    if not isinstance(record, dict):
        return record
    if record.get("type") != _SCHEMA_EVIDENCE_TYPE:
        return record
    schema = record.get("schema")
    if not isinstance(schema, list) or not schema:
        return record
    if not _nonempty_text(record.get("source_id")) or not _nonempty_text(record.get("table")):
        return record
    return {key: value for key, value in record.items() if key != "text"}


def _nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def _lean_check_result(payload):
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return payload
    output = dict(payload)
    output["checks"] = [dict(check) if isinstance(check, dict) else check for check in checks]
    for check in output["checks"]:
        if not isinstance(check, dict) or check.get("passed") is not True:
            continue
        # ONLY the redundant `details` view is removed, and only for a passed check.
        # evidence/message/error/explanation/detail/diagnostics are outside task scope
        # and may hold meaningful context, so they are always retained.
        check.pop("details", None)
    return output


def _lean_result(tool_name, payload, preview_limit):
    if not isinstance(payload, dict):
        return payload
    if tool_name == "search_context" and isinstance(payload.get("evidence"), list) and "metric_contracts" in payload:
        return _lean_search_context(payload)
    if tool_name == "check_result" and isinstance(payload.get("checks"), list):
        return _lean_check_result(payload)
    # Only execute_step emits the artifact metadata shape; explicit inspect_artifact
    # result views are a different tool and stay verbatim.
    if tool_name == "execute_step":
        return _cap_execute_step_result(payload, preview_limit)
    return payload


def check_history(history, maximum=48000):
    estimate = estimate_tokens(history)
    if estimate > maximum:
        raise RuntimeError("context_budget_exceeded; approximate token estimate; complete protocol retained")
    return {"estimate": estimate, "maximum": maximum, "method": "approximate_utf8_bytes_div2"}


def prepare_history(history, maximum=48000, state=None, preview_limit=DEFAULT_PREVIEW_LIMIT):
    """Keep every protocol item/pair; remove redundant diagnostics from result views.

    Original full responses remain in SQLite/transcript. Completed exchanges can
    be replaced by a state summary; no individual tool request/reply is sliced.
    `preview_limit` bounds only the execute_step artifact preview (and the same
    projection inside the archival snapshot); it never truncates persisted data.
    """
    preview_limit = _resolve_preview_limit(preview_limit)
    prepared = copy.deepcopy(history)
    tool_names = _tool_names_by_call_id(history)

    def compact(value):
        if isinstance(value, list):
            return [compact(item) for item in value]
        if not isinstance(value, dict):
            return value
        output = {}
        for key, item in value.items():
            if key in {"explain", "lineage", "input_hashes", "path", "source_path"}:
                continue
            if key == "metric_contract" and "artifact_id" in value:
                continue
            if key == "sql" and "artifact_id" in value:
                continue
            if key == "steps" and isinstance(item, list):
                output[key] = [{k: step[k] for k in ("step_id", "type", "output_name", "depends_on") if k in step} for step in item]
            else:
                output[key] = compact(item)
        return output

    for item in prepared:
        if item.get("type") != "function_call_output":
            continue
        try:
            payload = json.loads(item["output"])
        except (ValueError, KeyError, TypeError):
            continue
        item["output"] = dumps(_lean_result(tool_names.get(item.get("call_id")), compact(payload), preview_limit))
    if estimate_tokens(prepared) <= maximum:
        return prepared
    # Compact WHOLE completed exchanges. Never leave a tool request without its reply.
    groups, current, prefix = [], [], []
    has_outputs = False
    for item in prepared:
        is_model = item.get("type") in {"chat_message", "function_call", "message", "reasoning"}
        is_user = item.get("role") == "user"
        if (is_model or is_user) and current and has_outputs:
            groups.append(current)
            current = []
            has_outputs = False
        if not groups and not current and item.get("role") in {"system", "user"}:
            prefix.append(item)
        else:
            current.append(item)
        if item.get("type") == "function_call_output":
            has_outputs = True
    if current:
        groups.append(current)
    for keep in (6, 4, 2, 1):
        recent = [item for group in groups[-keep:] for item in group]
        outputs = {item.get("call_id") for item in recent if item.get("type") == "function_call_output"}
        requests = set()
        for item in recent:
            if item.get("type") == "function_call":
                requests.add(item["call_id"])
            if item.get("type") == "chat_message":
                requests.update(call["id"] for call in item["message"].get("tool_calls", []))
        if outputs != requests:
            continue
        if state is None:
            snapshot = {"notice": "Earlier completed exchanges archived in local transcript; do not infer missing facts."}
        else:
            refs = {}
            for search in state.get("evidence", []):
                for record in search.get("evidence", []):
                    refs[record["evidence_id"]] = record
            ticket_searches = []
            for search_id, entry in list(state.get("ticket_searches", {}).items())[-4:]:
                result = entry if isinstance(entry, dict) else {}
                ticket_searches.append({
                    "search_id": search_id,
                    "source_id": result.get("source_id"),
                    "filters": result.get("filters"),
                    "source_ticket_count": result.get("source_ticket_count"),
                    "population_ticket_count": result.get("population_ticket_count"),
                    "matched_ticket_count": result.get("matched_ticket_count"),
                    "returned_ticket_count": result.get("returned_ticket_count"),
                    "evidence_ids": [r["evidence_id"] for r in result.get("evidence", [])],
                })
            snapshot = {"question": state["question"], "metric_contract": state.get("bound_contract"),
                        "plan_version": state["plan_version"], "plan": state["plan"],
                        "context_refs": list(refs.values()),
                        "observations": state.get("observed_facts", [])[-8:], "open_questions": state.get("open_questions", []),
                        "ticket_searches": ticket_searches,
                        "artifacts": {name: _project_artifact_metadata({k: a.get(k) for k in ("artifact_id", "output_name", "row_count", "schema", "preview")}, preview_limit) for name, a in state["artifacts"].items()},
                        "notice": "Earlier complete tool exchanges archived in local transcript. These are persisted references, not new instructions. Use inspect_artifact for details."}
        candidate = prefix + [{"role": "user", "content": "持久化状态摘要：" + dumps(snapshot)}] + recent
        if estimate_tokens(candidate) <= maximum:
            return candidate
    return prepared  # Caller explicitly stops rather than silently clipping protocol items.
