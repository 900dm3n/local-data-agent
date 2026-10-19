"""Host validated tools; model can propose actions but cannot approve memory."""

import json
import os
import re
import shutil
import uuid
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq

from .artifacts import file_hash, jsonable, verified_artifact, write_json
from .execution_diagnostics import attribute_step_failure
from .executor import Executor
from .explanation import build_explanation, render_explanation
from .lineage import build_lineage
from .memory import MemoryStore
from .plan import affected_steps, apply_plan_patch, validate_plan
from .report import render_report
from .retrieval import ContextRetriever, source_rows
from .semantic_ops import run_operator
from .table_narrative import compile_table_narrative
from .ticket_index import TicketIndex
from .validation import check_artifacts, validate_sources


def _schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


STRING = {"type": "string"}
INTEGER = {"type": "integer"}
STRINGS = {"type": "array", "items": {"type": "string"}}
DEFINITIONS = {
    "search_context": ("Discover source tables, fields and business evidence. types only schema/document/metric; empty arrays mean all. Use profile_values for field values.", {"query": STRING, "source_ids": STRINGS, "types": {"type": "array", "items": {"type": "string", "enum": ["schema", "document", "metric"]}}}),
    "inspect_schema": ("Inspect registered source table fields and keys.", {"source_id": STRING, "table": STRING}),
    "profile_values": ("Get bounded field values from registered source, never arbitrary file paths.", {"source_id": STRING, "table": STRING, "column": STRING}),
    "bind_metric": ("Select an approved business metric. Clarify ambiguous income definitions before binding.", {"metric_id": STRING}),
    "propose_plan": ("Create a MiniPlan. steps_json is a JSON list of LOAD/SELECT/RUN/SAVE steps. LOAD source_id/table; SELECT sql/depends_on; step_id/output_name unique. State why this plan serves the question.", {"steps_json": STRING, "reason": STRING}),
    "revise_plan": ("Replace the FULL plan with a new JSON step list. Add analysis or repair errors. Changed steps and downstream results invalidated.", {"base_plan_version": INTEGER, "steps_json": STRING, "reason": STRING}),
    "patch_plan": ("Change an existing MiniPlan incrementally instead of resubmitting it. upsert_steps_json is a JSON list of COMPLETE step definitions with step_id/type/output_name/depends_on (LOAD also source_id/table); an existing step_id replaces that whole step, a new step_id is appended. remove_step_ids deletes steps by step_id. Deletion must also remove or repair dependent references; merged full DAG is validated. base_plan_version is the current plan version and is required. A patch with no additions, replacements or removals is a valid no-op.", {"base_plan_version": INTEGER, "upsert_steps_json": STRING, "remove_step_ids": STRINGS, "reason": STRING}),
    "execute_step": ("Execute named step AND any incomplete upstream dependencies. Returns all new artifacts and a bounded selected result.", {"plan_version": INTEGER, "step_id": STRING}),
    "inspect_artifact": ("Inspect result/SQL/schema/lineage using registered output_name or artifact_id.", {"artifact_id": STRING, "view": STRING}),
    "record_observation": ("Record artifact-backed observations and still-open questions; then choose another tool if analysis incomplete. facts_json list {artifact_id,description}.", {"facts_json": STRING, "open_questions": STRINGS, "next_action": STRING}),
    "check_result": ("Run registered integrity/finite/monthly_reconciliation/money_units checks, plus independent source event reconciliation when applicable. No test gold.", {"artifact_ids": STRINGS, "check_ids": STRINGS}),
    "ask_user": ("Pause if business meaning or required parameter unclear. Saves question for CLI resume.", {"question": STRING, "reason": STRING, "choices": STRINGS}),
    "publish_tables": ("Preferred publication tool: select executed result tables; host automatically binds each displayed row/field and validates business meaning. Choose month totals and channel totals for period comparison, plus detail/comparison/semantic tables as requested. At most forty scalar facts, never silently truncate. summary qualitative, monetary numbers host-generated. Unresolved questions require stated limitations.", {"plan_version": INTEGER, "artifact_ids": STRINGS, "summary": STRING, "open_questions": STRINGS}),
    "publish_verified_tables": ("Preferred publication tool for purely monetary period/channel tables. Provide ONLY the plan version, executed artifact IDs and open questions: the host reads the actual verified Parquet rows selected, derives every amount, direction and comparison itself, and does not accept model-supplied financial prose. Use it instead of writing free financial direction text. Keep tickets and business explanations on publish_analysis.", {"plan_version": INTEGER, "artifact_ids": STRINGS, "open_questions": STRINGS}),
    "publish_analysis": ("Narrative publication tool: name an executed monthly totals table and its channel breakdown for a base/current period comparison. The host derives amounts, recomputes any saved ticket search and validates every source; no model amounts, counts or quotes are accepted.", {"plan_version": INTEGER, "month_artifact": STRING, "channel_artifact": STRING, "base_month": STRING, "current_month": STRING, "ticket_search_id": STRING, "open_questions": STRINGS}),
    "search_tickets": ("Search frozen source tickets by keyword. month filters created_at in Australia/Sydney local time, channel_code filters ticket.channel_code; empty strings mean no filter. limit is 1..8. Counters: source_ticket_count is the whole source, population_ticket_count the full filtered population, matched_ticket_count the keyword matches, returned_ticket_count the quotes returned. Returned tickets are reference-only association, not causality.", {"query": STRING, "source_id": STRING, "month": STRING, "channel_code": STRING, "limit": INTEGER}),
    "inspect_source_checks": ("Report the actual frozen source checks (existence, hash, row count, parseability) before trusting any derived local_month mapping. Read-only; uses no gold data.", {}),
    "finalize_run": ("Publish validated answer. claims_json list {claim_id,artifact_id,column,filter,aggregation,display_unit,label}. aggregation single_row|sum|count; unit CNY_yuan|raw. Monetary columns end _cents. Month values YYYY-MM, names obtained from artifacts. Summary qualitative ONLY: no digits, amounts or dates; numbers host-formatted via claims. Label cause hypotheses and evidence limitations. Empty open_questions only when answered or explicit limitation.", {"plan_version": INTEGER, "claims_json": STRING, "summary": STRING, "open_questions": STRINGS}),
}
TOOL_DEFINITIONS = [{"type": "function", "name": name, "description": description,
                     "strict": True, "parameters": _schema(properties)}
                    for name, (description, properties) in DEFINITIONS.items()]


def validate_arguments(name, arguments):
    if name not in DEFINITIONS:
        raise ValueError("Unknown tool")
    if isinstance(arguments, str):
        if len(arguments) > 100000:
            raise ValueError("Tool arguments exceed limit")
        arguments = json.loads(arguments)
    properties = DEFINITIONS[name][1]
    if not isinstance(arguments, dict) or set(arguments) != set(properties):
        raise ValueError(f"Arguments must contain exactly {list(properties)}")
    for key, schema in properties.items():
        value = arguments[key]
        if schema["type"] == "string" and (not isinstance(value, str) or len(value) > 100000):
            raise ValueError(f"{key} must be bounded string")
        if schema["type"] == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"{key} must be integer")
        if schema["type"] == "array" and (not isinstance(value, list) or len(value) > 64 or any(not isinstance(v, str) or len(v) > 2000 for v in value)):
            raise ValueError(f"{key} must be bounded string list")
    return arguments


def _project(rows, columns):
    """Host-side projection of verified Parquet rows; a missing required column is an error, never a silent zero."""
    projected = []
    for row in rows:
        for column in columns:
            if column not in row:
                raise ValueError(f"Required column {column} is absent from the verified table")
        projected.append({column: row[column] for column in columns})
    return projected


class ToolContext:
    def __init__(self, root, run_dir, payload, manifest, store, model):
        self.root, self.run_dir = Path(root), Path(run_dir)
        self.state, self.manifest, self.store, self.model = payload, manifest, store, model
        self.retriever = ContextRetriever(self.run_dir, manifest)
        self.executor = Executor(self.run_dir, manifest)

    def artifact(self, identifier):
        for name, artifact in self.state["artifacts"].items():
            if identifier in {name, artifact["artifact_id"]}:
                verified_artifact(self.run_dir, artifact)
                return artifact
        raise ValueError("Unknown artifact")

    def metric(self):
        binding = self.state.get("metric_binding")
        if not binding:
            raise ValueError("Bind an approved metric before planning/execution")
        candidates = [contract for contract in self.manifest["metric_contracts"]
                      if contract["metric_id"] == binding and contract.get("status") == "APPROVED"]
        if len(candidates) != 1 or candidates[0] != self.state.get("bound_contract"):
            raise ValueError("Approved metric binding changed or is ambiguous")
        return self.state["bound_contract"]

    def dispatch(self, name, args):
        args = validate_arguments(name, args)
        if name == "search_tickets":
            searches = self.state.setdefault("ticket_searches", {})
            if len(searches) >= 8:
                raise ValueError("At most 8 ticket searches per run")
            index = TicketIndex(self.manifest)
            try:
                result = index.search(**args)
            finally:
                index.close()
            search_id = "ticket_search_" + uuid.uuid4().hex
            searches[search_id] = result
            write_json(self.run_dir / "ticket_searches" / f"{search_id}.json", result)
            return {"search_id": search_id, **deepcopy(result)}
        if name == "inspect_source_checks":
            actual = validate_sources(self.manifest)
            failed = [{"name": item["name"], "passed": False,
                       "details": item.get("details", item.get("reason", ""))}
                      for item in actual.get("checks", []) if not item.get("passed")]
            answer = {"passed": bool(actual.get("passed")), "checks": [{"name": item["name"], "passed": bool(item.get("passed"))} for item in actual.get("checks", [])]}
            if failed:
                answer["failed_details"] = failed
            answer["timestamp_basis_verified"] = any(item["name"] == "aware_event_timestamps" and item.get("passed") is True for item in actual.get("checks", []))
            answer["derived_event_months_verified"] = any(item["name"] == "derived_event_months" and item.get("passed") is True for item in actual.get("checks", []))
            answer["note"] = ("Source UTC event months: successful payment/refund rows use the financial timestamp; non-success rows may fall back to attempted_at.")
            return answer
        if name == "search_context":
            result = self.retriever.search(args["query"], args["source_ids"] or None, args["types"] or None)
            self.state["evidence"].append(result)
            write_json(self.run_dir / "retrieval.json", self.state["evidence"])
            return result
        if name == "inspect_schema":
            return self.retriever.inspect_schema(**args)
        if name == "profile_values":
            return self.retriever.profile_values(**args)
        if name == "bind_metric":
            matches = [contract for contract in self.manifest.get("metric_contracts", [])
                       if contract["metric_id"] == args["metric_id"] and contract.get("status") == "APPROVED"]
            if len(matches) != 1:
                raise ValueError("Metric unavailable or unapproved; ask user for an explicit definition")
            if not self.state.get("metric_binding"):
                chosen = self.state.get("last_user_answer", self.state["question"]).lower()
                aliases = matches[0].get("aliases", [])
                if args["metric_id"] == "CASH_NET_v1":
                    aliases = [*aliases, "现金净额", "cash net", "cash_net_v1", "实收款减成功退款", "实际收款减成功退款"]
                is_explicit = args["metric_id"].lower() in chosen or any(str(alias).lower() in chosen for alias in aliases)
                alternatives = any(marker in chosen for marker in ["还是", "或者", " or "])
                if not is_explicit or alternatives:
                    raise ValueError("Metric intent is not uniquely confirmed. Use ask_user to clarify the requested business definition; an approved cash metric is not automatically the meaning of income.")
            if self.state["artifacts"] and self.state.get("metric_binding") != args["metric_id"]:
                raise ValueError("Changing metric after execution requires a new run")
            self.state["metric_binding"] = args["metric_id"]
            self.state["bound_contract"] = matches[0].copy()
            write_json(self.run_dir / "metric_contract.json", matches[0])
            return {"metric_contract": matches[0]}
        if name in {"propose_plan", "revise_plan", "patch_plan"}:
            return self.plan_call(name, args)
        if name == "execute_step":
            self.metric()
            if args["plan_version"] != self.state["plan_version"]:
                raise ValueError("Stale plan version")
            created = []
            output = self.execute(args["step_id"], created)
            return {"artifact": output, "executed_outputs": created, "mode": self.state["mode"]}
        if name == "inspect_artifact":
            artifact = self.artifact(args["artifact_id"])
            if args["view"] not in {"result", "sql", "schema", "lineage", "all"}:
                raise ValueError("Unsupported artifact view")
            if args["view"] == "all":
                return self.visible(artifact)
            return {args["view"]: artifact.get("preview" if args["view"] == "result" else args["view"])}
        if name == "record_observation":
            facts = json.loads(args["facts_json"])
            if not isinstance(facts, list) or len(facts) > 64:
                raise ValueError("Observation list too large")
            for fact in facts:
                if not isinstance(fact, dict) or set(fact) != {"artifact_id", "description"} or not isinstance(fact["description"], str):
                    raise ValueError("Each observation needs artifact_id and description")
                artifact = self.artifact(fact["artifact_id"])
                fact["artifact_id"] = artifact["artifact_id"]
                fact["status"] = "MODEL_OBSERVATION_REQUIRES_VALIDATION"
            self.state["observed_facts"].extend(facts)
            self.state["open_questions"] = args["open_questions"]
            self.state["next_action"] = args["next_action"]
            return {"saved": True, "open_questions": args["open_questions"]}
        if name == "check_result":
            selected = {self.artifact(identifier)["output_name"]: self.artifact(identifier) for identifier in args["artifact_ids"]}
            if not selected:
                raise ValueError("Select at least one artifact")
            result = check_artifacts(selected, args["check_ids"])
            ledger = self.reconcile(selected)
            result["checks"].extend(ledger["checks"])
            result["passed"] = result["passed"] and ledger["passed"]
            write_json(self.run_dir / "validation.json", result)
            return result
        if name == "ask_user":
            if not args["question"].strip():
                raise ValueError("Clarification question must be nonempty")
            self.state["pending_question"] = args
            self.state["status"] = "WAITING_INPUT"
            self.state["stop_reason"] = "needs_user_input"
            return {"status": "WAITING_INPUT", **args}
        if name == "publish_tables":
            return self.publish_tables(args)
        if name == "publish_verified_tables":
            return self.publish_verified_tables(args)
        if name == "publish_analysis":
            return self.publish_analysis(args)
        if name == "finalize_run":
            return self.finalize(args)
        raise ValueError("Unknown tool")

    def plan_call(self, name, args):
        """Single commit path for propose_plan, revise_plan and patch_plan; files land before any mutation."""
        self.metric()
        old = self.state["plan"]
        if name == "propose_plan":
            if old:
                raise ValueError("Use revise_plan for an existing plan")
            steps = validate_plan(json.loads(args["steps_json"]), [s["source_id"] for s in self.manifest["sources"]])
        elif name == "revise_plan":
            if args["base_plan_version"] != self.state["plan_version"]:
                raise ValueError("Stale plan version")
            steps = validate_plan(json.loads(args["steps_json"]), [s["source_id"] for s in self.manifest["sources"]])
        else:
            if not old:
                raise ValueError("Patch needs an existing plan")
            if args["base_plan_version"] != self.state["plan_version"]:
                raise ValueError("Stale plan version")
            upserts = json.loads(args["upsert_steps_json"])
            steps = apply_plan_patch(old, upserts, args["remove_step_ids"],
                                     [s["source_id"] for s in self.manifest["sources"]])
        requested_upserts = [step["step_id"] for step in upserts] if name == "patch_plan" else []
        requested_removals = list(args["remove_step_ids"]) if name == "patch_plan" else []
        affected = sorted(affected_steps(old, steps))
        patch = name == "patch_plan"
        artifacts = self.state["artifacts"]
        invalidated = [key for key, artifact in artifacts.items() if artifact["step_id"] in affected]
        if patch and not affected:
            # Equivalent patch: report the decision without writing a plan file or changing any state.
            return {"plan_version": self.state["plan_version"], "changed": False,
                    "upserted": requested_upserts, "removed": requested_removals,
                    "affected": [], "invalidated": [], "retained": list(artifacts)}
        version = self.state["plan_version"] + 1
        record = {"version": version, "steps": steps, "reason": args["reason"], "invalidated": invalidated}
        if patch:
            record.update({"patch": True, "base_version": self.state["plan_version"],
                           "upserts": requested_upserts, "removals": requested_removals, "affected": affected})
        write_json(self.run_dir / "plans" / f"plan_{version}.json", record)
        for key in invalidated:
            del artifacts[key]
        self.state["plan"] = steps
        self.state["plan_version"] = version
        self.state["claims"] = []
        self.state.pop("report_path", None)
        self.state.pop("answer", None)
        self.state.pop("narrative_verification", None)
        if patch:
            return {"plan_version": version, "changed": True,
                    "upserted": requested_upserts, "removed": requested_removals,
                    "affected": affected, "invalidated": invalidated, "retained": list(artifacts)}
        return {"plan_version": version, "steps": steps,
                "invalidated": invalidated, "retained": list(artifacts)}

    def visible(self, artifact):
        keys = {"artifact_id", "step_id", "output_name", "kind", "row_count", "schema", "preview", "operator", "mode", "prompt_version"}
        return {key: value for key, value in artifact.items() if key in keys}

    def publish_tables(self, args, *, explanation=None, table_narrative=None):
        """Presentation is deterministic. The Agent still chooses tables and narrative.

        ``explanation`` (ticket evidence) and ``table_narrative`` (host-compiled
        table facts) are private host paths; no model tool call can supply either.
        """
        if explanation is not None and table_narrative is not None:
            raise ValueError("Provide either explanation or table_narrative, not both")
        finance = {"net_cents", "paid_cents", "refund_cents", "base_net_cents", "current_net_cents",
                   "base_paid_cents", "current_paid_cents", "base_refund_cents", "current_refund_cents",
                   "delta_cents", "paid_delta_cents", "refund_delta_cents", "change_pct", "payment_count", "refund_count"}
        field_labels = {"net_cents": "现金净额", "paid_cents": "支付金额", "refund_cents": "退款金额",
                        "base_net_cents": "基期现金净额", "current_net_cents": "本期现金净额",
                        "delta_cents": "净额差额", "change_pct": "变化率（百分数）",
                        "payment_count": "支付笔数", "refund_count": "退款笔数",
                        "refund_id": "退款编号", "order_id": "订单编号", "ticket_id": "工单编号", "label": "语义标签（待复核）"}
        claims, seen = [], set()
        for identifier in args["artifact_ids"]:
            artifact = self.artifact(identifier)
            if artifact["artifact_id"] in seen:
                raise ValueError("Duplicate selected table")
            seen.add(artifact["artifact_id"])
            table = pq.read_table(verified_artifact(self.run_dir, artifact))
            if not 1 <= table.num_rows <= 20:
                raise ValueError("Presentation requires 1..20 rows per selected table; create a query for the requested scope")
            columns = set(table.column_names)
            record_keys = [name for name in ("refund_id", "ticket_id", "order_id") if name in columns]
            keys = record_keys[:1] or [name for name in ("month", "base_month", "current_month", "channel_code") if name in columns]
            if not keys and table.num_rows != 1:
                raise ValueError("Table presentation requires record IDs or period/channel keys")
            display_columns = [name for name in table.column_names if name in finance]
            if record_keys:
                display_columns = [name for name in table.column_names if name in finance | {"refund_id", "order_id", "ticket_id", "label"}]
            if not display_columns:
                raise ValueError("No registered presentable columns; use explicit bound claims for supported scalar fields")
            for row_number, row in enumerate(table.to_pylist(), 1):
                filters = {key: row[key] for key in keys}
                # Labels contain period/channel context; IDs are shown as raw bound cells.
                labels = [str(row[key]) for key in ("month", "base_month", "current_month", "channel_code") if key in keys]
                prefix = " / ".join(labels)
                for column in display_columns:
                    label = field_labels.get(column, column.replace("_", " "))
                    if prefix:
                        label = prefix + " " + label
                    claims.append({"claim_id": f"auto_{len(claims)+1}", "artifact_id": artifact["artifact_id"],
                                   "column": column, "filter": filters, "aggregation": "single_row",
                                   "display_unit": "CNY_yuan" if column.endswith("_cents") else "raw", "label": label})
                    if len(claims) > 40:
                        raise ValueError("Selected tables exceed forty scalar facts; narrow the selected outputs, no rows were silently omitted")
        return self.finalize({"plan_version": args["plan_version"], "claims_json": json.dumps(claims, ensure_ascii=False),
                              "summary": args["summary"], "open_questions": args["open_questions"]},
                             explanation=explanation, table_narrative=table_narrative)

    def publish_verified_tables(self, args):
        """Host-compiled narrative publication for purely monetary tables.

        The model supplies identifiers only. Every amount, direction and
        comparison in the summary is compiled here from the verified Parquet
        rows that were actually selected, then handed to the existing
        ``publish_tables``/``finalize`` source-event verification path so all
        claim, source and reconciliation checks stay unchanged.
        """
        contract = self.metric()
        if contract.get("metric_id") != "CASH_NET_v1":
            raise ValueError("Narrative publication requires the approved cash net metric")
        if args["plan_version"] != self.state["plan_version"]:
            raise ValueError("Stale plan version")
        tables = []
        for identifier in args["artifact_ids"]:
            artifact = self.artifact(identifier)
            # Row count is checked from Parquet metadata BEFORE reading; exact
            # integral Arrow decimals are converted to int without float loss.
            rows = pq.ParquetFile(verified_artifact(self.run_dir, artifact)).metadata.num_rows
            if not 1 <= rows <= 20:
                raise ValueError("Presentation requires 1..20 rows per selected table; create a query for the requested scope")
            table = pq.read_table(verified_artifact(self.run_dir, artifact))
            tables.append({"artifact_id": artifact["artifact_id"],
                           "rows": [jsonable(row) for row in table.to_pylist()]})
        # Compile before publication: a rejected shape never reaches finalize.
        compiled = compile_table_narrative(tables)
        return self.publish_tables({"plan_version": args["plan_version"],
                                    "artifact_ids": args["artifact_ids"],
                                    "summary": compiled["summary"],
                                    "open_questions": args["open_questions"]},
                                   table_narrative=compiled)

    def publish_analysis(self, args):
        """Narrative publication. Host derives amounts, quotes and scope; the caller supplies identifiers only."""
        if args["plan_version"] != self.state["plan_version"]:
            raise ValueError("Stale plan version")
        contract = self.metric()
        if contract.get("metric_id") != "CASH_NET_v1":
            raise ValueError("Narrative publication requires the approved cash net metric")
        month_artifact = self.artifact(args["month_artifact"])
        channel_artifact = self.artifact(args["channel_artifact"])
        if month_artifact["artifact_id"] == channel_artifact["artifact_id"]:
            raise ValueError("Month and channel publication need two different artifacts")
        if args["base_month"] == args["current_month"]:
            raise ValueError("Base and current periods must differ")
        month_rows = pq.read_table(verified_artifact(self.run_dir, month_artifact)).to_pylist()
        channel_rows = pq.read_table(verified_artifact(self.run_dir, channel_artifact)).to_pylist()
        if not month_rows or not channel_rows:
            raise ValueError("Publication requires nonempty month and channel tables")
        required = {"month", "paid_cents", "refund_cents", "net_cents"}
        if not required.issubset({field["name"] for field in month_artifact["schema"]}):
            raise ValueError("Month table must project month, paid_cents, refund_cents and net_cents")
        if not required.union({"channel_code"}).issubset({field["name"] for field in channel_artifact["schema"]}):
            raise ValueError("Channel table must project month, channel_code, paid_cents, refund_cents and net_cents")
        month_rows = _project(month_rows, required)
        channel_rows = _project(channel_rows, required | {"channel_code"})
        months = {row["month"] for row in month_rows}
        channel_months = {row["month"] for row in channel_rows}
        if args["base_month"] not in months or args["current_month"] not in months:
            raise ValueError("Month table does not cover the requested periods")
        if args["base_month"] not in channel_months or args["current_month"] not in channel_months:
            raise ValueError("Channel table does not cover the requested periods")
        saved_result = None
        if args["ticket_search_id"]:
            searches = self.state.get("ticket_searches", {})
            if args["ticket_search_id"] not in searches:
                raise ValueError("Unknown ticket_search_id; only a saved search from this run can be published")
            saved_result = searches[args["ticket_search_id"]]
            params = saved_result["filters"]
            index = TicketIndex(self.manifest)
            try:
                recomputed = index.search(query=saved_result["query"], source_id=saved_result["source_id"],
                                          month=params["month"], channel_code=params["channel_code"],
                                          limit=params["limit"])
            finally:
                index.close()
            if recomputed != saved_result:
                raise ValueError("Saved ticket search no longer matches the frozen tickets; citations or counts changed")
        actual_checks = validate_sources(self.manifest)
        if actual_checks.get("passed") is not True:
            raise ValueError("Frozen source validation failed for this publication")
        explanation = build_explanation(month_rows, channel_rows, args["base_month"], args["current_month"],
                                        ticket_result=saved_result, source_validation=actual_checks,
                                        artifact_refs={"month": [month_artifact["artifact_id"]],
                                                       "channel": [channel_artifact["artifact_id"]]})
        return self.publish_tables({"plan_version": args["plan_version"],
                                    "artifact_ids": [month_artifact["artifact_id"], channel_artifact["artifact_id"]],
                                    "summary": "依据已核验的金额产物生成说明；业务根因尚未证实，关联不等于因果。",
                                    "open_questions": args["open_questions"]}, explanation=explanation)

    @attribute_step_failure
    def execute(self, step_id, created):
        steps = {step["step_id"]: step for step in self.state["plan"]}
        if step_id not in steps:
            raise ValueError("Unknown step_id")
        step = steps[step_id]
        artifacts = self.state["artifacts"]
        if step["output_name"] in artifacts:
            artifact = artifacts[step["output_name"]]
            verified_artifact(self.run_dir, artifact)
            return self.visible(artifact)
        for dependency in step["depends_on"]:
            self.execute(dependency, created)
        if step["type"] in {"LOAD", "SELECT"}:
            result = self.executor.execute(step, artifacts)
        elif step["type"] == "RUN":
            if self.state.get("semantic_requests", 0) >= 2:
                raise ValueError("Semantic API request budget exceeded (maximum two per run)")
            self.state["semantic_requests"] = self.state.get("semantic_requests", 0) + 1
            self.store.save(self.state, "semantic_request", {"request_number": self.state["semantic_requests"]})
            inputs = {steps[dep]["output_name"]: artifacts[steps[dep]["output_name"]] for dep in step["depends_on"]}
            result = run_operator(step, inputs, self.run_dir, self.model, self.state["mode"])
            self.state["usage"].append({"kind": "semantic", **result.get("usage", {})})
        else:
            source = artifacts[steps[step["depends_on"][0]]["output_name"]]
            identifier = "artifact_" + uuid.uuid4().hex
            path = self.run_dir / "artifacts" / (identifier + ".parquet")
            temporary = path.with_suffix(".tmp")
            shutil.copyfile(verified_artifact(self.run_dir, source), temporary)
            os.replace(temporary, path)
            result = {**source, "artifact_id": identifier, "step_id": step_id,
                      "output_name": step["output_name"], "path": str(path), "sha256": file_hash(path)}
        if not check_artifacts({step["output_name"]: result}, ["finite"])["passed"]:
            raise ValueError("Nonfinite numeric result; handle zero denominators explicitly before publication")
        result["metric_contract"] = self.metric()
        artifacts[step["output_name"]] = result
        created.append(step["output_name"])
        write_json(self.run_dir / "artifacts.json", artifacts)
        self.store.save(self.state, "step_completed", {"step_id": step_id, "artifact_id": result["artifact_id"]})
        return self.visible(result)

    def reconcile(self, selected, publication=False):
        """Generic approved event-ledger rule against frozen sources, not fixture gold."""
        if self.metric().get("metric_id") != "CASH_NET_v1":
            raise ValueError("Unsupported approved metric verifier")
        for snapshot in self.manifest["sources"]:
            if file_hash(snapshot["path"]) != snapshot["sha256"]:
                raise ValueError("Frozen source hash mismatch")
        source = next(s for s in self.manifest["sources"] if s["source_id"] == "business")
        orders, refunds = source_rows(source, "orders"), source_rows(source, "refunds")
        channel_source = next(s for s in self.manifest["sources"] if s["source_id"] == "channels")
        known_channels = {row["channel_code"] for row in source_rows(channel_source, "channels")}
        lookup = {row["order_id"]: row for row in orders}
        total, channels = defaultdict(int), defaultdict(int)
        parts, channel_parts = defaultdict(int), defaultdict(int)
        counts, channel_counts = defaultdict(int), defaultdict(int)
        refund_lookup = {row["refund_id"]: row for row in refunds}
        for rows, status, time_field, amount_field, sign in (
            (orders, "paid", "paid_at", "paid_cents", 1),
            (refunds, "succeeded", "refunded_at", "refund_cents", -1),
        ):
            for row in rows:
                if row["status"] != status:
                    continue
                month = datetime.fromisoformat(row[time_field].replace("Z", "+00:00")).astimezone(ZoneInfo(self.metric()["timezone"])).strftime("%Y-%m")
                channel = row.get("channel_code") or lookup[row["order_id"]]["channel_code"]
                total[month] += sign * row[amount_field]
                channels[(month, channel)] += sign * row[amount_field]
                part = "paid_cents" if sign == 1 else "refund_cents"
                parts[(month, part)] += row[amount_field]
                channel_parts[(month, channel, part)] += row[amount_field]
                count_name = "payment_count" if sign == 1 else "refund_count"
                counts[(month, count_name)] += 1
                channel_counts[(month, channel, count_name)] += 1
        checks = []
        verified_columns = {}
        for name, artifact in selected.items():
            columns = {field["name"] for field in artifact["schema"]}
            period_money = {"base_net_cents", "current_net_cents", "base_paid_cents", "current_paid_cents", "base_refund_cents", "current_refund_cents", "paid_delta_cents", "refund_delta_cents", "delta_cents"}
            money = columns.intersection({"net_cents", "paid_cents", "refund_cents"} | period_money)
            count_roles = columns.intersection({"payment_count", "refund_count"})
            money |= count_roles
            if not money:
                continue
            rows = pq.read_table(verified_artifact(self.run_dir, artifact)).to_pylist()
            supported = (
                ("refund_id" in columns and money == {"refund_cents"})
                or ("order_id" in columns and money == {"paid_cents"})
                or "month" in columns
                or {"base_month", "current_month"}.issubset(columns)
            )
            if not supported:
                required_scope = ["base_month", "current_month"] if money & period_money else ["month"]
                checks.append({"name": "source_event_reconciliation:" + name, "passed": False,
                               "details": {"metric": "CASH_NET_v1",
                                           "counterexamples": [{"group": "unsupported", "error": "missing_scope_columns"}],
                                           "actual_columns": sorted(columns),
                                           "required_scope_columns": required_scope,
                                           "hints": "This output shape has no registered period scope. Named periods are YYYY-MM values; project them as base_month/current_month (comparison derived tables: base_month/current_month/delta_cents/change_pct) or as a single month column for month-level rows. Keep channel_code in channel-filtered outputs. Schema guidance only: no expected amounts or SQL are supplied here."}})
                continue
            bad = []
            seen = set()
            for row in rows:
                if "channel_code" in columns and row["channel_code"] not in known_channels:
                    bad.append({"error": "output channel_code is absent from the registered channel mapping"})
                    continue
                period_fields = [field for field in ("month", "base_month", "current_month") if field in columns]
                if any(not isinstance(row[field], str) or not re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", row[field]) or row[field].startswith("0000") for field in period_fields):
                    bad.append({"error": "period identifiers must be YYYY-MM strings; amount values cannot be used as month identifiers"})
                    continue
                if "refund_id" in columns and money == {"refund_cents"}:
                    original = refund_lookup.get(row["refund_id"])
                    valid = original is not None and (not publication or original["status"] == "succeeded") and original["refund_cents"] == row["refund_cents"]
                    if original:
                        if "order_id" in columns:
                            valid = valid and row["order_id"] == original["order_id"]
                        if "channel_code" in columns:
                            valid = valid and row["channel_code"] == lookup[original["order_id"]]["channel_code"]
                        if "month" in columns:
                            actual_month = datetime.fromisoformat(original["refunded_at"].replace("Z", "+00:00")).astimezone(ZoneInfo(self.metric()["timezone"])).strftime("%Y-%m")
                            valid = valid and row["month"] == actual_month
                    key = row["refund_id"]
                elif "order_id" in columns and money == {"paid_cents"}:
                    original = lookup.get(row["order_id"])
                    valid = original is not None and (not publication or original["status"] == "paid") and original["paid_cents"] == row["paid_cents"]
                    if original and "channel_code" in columns:
                        valid = valid and row["channel_code"] == original["channel_code"]
                    key = row["order_id"]
                elif "month" in columns and money.issubset({"net_cents", "paid_cents", "refund_cents", "payment_count", "refund_count"}):
                    key = (row["month"], row.get("channel_code"))
                    valid = True
                    for column in money:
                        if column == "net_cents":
                            expected = channels.get(key, 0) if "channel_code" in columns else total.get(row["month"], 0)
                        elif column in {"payment_count", "refund_count"}:
                            expected = channel_counts.get((*key, column), 0) if "channel_code" in columns else counts.get((row["month"], column), 0)
                        else:
                            expected = channel_parts.get((*key, column), 0) if "channel_code" in columns else parts.get((row["month"], column), 0)
                        valid = valid and row[column] == expected
                elif {"base_month", "current_month"}.issubset(columns) and money.issubset(period_money):
                    key = (row["base_month"], row["current_month"], row.get("channel_code"))
                    if "channel_code" in columns:
                        before = channels.get((row["base_month"], row["channel_code"]), 0)
                        after = channels.get((row["current_month"], row["channel_code"]), 0)
                    else:
                        before, after = total.get(row["base_month"], 0), total.get(row["current_month"], 0)
                    def part(month, column):
                        return channel_parts.get((month, row["channel_code"], column), 0) if "channel_code" in columns else parts.get((month, column), 0)
                    reference = {
                        "base_net_cents": before, "current_net_cents": after,
                        "base_paid_cents": part(row["base_month"], "paid_cents"),
                        "current_paid_cents": part(row["current_month"], "paid_cents"),
                        "base_refund_cents": part(row["base_month"], "refund_cents"),
                        "current_refund_cents": part(row["current_month"], "refund_cents"),
                    }
                    reference["delta_cents"] = None if before is None or after is None else after - before
                    reference["paid_delta_cents"] = reference["current_paid_cents"] - reference["base_paid_cents"]
                    reference["refund_delta_cents"] = reference["current_refund_cents"] - reference["base_refund_cents"]
                    valid = before is not None and after is not None and all(row[column] == reference[column] for column in money)
                    if "change_pct" in columns:
                        expected_pct = None if before == 0 else (after - before) / before * 100
                        valid = valid and ((row["change_pct"] is None and expected_pct is None) or
                                           (expected_pct is not None and row["change_pct"] is not None and abs(float(row["change_pct"]) - expected_pct) < 0.005000001))
                else:
                    key, valid = "unsupported", False
                if not valid or key in seen:
                    bad.append({"group": str(key), "error": "event-ledger mismatch, unsupported schema or duplicate group",
                                "hints": "Retain channel_code in channel-filtered outputs; absent channel_code means all channels. Period IDs are YYYY-MM. change_pct is percent points, delta is current minus base, cents remain exact."})
                seen.add(key)
            checks.append({"name": "source_event_reconciliation:" + name, "passed": bool(rows) and not bad,
                           "details": {"metric": "CASH_NET_v1", "counterexamples": bad[:5]}})
            if rows and not bad:
                verified_columns[name] = list(money | ({"change_pct"} if "delta_cents" in money and "change_pct" in columns else set()))
        return {"passed": all(check["passed"] for check in checks), "checks": checks, "verified_columns": verified_columns}

    def finalize(self, args, *, explanation=None, table_narrative=None):
        self.metric()
        if explanation is not None and table_narrative is not None:
            raise ValueError("Provide either explanation or table_narrative, not both")
        if args["plan_version"] != self.state["plan_version"]:
            raise ValueError("Stale plan version")
        if args["open_questions"] and not any(term in args["summary"] for term in ["未", "无法", "不足", "假设", "待核查", "限制", "不能"]):
            raise ValueError("State explicit limitations in summary for unresolved questions")
        artifacts = self.state["artifacts"]
        if not artifacts:
            raise ValueError("No executed evidence to publish")
        claims = json.loads(args["claims_json"])
        if not isinstance(claims, list) or not 1 <= len(claims) <= 64:
            raise ValueError("Publish requires 1..64 bound claims")
        selected = {self.artifact(claim["artifact_id"])["output_name"]: self.artifact(claim["artifact_id"]) for claim in claims}
        validation = check_artifacts(selected, ["integrity", "finite"])
        ledger = self.reconcile(selected, publication=True)
        validation["checks"].extend(ledger["checks"])
        validation["passed"] = validation["passed"] and ledger["passed"]
        write_json(self.run_dir / "validation.json", validation)
        if not validation["passed"]:
            return {"error": "publication_validation_failed", "validation": validation}
        for claim in claims:
            artifact = self.artifact(claim["artifact_id"])
            column = claim.get("column", "")
            field_type = next((field["type"] for field in artifact["schema"] if field["name"] == column), "")
            is_numeric = any(token in field_type.lower() for token in ("int", "double", "float", "decimal"))
            is_record_count = claim.get("aggregation") == "count" and column in {"order_id", "refund_id", "ticket_id"}
            if is_record_count:
                source_id = "tickets" if column == "ticket_id" else "business"
                table = "tickets" if column == "ticket_id" else "orders" if column == "order_id" else "refunds"
                source = next(s for s in self.manifest["sources"] if s["source_id"] == source_id)
                originals = {row[column] for row in source_rows(source, table)}
                rows = pq.read_table(verified_artifact(self.run_dir, artifact)).to_pylist()
                ids = [row[column] for row in rows]
                if len(ids) != len(set(ids)) or any(identifier not in originals for identifier in ids):
                    raise ValueError("Record count contains duplicate or nonexistent source IDs")
            if claim.get("display_unit") == "CNY_yuan" or column.endswith("_cents") or column == "change_pct" or is_numeric:
                if column not in ledger["verified_columns"].get(artifact["output_name"], []):
                    raise ValueError(f"Claim {claim.get('claim_id')} column {column} in {artifact['output_name']} has no registered business verifier; use month/net_cents, month/paid_cents/refund_cents/payment_count/refund_count, source record IDs, or base_month/current_month/delta_cents/change_pct")
        if table_narrative is not None:
            # Host-compiled proof is persisted only after every source and claim
            # check above has passed.
            write_json(self.run_dir / "table_narrative.json", table_narrative)
        self.state["status"] = "VALIDATING"
        self.store.save(self.state, "validating")
        report = render_report(self.run_dir, claims, artifacts, args["summary"],
                               evidence=explanation["ticket_evidence"] if explanation is not None else None)
        if explanation is not None:
            report["answer"] = report["answer"] + "\n\n" + render_explanation(explanation)
            report_tmp = self.run_dir / "report.md.tmp"
            report_tmp.write_text(report["answer"], encoding="utf-8")
            os.replace(report_tmp, self.run_dir / "report.md")
            write_json(self.run_dir / "analysis.json", explanation)
        lineage = build_lineage(self.state["plan"], artifacts, self.manifest)
        write_json(self.run_dir / "lineage.json", lineage)
        write_json(self.run_dir / "claims.json", report["claims"])
        if explanation is not None:
            narrative_verification = "host_deterministic_evidence"
        elif table_narrative is not None:
            narrative_verification = "host_deterministic_tables"
        else:
            narrative_verification = "model_prose_requires_review"
        publication_id = "publication_" + uuid.uuid4().hex
        publication_dir = self.run_dir / "publications" / publication_id
        publication_dir.mkdir(parents=True)
        for name in ("report.md", "claims.json", "validation.json", "lineage.json", "evidence.json"):
            if (self.run_dir / name).is_file():
                shutil.copyfile(self.run_dir / name, publication_dir / name)
        if explanation is not None and (self.run_dir / "analysis.json").is_file():
            shutil.copyfile(self.run_dir / "analysis.json", publication_dir / "analysis.json")
        if table_narrative is not None:
            shutil.copyfile(self.run_dir / "table_narrative.json", publication_dir / "table_narrative.json")
        report["report_path"] = str(publication_dir / "report.md")
        self.state.setdefault("publications", []).append({"publication_id": publication_id,
                                                         "report_path": report["report_path"], "plan_version": self.state["plan_version"],
                                                         "narrative_verification": narrative_verification})
        self.state.update({"claims": report["claims"], "report_path": str(report["report_path"]),
                           "answer": report["answer"], "status": "SUCCEEDED",
                           "narrative_verification": narrative_verification,
                           "open_questions": args["open_questions"], "stop_reason": "validated_answer_published"})
        card = MemoryStore(self.root).draft(self.state["run_id"],
            {"metric_binding": self.state["metric_binding"], "plan": self.state["plan"],
             "scope_note": "Plan and definition only. Result amounts remain snapshot-specific."},
            {"fixture": self.state["fixture"], "metric": self.state["metric_binding"]})
        self.state["memory_draft"] = card["memory_id"]
        manifest = {"run_id": self.state["run_id"], "mode": self.state["mode"], "status": "SUCCEEDED",
                    "metric_binding": self.state["metric_binding"], "artifacts": artifacts,
                    "report": {"path": str(report["report_path"]), "sha256": file_hash(report["report_path"])},
                    "validation": validation, "memory_draft": card["memory_id"],
                    "narrative_verification": narrative_verification}
        write_json(self.run_dir / "manifest.json", manifest)
        return {"status": "SUCCEEDED", "answer": report["answer"], "report_path": str(report["report_path"]),
                "memory_draft": card["memory_id"], "narrative_verification": narrative_verification}
