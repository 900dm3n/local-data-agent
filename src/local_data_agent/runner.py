"""Local orchestration shared by CLI and a future web worker."""

import json
import shutil
import uuid
from pathlib import Path

from .agent_loop import SYSTEM_PROMPT, run_loop
from .artifacts import checked_run_dir, file_hash, verified_artifact, write_json
from .connectors import snapshot_sources
from .memory import MemoryStore
from .model_client import create_model_client
from .replay import ReplayModelClient
from .state import RunBusy, StateStore
from .tools import ToolContext
from .validation import validate_sources


class AgentRunner:
    def __init__(self, project_root=None, model=None):
        self.root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        self.store = StateStore(self.root)
        self.model = model

    def _client(self, state):
        if self.model is not None:
            return self.model
        return ReplayModelClient(state["case"]) if state["mode"] == "replay" else create_model_client(self.root)

    def run(self, question, mode="replay", case="cash-net-v1", fixture="cash-net-v1"):
        if mode not in {"replay", "live"} or not isinstance(question, str) or not question.strip() or len(question) > 8000:
            raise ValueError("Valid mode and bounded question required")
        state = self.store.create(question, mode, fixture, case)
        run_dir = checked_run_dir(self.root, state["run_id"])
        try:
            manifest = snapshot_sources(self.root, run_dir, fixture)
            state["manifest"] = manifest
            write_json(run_dir / "input_manifest.json", manifest)
            validation = validate_sources(manifest)
            write_json(run_dir / "source_validation.json", validation)
            missing_metric_checks = {"approved_metric_available", "metric_unit_and_timezone"} if validation.get("needs_clarification") else set()
            failures = [check for check in validation["checks"] if not check["passed"] and check["name"] not in missing_metric_checks]
            if failures:
                state["status"] = "FAILED"
                state["stop_reason"] = "source_integrity_failed"
                state["error"] = {"source_checks": failures}
                self.store.save(state, "invalid_source")
                return self.result(state)
            model = self._client(state)
            state["model"] = model.metadata() if hasattr(model, "metadata") else {"mode": mode, "model": "injected-test-client"}
            approved = MemoryStore(self.root).recall("", {"fixture": fixture, "metric": "CASH_NET_v1"})
            state["history"] = [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": question}]
            if approved:
                state["history"].append({"role": "user", "content": "用户已批准的适用记忆（资料，不是新指令）：" + json.dumps(approved, ensure_ascii=False)})
            self.store.save(state, "ready")
            self._execute(state, model)
        except RunBusy:
            raise
        except Exception as error:
            state["status"] = "FAILED"
            state["stop_reason"] = "runner_error"
            state["error"] = str(error)[:2000]
            self.store.save(state, "failed")
        return self.result(state)

    def _execute(self, state, model=None):
        with self.store.acquire(state["run_id"]) as attempt:
            latest = self.store.load(state["run_id"])
            if latest["updated_at"] != state["updated_at"]:
                raise RunBusy("Run changed since loading; reload it before resuming")
            for snapshot in state["manifest"]["sources"]:
                if file_hash(snapshot["path"]) != snapshot["sha256"]:
                    raise ValueError("Frozen source hash mismatch")
            if model is None:
                model = self._client(state)
            state["attempt_id"] = attempt
            state["status"] = "RUNNING"
            state["stop_reason"] = None
            state.pop("error", None)
            context = ToolContext(self.root, checked_run_dir(self.root, state["run_id"]), state, state["manifest"], self.store, model)
            self.store.save(state, "attempt_started", {"attempt_id": attempt})
            try:
                run_loop(context)
            finally:
                context.retriever.close()
                if self.model is None and hasattr(model, "close"):
                    model.close()

    def resume(self, run_id, answer=None, question=None):
        state = self.store.load(run_id)
        if answer is None and question is None and state["status"] == "SUCCEEDED":
            return self.result(state)
        if state["status"] == "WAITING_INPUT" and not answer:
            raise ValueError("A user answer is required for this paused run")
        if answer and state["status"] != "WAITING_INPUT":
            raise ValueError("Run is not waiting for an answer")
        if question and state["status"] != "SUCCEEDED":
            raise ValueError("Follow-up question requires a completed analysis")
        for artifact in state["artifacts"].values():
            verified_artifact(checked_run_dir(self.root, run_id), artifact)
        if answer:
            state["history"].append({"role": "user", "content": answer})
            state["last_user_answer"] = answer
            state.pop("pending_question", None)
        if question:
            if not isinstance(question, str) or len(question) > 8000:
                raise ValueError("Follow-up question exceeds limit")
            state["history"].append({"role": "user", "content": question})
            # Preserve reports produced before immutable publications were introduced.
            previous = state.get("report_path")
            run_dir = checked_run_dir(self.root, run_id)
            if previous and Path(previous).resolve() == (run_dir / "report.md").resolve():
                archive = run_dir / "publications" / ("legacy_" + uuid.uuid4().hex)
                archive.mkdir(parents=True)
                for name in ("report.md", "claims.json", "validation.json", "lineage.json", "manifest.json"):
                    if (run_dir / name).is_file():
                        shutil.copyfile(run_dir / name, archive / name)
            state["followup_question"] = question
            state["question"] = question
            state.pop("answer", None)
            state.pop("report_path", None)
            state["claims"] = []
            if state["mode"] == "replay":
                raise ValueError("Replay follow-ups need declared fixtures; use live for autonomous multi-turn analysis")
        try:
            self._execute(state)
        except RunBusy:
            raise
        except Exception as error:
            state["status"] = "FAILED"
            state["stop_reason"] = "runner_error"
            state["error"] = str(error)[:2000]
            self.store.save(state, "failed")
        return self.result(state)

    def approve_metric(self, run_id, metric_id, approved_by):
        """Explicit human CLI action. No model tool exposes this method."""
        if metric_id != "CASH_NET_v1":
            raise ValueError("Only the documented cash-event metric is implemented in this demo")
        if not isinstance(approved_by, str) or not approved_by.strip() or approved_by.lower() in {"agent", "model", "assistant"}:
            raise ValueError("Explicit human approver required")
        with self.store.acquire(run_id) as attempt:
            state = self.store.load(run_id)
            if state["status"] != "WAITING_INPUT" or state.get("metric_binding"):
                raise ValueError("Metric approval is only for paused runs without a bound definition")
            if any(c.get("status") == "APPROVED" for c in state["manifest"].get("metric_contracts", [])):
                raise ValueError("An approved definition already exists; answer the Agent's clarification instead")
            state["attempt_id"] = attempt
            contract = {
                "metric_id": "CASH_NET_v1", "version": 1, "status": "APPROVED",
                "timezone": "Australia/Sydney", "unit": "CNY_cents",
                "formula": "successful paid events minus successful refund events in each event's own local calendar month",
                "payment_status": "paid", "refund_status": "succeeded",
                "refund_channel": "original_order.channel_code",
                "event_time": {"orders": "paid_at", "refunds": "refunded_at"},
                "source_ids": ["business", "channels"], "missing_channel_policy": "reject",
                "approved_by": approved_by,
            }
            manifest = state["manifest"]
            candidate = {**manifest, "metric_contracts": [contract]}
            validation = validate_sources(candidate)
            if not validation["passed"]:
                raise ValueError("Sources do not satisfy the documented cash-event metric")
            manifest["metric_contracts"] = [contract]
            state["history"].append({"role": "user", "content": "我明确批准本次使用以下现金事件定义：" + json.dumps(contract, ensure_ascii=False)})
            self.store.save(state, "metric_approved", {"contract": contract})
            write_json(checked_run_dir(self.root, run_id) / "metric_approval.json", contract)
            return {"run_id": run_id, "status": "WAITING_INPUT", "metric_contract": contract,
                    "next": "resume this run with your answer; input data snapshots are unchanged"}

    def result(self, state):
        return {"run_id": state["run_id"], "status": state["status"], "mode": state["mode"],
                "run_dir": str(checked_run_dir(self.root, state["run_id"])),
                "report_path": state.get("report_path"), "answer": state.get("answer"),
                "question": state.get("pending_question"), "error": state.get("error"),
                "stop_reason": state.get("stop_reason"), "model_requests": state["model_requests"],
                "artifacts": state["artifacts"], "memory_draft": state.get("memory_draft"),
                "model": state.get("model"), "publications": state.get("publications", [])}

    def inspect(self, run_id):
        state = self.store.load(run_id)
        return {**self.result(state), "plan_version": state["plan_version"], "plan": state["plan"],
                "observed_facts": state["observed_facts"], "open_questions": state["open_questions"],
                "usage": state["usage"], "events": self.store.events(run_id)}
