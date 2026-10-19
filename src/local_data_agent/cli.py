"""Small synchronous CLI: model calls and all analysis run on the local host."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _redact_text(message: str) -> str:
    for name in ("DATA_AGENT_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            message = message.replace(value, "[REDACTED]")
    message = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "[REDACTED]", message)
    return message


def _safe_error(error: Exception) -> str:
    return _redact_text(str(error))[:2000]


def _redact_result(value):
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, dict):
        return {key: _redact_result(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_result(item) for item in value]
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="本地 Data Agent：发现、规划、分析与可追溯回答")
    result.add_argument("--project-root", type=Path, default=default_project_root())
    commands = result.add_subparsers(dest="command", required=True)

    def command(name: str, help_text: str):
        item = commands.add_parser(name, help=help_text)
        item.add_argument("--project-root", type=Path, default=argparse.SUPPRESS)
        return item

    doctor = command("doctor", "检查本地环境；网络探针需显式 --probe")
    doctor.add_argument("--probe", action="store_true")
    bootstrap = command("bootstrap", "创建合成数据，拒绝覆盖已有数据")
    bootstrap.add_argument("--fixture", default="cash-net-v1")
    bootstrap.add_argument("--seed", type=int, default=0)
    bootstrap.add_argument("--variant")
    run = command("run", "运行回放或真实模型分析")
    run.add_argument("--mode", choices=("replay", "live"), default="replay")
    run.add_argument("--question", default="按现金净额口径比较八月和九月，并分析下降")
    run.add_argument("--case", default="cash-net-v1")
    run.add_argument("--fixture", default="cash-net-v1")
    resume = command("resume", "按 run_id 回答追问或继续分析")
    resume.add_argument("run_id")
    resume.add_argument("--answer")
    resume.add_argument("--question")
    inspect = command("inspect", "查看本地运行状态与产物索引")
    inspect.add_argument("run_id")
    metric_approval = command("approve-metric", "用户明确批准暂停运行的演示现金事件口径；不由模型调用")
    metric_approval.add_argument("run_id")
    metric_approval.add_argument("--metric", choices=("CASH_NET_v1",), required=True)
    metric_approval.add_argument("--approved-by", required=True)
    memory = command("memory", "列出或明确批准记忆卡")
    memory_commands = memory.add_subparsers(dest="memory_command", required=True)
    listing = memory_commands.add_parser("list")
    listing.add_argument("--status", choices=("DRAFT", "APPROVED", "STALE", "REJECTED"))
    approval = memory_commands.add_parser("approve")
    approval.add_argument("memory_id")
    approval.add_argument("--approved-by", required=True)
    evaluate = command("evaluate", "独立评价；回放结果不等于模型智能表现")
    evaluate.add_argument("--suite", choices=("deterministic", "agent-live"), default="deterministic")
    evaluate.add_argument("--repeats", type=int, default=3)
    evaluate.add_argument("--cases", help="Optional comma-separated live case IDs, e.g. L01,L02")
    return result


def dispatch(args: argparse.Namespace) -> dict:
    root = args.project_root.resolve()
    if args.command == "doctor":
        from .doctor import diagnose
        return diagnose(root, probe=args.probe)
    if args.command == "bootstrap":
        from .fixtures import bootstrap
        registry = bootstrap(root, fixture=args.fixture, seed=args.seed, variant=args.variant)
        return {"status": "succeeded", "registry": registry}
    if args.command in {"run", "resume", "inspect", "approve-metric"}:
        from .runner import AgentRunner
        runner = AgentRunner(root)
        if args.command == "run":
            return runner.run(args.question, mode=args.mode, case=args.case, fixture=args.fixture)
        if args.command == "resume":
            return runner.resume(args.run_id, answer=args.answer, question=args.question)
        if args.command == "approve-metric":
            return runner.approve_metric(args.run_id, args.metric, args.approved_by)
        return runner.inspect(args.run_id)
    if args.command == "memory":
        from .memory import MemoryStore
        memory = MemoryStore(root)
        if args.memory_command == "list":
            return {"status": "succeeded", "cards": memory.list_cards(status=args.status)}
        return {"status": "succeeded", "card": memory.approve(args.memory_id, args.approved_by)}
    if args.command == "evaluate":
        if not 1 <= args.repeats <= 10:
            raise ValueError("repeats must be between 1 and 10")
        from .evaluation import evaluate
        kwargs = {"suite": args.suite, "repeats": args.repeats}
        if args.cases:
            cases = [item.strip() for item in args.cases.split(",")]
            if len(cases) > 24 or any(not re.fullmatch(r"L\d{2}", item) for item in cases) or len(set(cases)) != len(cases):
                raise ValueError("cases must be a unique comma-separated list of live IDs such as L01,L02")
            if args.suite != "agent-live":
                raise ValueError("--cases only applies to agent-live")
            kwargs["cases"] = cases
        return evaluate(root, **kwargs)
    raise ValueError("Unknown command")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
        if not isinstance(result, dict):
            raise RuntimeError("Command returned an invalid result")
    except (Exception, KeyboardInterrupt) as error:
        status = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        if type(error).__name__ == "ConfigurationError" and type(error).__module__.endswith("model_client"):
            status = "unverified"
        result = {"status": status,
                  "error_type": type(error).__name__, "error": _safe_error(error)}
    print(json.dumps(_redact_result(result), ensure_ascii=False, indent=2, default=str))
    if str(result.get("status", "")).lower() in {"failed", "cancelled", "unverified", "blocked"}:
        return 1
    if result.get("passed") is False:
        return 1
    return 0
