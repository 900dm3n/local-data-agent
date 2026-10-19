"""Read-only environment checks, with an explicitly requested model probe."""
from __future__ import annotations

from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sqlite3


REQUIRED_PACKAGES = ("pydantic", "duckdb", "sqlglot", "pyarrow", "tzdata", "openai")


def diagnose(project_root: Path, probe: bool = False) -> dict:
    packages = {}
    for name in REQUIRED_PACKAGES:
        try:
            packages[name] = {"installed": True, "version": metadata.version(name)}
        except metadata.PackageNotFoundError:
            packages[name] = {"installed": False}
    with sqlite3.connect(":memory:") as db:
        try:
            db.execute("CREATE VIRTUAL TABLE docs USING fts5(body)")
            fts5 = True
        except sqlite3.DatabaseError:
            fts5 = False
    settings = {}
    local_config = project_root / "config.local.json"
    if local_config.is_file():
        parsed = json.loads(local_config.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("Local configuration must be an object")
        settings = parsed
    model_name = os.environ.get("DATA_AGENT_MODEL") or settings.get("model")
    key_file = os.environ.get("DATA_AGENT_CONFIG_FILE") or settings.get("key_file")
    configured = bool(model_name and (os.environ.get("DATA_AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY")
                                    or key_file and Path(key_file).is_file()))
    result = {
        "status": "succeeded" if fts5 and all(p["installed"] for p in packages.values()) else "failed",
        "python": platform.python_version(), "platform": platform.system(),
        "project_root": str(project_root.resolve()), "sqlite": sqlite3.sqlite_version,
        "fts5": fts5, "packages": packages,
        "model": {"configured": configured, "api_mode": os.environ.get("DATA_AGENT_API_MODE") or settings.get("api_mode", "responses"),
                  "verification": "unverified", "network_called": False},
    }
    if probe:
        if not configured:
            result["status"] = "unverified"
            result["model"]["reason"] = "Set DATA_AGENT_MODEL and DATA_AGENT_API_KEY or OPENAI_API_KEY; no replay fallback."
        else:
            from .model_client import create_model_client
            model = create_model_client(project_root)
            tools = [{"type": "function", "name": "local_probe", "description": "Verify one local tool request.",
                      "parameters": {"type": "object", "properties": {"value": {"type": "string"}},
                                     "required": ["value"], "additionalProperties": False}, "strict": True}]
            response = model.respond([{"role": "user", "content": "Call local_probe with value 'ready'. Do not answer directly."}], tools)
            calls = response.get("tool_calls", [])
            valid = False
            for call in calls:
                arguments = call.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        arguments = None
                if call.get("name") == "local_probe" and arguments == {"value": "ready"}:
                    valid = True
            result["model"].update({"verification": "verified" if valid else "failed", "network_called": True,
                                    "probe_scope": "function request only; agent analysis evaluated separately",
                                    "tool_call_received": valid, "usage": response.get("usage", {})})
            if not valid:
                result["status"] = "failed"
    return result
