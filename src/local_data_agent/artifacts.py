"""Bounded paths and durable, atomic JSON checkpoints (never store credentials)."""

import hashlib
import json
import os
import re
import uuid
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path


def jsonable(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    return value


def dumps(value):
    return json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(dumps(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def checked_run_dir(root, run_id):
    if not re.fullmatch(r"run_[a-f0-9]{32}", run_id):
        raise ValueError("Invalid run_id")
    base = (Path(root) / "var" / "workspaces").resolve()
    target = (base / run_id).resolve()
    if not target.is_relative_to(base):
        raise ValueError("Run path escapes workspace")
    return target


def verified_artifact(run_dir, artifact):
    root = Path(run_dir).resolve()
    path = Path(artifact["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Artifact is outside this run or missing")
    if file_hash(path) != artifact["sha256"]:
        raise ValueError("Artifact hash mismatch")
    return path

