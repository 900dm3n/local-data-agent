"""SQLite checkpoints and call-id idempotency; no DB lock while calling models."""

import ctypes
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import checked_run_dir, dumps, write_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class RunBusy(RuntimeError):
    """Concurrent execution cannot overwrite the active run's state."""


def pid_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        code = ctypes.c_ulong()
        try:
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class StateStore:
    def __init__(self, project_root):
        self.root = Path(project_root).resolve()
        self.path = self.root / "var" / "state.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                    owner_pid INTEGER, lease_until REAL, attempt_id TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL, kind TEXT NOT NULL,
                    payload TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tool_calls (
                    run_id TEXT NOT NULL, call_id TEXT NOT NULL,
                    name TEXT NOT NULL, arguments TEXT NOT NULL,
                    output TEXT, status TEXT NOT NULL,
                    PRIMARY KEY(run_id,call_id)
                );
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        connection.execute("PRAGMA busy_timeout=15000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create(self, question, mode, fixture, case):
        run_id = "run_" + uuid.uuid4().hex
        run_dir = checked_run_dir(self.root, run_id)
        run_dir.mkdir(parents=True)
        payload = {
            "run_id": run_id, "task_id": "task_" + uuid.uuid4().hex,
            "question": question, "mode": mode, "fixture": fixture, "case": case,
            "status": "QUEUED", "created_at": utc_now(), "updated_at": utc_now(),
            "history": [], "plan": [], "plan_version": 0, "artifacts": {},
            "claims": [], "evidence": [], "metric_binding": None,
            "observed_facts": [], "open_questions": [], "usage": [],
            "model_requests": 0, "stop_reason": None, "repair_counts": {},
        }
        with self.connect() as connection:
            connection.execute("INSERT INTO runs(run_id,payload) VALUES (?,?)", (run_id, dumps(payload)))
            self._event(connection, run_id, "created", {"mode": mode, "question": question})
        self.export(payload)
        return payload

    def load(self, run_id):
        checked_run_dir(self.root, run_id)
        with self.connect() as connection:
            row = connection.execute("SELECT payload FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            raise ValueError("Unknown run_id")
        return json.loads(row[0])

    def _event(self, connection, run_id, kind, payload):
        connection.execute("INSERT INTO events(run_id,kind,payload,created_at) VALUES (?,?,?,?)",
                           (run_id, kind, dumps(payload), utc_now()))

    def save(self, payload, kind="checkpoint", event=None):
        payload["updated_at"] = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT attempt_id,owner_pid FROM runs WHERE run_id=?", (payload["run_id"],)).fetchone()
            if current and current[1] and (current[1] != os.getpid() or current[0] != payload.get("attempt_id")):
                raise RunBusy("Attempt no longer owns this run")
            connection.execute("UPDATE runs SET payload=?,lease_until=? WHERE run_id=?",
                               (dumps(payload), time.time() + 120, payload["run_id"]))
            self._event(connection, payload["run_id"], kind, event or {"status": payload["status"]})
        self.export(payload)

    def export(self, payload):
        run_dir = checked_run_dir(self.root, payload["run_id"])
        fields = ("question", "status", "metric_binding", "plan_version", "observed_facts",
                  "open_questions", "model_requests", "usage", "stop_reason", "repair_counts")
        write_json(run_dir / "agent_state.json", {key: payload.get(key) for key in fields})
        write_json(run_dir / "plan.json", {"version": payload["plan_version"], "steps": payload["plan"]})
        # SQLite is authoritative; exporting this checkpoint again is safe after a crash.
        write_json(run_dir / "checkpoint.json", payload)
        transcript = "\n".join(dumps(item) for item in payload["history"]) + "\n"
        path = run_dir / "transcript.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text(transcript, encoding="utf-8")
        os.replace(temporary, path)

    @contextmanager
    def acquire(self, run_id):
        attempt = "attempt_" + uuid.uuid4().hex
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT owner_pid,lease_until FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if not row:
                raise ValueError("Unknown run_id")
            if row[0] and row[1] and row[1] > time.time() and pid_alive(row[0]):
                raise RunBusy("This run is already being executed")
            connection.execute("UPDATE runs SET owner_pid=?,lease_until=?,attempt_id=? WHERE run_id=?",
                               (os.getpid(), time.time() + 120, attempt, run_id))
        try:
            yield attempt
        finally:
            with self.connect() as connection:
                connection.execute("UPDATE runs SET owner_pid=NULL,lease_until=NULL WHERE run_id=? AND attempt_id=?",
                                   (run_id, attempt))

    def start_call(self, run_id, call_id, name, arguments):
        encoded = dumps(arguments)
        with self.connect() as connection:
            row = connection.execute("SELECT name,arguments,output,status FROM tool_calls WHERE run_id=? AND call_id=?",
                                     (run_id, call_id)).fetchone()
            if row:
                if row[0] != name or json.loads(row[1]) != arguments:
                    raise ValueError("call_id reused with different arguments")
                return json.loads(row[2]) if row[3] == "SUCCEEDED" else None
            connection.execute("INSERT INTO tool_calls VALUES (?,?,?,?,NULL,'RUNNING')",
                               (run_id, call_id, name, encoded))
        return None

    def finish_call(self, run_id, call_id, output, state=None):
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if state is not None:
                owner = connection.execute("SELECT attempt_id,owner_pid FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if owner and owner[1] and (owner[1] != os.getpid() or owner[0] != state.get("attempt_id")):
                    raise RunBusy("Attempt no longer owns this run")
            connection.execute("UPDATE tool_calls SET output=?,status='SUCCEEDED' WHERE run_id=? AND call_id=?",
                               (dumps(output), run_id, call_id))
            if state is not None:
                state["updated_at"] = utc_now()
                connection.execute("UPDATE runs SET payload=?,lease_until=? WHERE run_id=?",
                                   (dumps(state), time.time() + 120, run_id))
            self._event(connection, run_id, "tool_result", {"call_id": call_id, "output": output})

    def events(self, run_id):
        with self.connect() as connection:
            rows = connection.execute("SELECT event_id,kind,payload,created_at FROM events WHERE run_id=? ORDER BY event_id",
                                      (run_id,)).fetchall()
        return [{"event_id": r[0], "kind": r[1], "payload": json.loads(r[2]), "created_at": r[3]} for r in rows]

