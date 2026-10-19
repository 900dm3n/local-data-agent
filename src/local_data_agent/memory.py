"""Explicit user-approved memory; model tools cannot approve cards."""
from __future__ import annotations
import json
from contextlib import contextmanager
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


class MemoryStore:
    def __init__(self, project_root: Path):
        folder = Path(project_root) / "var"
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / "memory.sqlite"
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS memory_cards(memory_id TEXT PRIMARY KEY,run_id TEXT,content TEXT,scope TEXT,status TEXT,version INTEGER,created_at TEXT,approved_by TEXT,approved_at TEXT)")

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def draft(self, run_id, content, scope) -> dict:
        if not isinstance(content, (str, dict)) or not isinstance(scope, (str, dict)):
            raise ValueError("Memory content and scope must be JSON objects or strings")
        if len(json.dumps(content)) > 20000:
            raise ValueError("Memory exceeds size limit")
        card = {"memory_id": "mem_" + uuid.uuid4().hex, "run_id": run_id, "content": content, "scope": scope, "status": "DRAFT", "version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "approved_by": None, "approved_at": None}
        with self._db() as db:
            db.execute("INSERT INTO memory_cards VALUES(?,?,?,?,?,?,?,?,?)", (card["memory_id"], run_id, json.dumps(content, ensure_ascii=False), json.dumps(scope, ensure_ascii=False), "DRAFT", 1, card["created_at"], None, None))
        return card

    def approve(self, memory_id, approved_by) -> dict:
        if not isinstance(approved_by, str) or not approved_by.strip() or approved_by.lower() in {"agent", "model", "assistant"}:
            raise ValueError("An explicit human approver is required")
        with self._db() as db:
            row = db.execute("SELECT status FROM memory_cards WHERE memory_id=?", (memory_id,)).fetchone()
            if not row:
                raise ValueError("Unknown memory identifier")
            if row[0] != "DRAFT":
                raise ValueError("Only DRAFT cards can be approved")
            db.execute("UPDATE memory_cards SET status='APPROVED',approved_by=?,approved_at=? WHERE memory_id=?", (approved_by, datetime.now(timezone.utc).isoformat(), memory_id))
        return next(c for c in self.list_cards() if c["memory_id"] == memory_id)

    def list_cards(self, status=None) -> list:
        if status and status not in {"DRAFT", "APPROVED", "STALE", "REJECTED"}:
            raise ValueError("Invalid memory status")
        with self._db() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM memory_cards" + (" WHERE status=?" if status else "") + " ORDER BY created_at", (status,) if status else ()).fetchall()
        cards = [dict(r) for r in rows]
        for card in cards:
            card["content"], card["scope"] = json.loads(card["content"]), json.loads(card["scope"])
        return cards

    def recall(self, query, scope) -> list:
        terms = str(query).lower().split()
        cards = self.list_cards("APPROVED")
        matches = []
        for card in cards:
            if card["scope"] != scope:
                continue
            text = json.dumps(card["content"], ensure_ascii=False).lower()
            score = sum(t in text for t in terms)
            if score or not terms:
                matches.append(dict(card, relevance_score=score))
        # Return conflicting applicable cards together: never choose the latest silently.
        return sorted(matches, key=lambda c: (-c["relevance_score"], c["created_at"]))[:12]
