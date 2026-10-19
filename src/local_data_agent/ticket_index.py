"""Bounded ticket-text evidence and verified ticket/order association.

Two independent, read-only concerns live here:

* ``TicketIndex`` indexes the *text* of registered JSONL ticket sources into
  bounded, exact ``retrieval.document_chunks`` snippets and answers a filtered
  search with at most one strongest snippet per ticket.
* ``associate`` inspects *only* registered SQLite sources to decide whether a
  ticket's raw ``order_id`` text can be verified against an explicit
  ``support_ticket_links`` row and/or exactly one registered orders source.

The association verdict is a *reference* fact only. It is never a financial
causality claim: a verified link says the ticket references an order that
exists in the registered orders source, nothing about money, refunds or fault.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .connectors import readonly_sqlite
from .retrieval import MAX_EVIDENCE_RECORDS, MAX_EVIDENCE_TEXT_BYTES, document_chunks

MAX_QUERY_CHARS = 2000
MAX_CHANNEL_CHARS = 128
MAX_RECORDS_PER_SOURCE = 10000
MAX_TEXT_CHARS = 100000
MAX_ASSOC_ROWS = 100000
MAX_LIMIT = 8

# Query words are OR'd and expanded with the Chinese/English aliases the tickets
# actually use, so 退款/退钱/重复扣款/多扣 all reach the same refund wording.
ALIAS_GROUPS = (
    ("退款", "退钱", "重复扣款", "多扣", "退费", "refund", "refunded", "refunds"),
    ("物流", "配送", "快递", "发货", "delivery", "shipping", "courier"),
    ("包装", "破损", "损坏", "破了", "damaged", "package", "packaging"),
    ("支付", "付款", "扣款", "payment", "paid", "pay"),
    ("工单", "客服", "投诉", "ticket", "support", "complaint"),
)
_ALIASES = {word: group for group in ALIAS_GROUPS for word in group}
_WORD_RE = re.compile(r"[0-9A-Za-z_\u4e00-\u9fff]+")
# Australia/Sydney is a DST zone: a fixed +10:00 offset is wrong between October
# and April, so the real tz database rule is used.
_SYDNEY = ZoneInfo("Australia/Sydney")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _words(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower())]


def _expand(query: str) -> list[str]:
    """Query words plus the sibling aliases of every alias group they touch."""
    words, seen = [], set()
    for word in _words(query):
        for candidate in (word, *_ALIASES.get(word, ())):
            if candidate not in seen:
                seen.add(candidate)
                words.append(candidate)
    return words


def _ticket_month(created_at):
    """(month, flag). ``month`` is the Australia/Sydney month, or None.

    Only a timestamp that carries an explicit offset is a real instant; a naive
    timestamp has no unambiguous Sydney month, so it is *flagged invalid* and
    never silently guessed into one.
    """
    if not isinstance(created_at, str) or not created_at.strip():
        return None, "invalid"
    raw = created_at.strip()
    try:
        stamp = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw)
    except ValueError:
        return None, "invalid"
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        return None, "invalid"
    return stamp.astimezone(_SYDNEY).strftime("%Y-%m"), "ok"


def _valid_month(month) -> bool:
    if not isinstance(month, str) or not re.fullmatch(r"\d{4}-\d{2}", month):
        return False
    return 1 <= int(month[5:7]) <= 12


def _valid_channel(channel_code) -> bool:
    return isinstance(channel_code, str) and len(channel_code) <= MAX_CHANNEL_CHARS


class TicketIndex:
    """Bounded, read-only ticket-text evidence over registered JSONL sources."""

    def __init__(self, manifest: dict):
        self.manifest = manifest
        self.sources = {s.get("source_id"): s for s in manifest.get("sources", [])}
        self.records = []           # one entry per bounded snippet, doc-global order
        self.tickets = {}           # (source_id, ticket_id) -> full canonical row
        self.population = {}        # source_id -> ticket count, always the FULL population
        self.eligible = {}          # source_id -> ticket count that survived validation
        self.source_hashes = {}
        self.populations = {}
        for source in manifest.get("sources", []):
            if source.get("source_id") is not None:
                self.populations.setdefault(source["source_id"], source)
        # Index first: the validated rows it produces are the association input, so
        # JSONL is parsed exactly once per source.
        self._build_index()
        validated_rows = [{"source_id": t["source_id"], "ticket_id": t["ticket_id"],
                           "raw_order_id": t["record"].get("order_id"),
                           "raw_ticket_channel": (t["record"].get("channel_code")
                                                  if isinstance(t["record"].get("channel_code"), str) else None),
                           "source_line": t["source_line"]}
                          for t in self._validated_tickets]
        # Association is computed over the WHOLE filtered set before any query, so
        # its counts are never derived from the returned sample.
        self._association = associate(manifest, validated_rows)
        self._assoc_by_ticket = {}
        for verdict in self._association["ticket_verdicts"]:
            self._assoc_by_ticket[(verdict["source_id"], verdict["ticket_id"])] = verdict

    # ---------------------------------------------------------------- indexing

    @staticmethod
    def _ticket_schema(source: dict) -> set:
        """The registered ``tickets`` table of a source, and only that table.

        Another registered table (orders, links, ...) never makes a source a
        ticket source, and a JSONL source registered without ``schema.tickets``
        is simply not a ticket-text source at all.
        """
        schema = source.get("schema")
        if not isinstance(schema, dict):
            return set()
        fields = schema.get("tickets")
        if not isinstance(fields, list):
            return set()
        return {c.get("name") for c in fields if isinstance(c, dict)}

    def _validate_source(self, source: dict) -> dict:
        source_id, path = source.get("source_id"), Path(source.get("path", ""))
        if not path.is_file():
            raise ValueError(f"Registered source file is missing: {source_id}")
        if not {"ticket_id", "text"} <= self._ticket_schema(source):
            raise ValueError(f"Source {source_id!r} has no registered ticket_id/text schema")
        # Streamed: the file's bytes are never held in memory as a whole.
        try:
            digest = _sha256_file(path)
        except OSError as error:
            raise ValueError(f"Registered source file is unreadable: {source_id}") from None
        declared = source.get("sha256")
        if declared is not None and declared != digest:
            raise ValueError(f"Source {source_id!r} sha256 does not match its registered hash")
        self.source_hashes[source_id] = digest
        return {"source_id": source_id, "path": path, "sha256": digest}

    def _read_tickets(self, prepared: dict) -> list[dict]:
        source_id = prepared["source_id"]
        tickets, count = [], 0
        # newline="" keeps CR/LF handling to Python's universal newlines while the
        # loop enumerates *physical* rows, so source_line is the real line number.
        with prepared["path"].open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                count += 1
                if count > MAX_RECORDS_PER_SOURCE:
                    raise ValueError(f"Source {source_id!r} exceeds {MAX_RECORDS_PER_SOURCE} records")
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    # Malformed JSON or a strict-decode failure rejects the whole
                    # source: never a silent skip.
                    raise ValueError(f"Source {source_id!r} line {line_number} is malformed JSON") from None
                if not isinstance(row, dict):
                    raise ValueError(f"Source {source_id!r} line {line_number} is not a JSON object")
                ticket_id, body = row.get("ticket_id"), row.get("text")
                if not isinstance(ticket_id, str) or not ticket_id.strip():
                    raise ValueError(f"Source {source_id!r} line {line_number} has an empty or non-string ticket_id")
                if not isinstance(body, str):
                    raise ValueError(f"Source {source_id!r} line {line_number} has a non-string text")
                if len(body) > MAX_TEXT_CHARS:
                    raise ValueError(f"Source {source_id!r} line {line_number} text exceeds {MAX_TEXT_CHARS} characters")
                # len(...) in characters over the FULL text, never the snippet extent.
                tickets.append({"source_id": source_id, "ticket_id": ticket_id, "text": body,
                                "line_count": max(1, len(body.splitlines())),
                                "record": row, "source_line": line_number})
        ids = [t["ticket_id"] for t in tickets]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Source {source_id!r} has duplicate ticket_id values")
        self.population[source_id] = count
        self.eligible[source_id] = len(tickets)
        return tickets

    def _build_index(self):
        vocabulary = {}
        self._validated_tickets = []
        for source in self.manifest.get("sources", []):
            if source.get("kind") != "jsonl":
                continue
            if not {"ticket_id", "text"} <= self._ticket_schema(source):
                # Not a registered ticket source (a CSV/other-data JSONL source, or
                # one registered without tickets): ignored, never a hard failure.
                continue
            prepared = self._validate_source(source)
            for ticket in self._read_tickets(prepared):
                self._validated_tickets.append(ticket)
                source_id, ticket_id = ticket["source_id"], ticket["ticket_id"]
                record = ticket["record"]
                tmonth, tflag = _ticket_month(record.get("created_at"))
                # Top level only: a nested source object is a different record shape
                # and must never be mistaken for the ticket's own channel.
                raw_channel = record.get("channel_code")
                channel = raw_channel if isinstance(raw_channel, str) else None
                # The canonical full record is hashed once; every snippet of the ticket
                # carries the same record_hash, so the hash always covers the full row.
                record_hash = _sha256_bytes(_canonical(record).encode("utf-8"))
                self.tickets[(source_id, ticket_id)] = {
                    "source_id": source_id, "ticket_id": ticket_id, "record_hash": record_hash,
                    "source_hash": prepared["sha256"], "source_line": ticket["source_line"],
                    "created_at": record.get("created_at"), "ticket_month": tmonth,
                    "ticket_month_valid": tflag == "ok", "ticket_channel": channel,
                    "text": ticket["text"], "raw": record}
                for chunk in document_chunks(ticket["text"], ticket_id):
                    key = f"ticket:{source_id}:{ticket_id}:{chunk['start_char']}-{chunk['end_char']}"
                    self.records.append({
                        "evidence_id": key, "type": "ticket", "source_id": source_id,
                        "ticket_id": ticket_id, "source_hash": prepared["sha256"],
                        "record_hash": record_hash, "source_line": ticket["source_line"],
                        "text": chunk["text"], "start_line": chunk["start_line"], "end_line": chunk["end_line"],
                        "line_count": ticket["line_count"],
                        "start_char": chunk["start_char"], "end_char": chunk["end_char"],
                        "created_at": record.get("created_at"), "ticket_month": tmonth,
                        "ticket_channel": channel, "_words": set(_words(chunk["text"]))})
                    for word in self.records[-1]["_words"]:
                        vocabulary.setdefault(word, set()).add(len(self.records) - 1)
        self._vocabulary = vocabulary
        self.index = sqlite3.connect(":memory:")
        self.index.execute("PRAGMA trusted_schema=OFF")
        self.fts = False
        try:
            self.index.execute("CREATE VIRTUAL TABLE chunks USING fts5(text, tokenize='trigram')")
            self.index.executemany("INSERT INTO chunks(rowid,text) VALUES(?,?)",
                                   [(i + 1, r["text"]) for i, r in enumerate(self.records)])
            self.fts = True
        except sqlite3.OperationalError:
            # No trigram FTS5 available: the explicit substring + Chinese alias
            # fallback below is the retrieval path, still bounded and deterministic.
            self.fts = False

    # ----------------------------------------------------------------- scoring

    def _score(self, record: dict, query: str, aliases: list[str]) -> int:
        # Literal query words first, then alias-OR hits; ties break on evidence id.
        body = record["text"].lower()
        literal = sum(1 for w in query if w and w in body)
        hits = sum(1 for w in aliases if w and w in body)
        return literal * 2 + hits

    def _candidates(self, query: str):
        """(best snippet, score) per ticket, over the full eligible population."""
        alias_words = _expand(query)
        scores = {}
        if self.fts:
            for word in alias_words:
                if len(word) < 3:
                    continue
                escaped = '"' + word.replace('"', '""') + '"'
                for (rowid,) in self.index.execute("SELECT rowid FROM chunks WHERE chunks MATCH ?", (escaped,)):
                    scores[rowid - 1] = scores.get(rowid - 1, 0) + 1
        literal_words = set(_words(query))
        best = {}
        for i, record in enumerate(self.records):
            score = scores.get(i, 0) * 5 + self._score(record, literal_words, alias_words)
            if score <= 0:
                continue
            key = (record["source_id"], record["ticket_id"])
            current = best.get(key)
            # One strongest snippet per ticket; ties resolve deterministically.
            if current is None or (-score, record["evidence_id"]) < (-current[1], current[0]["evidence_id"]):
                best[key] = (record, score)
        return sorted(best.values(), key=lambda pair: (-pair[1], pair[0]["evidence_id"]))

    # ------------------------------------------------------------------ search

    def search(self, query, source_id, month="", channel_code="", limit=6) -> dict:
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"Query must be at most {MAX_QUERY_CHARS} characters")
        source = self.sources.get(source_id)
        if source is None:
            raise ValueError("Unknown source identifier")
        if source.get("kind") != "jsonl" or not {"ticket_id", "text"} <= self._ticket_schema(source):
            # An unsupported kind, or a registered source that is not a ticket
            # source, is rejected here rather than at construction time.
            raise ValueError("Unsupported source kind for ticket search")
        if month != "" and not _valid_month(month):
            raise ValueError("Month must be empty or a valid YYYY-MM")
        if not _valid_channel(channel_code):
            raise ValueError(f"channel_code must be a string of at most {MAX_CHANNEL_CHARS} characters")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
            raise ValueError(f"limit must be an integer between 1 and {MAX_LIMIT}")

        # Metadata filters apply BEFORE ranking: a ticket outside the month or
        # channel window is not a candidate at all, never merely ranked lower.
        filtered = []
        for (sid, ticket_id), ticket in self.tickets.items():
            if sid != source_id:
                continue
            if month:
                if not ticket["ticket_month_valid"] or ticket["ticket_month"] != month:
                    continue
            if channel_code and ticket["ticket_channel"] != channel_code:
                continue
            filtered.append(ticket)
        by_key = {(t["source_id"], t["ticket_id"]): t for t in filtered}

        ranked = [(record, score) for record, score in self._candidates(query)
                  if (record["source_id"], record["ticket_id"]) in by_key]
        if query.strip():
            selected = ranked
        else:
            # An empty query is an explicit *filtered browse*, not missing context:
            # every surviving ticket is returned in deterministic order.
            selected = [({"evidence_id": self._first_evidence(t)}, 0) for t in
                        sorted(filtered, key=lambda t: t["ticket_id"])]

        truncated = len(selected) > limit
        chosen = selected[:limit]
        evidence, remaining, kept = [], MAX_EVIDENCE_TEXT_BYTES, 0
        for item, _score in chosen:
            # A ticket with no citable snippet of its own (an empty text has no
            # chunk at all) is NEVER given a fabricated quote: it is reported as
            # a truncation, so a missing citation is visible rather than silent.
            record = next((r for r in self.records if r["evidence_id"] == item["evidence_id"]), None)
            if record is None:
                truncated = True
                continue
            cost = len(record["text"].encode("utf-8"))
            if cost > remaining:
                truncated = True
                continue
            remaining -= cost
            kept += 1
            evidence.append(self._evidence(record))

        counts = {"verified": 0, "unverified": 0, "conflicting": 0, "unlinked": 0}
        for ticket in filtered:
            verdict = self._assoc_by_ticket.get((ticket["source_id"], ticket["ticket_id"]))
            status = verdict["status"] if verdict else "unlinked"
            counts[status] = counts.get(status, 0) + 1

        note = ("Association is a reference check only: a verified status means the ticket's order_id "
                "resolves to exactly one registered order row. It is not evidence of financial causality, "
                "fault, refund entitlement or any monetary claim, and no amount is aggregated.")
        # ``source_ticket_count`` is the WHOLE registered source population;
        # ``population_ticket_count`` is the population that survived the
        # requested filters, so matched <= population <= source always holds.
        source_ticket_count = self.population.get(source_id, 0)
        return {
            "query": query,
            "source_id": source_id,
            "filters": {"month": month, "channel_code": channel_code, "limit": limit,
                        "month_timezone": "Australia/Sydney", "channel_semantics":
                        "ticket.channel_code of the ticket record (the channel the ticket was filed "
                        "from), never an order-channel or delivery channel inferred from the text"},
            "evidence": evidence,
            "evidence_max": MAX_EVIDENCE_RECORDS,
            "evidence_text_bytes_max": MAX_EVIDENCE_TEXT_BYTES,
            "evidence_text_bytes": sum(len(e["text"].encode("utf-8")) for e in evidence),
            "source_ticket_count": source_ticket_count,
            "population_ticket_count": len(filtered),
            "filtered_ticket_count": len(filtered),
            "eligible_ticket_count": self.eligible.get(source_id, 0),
            "matched_ticket_count": len(ranked) if query.strip() else len(filtered),
            "returned_ticket_count": kept,
            "truncated": truncated,
            "missing_context": not evidence,
            "empty_query_browse": not query.strip(),
            "association_counts": counts,
            "association_note": note,
            "paths_exposed": False,
            "retrieval_mode": "fts5-trigram+aliases" if self.fts else "substring+chinese-aliases",
        }

    def _first_evidence(self, ticket: dict) -> str:
        for record in self.records:
            if record["source_id"] == ticket["source_id"] and record["ticket_id"] == ticket["ticket_id"]:
                return record["evidence_id"]
        return ""

    def _evidence(self, record: dict) -> dict:
        key = (record["source_id"], record["ticket_id"])
        verdict = self._assoc_by_ticket.get(key, {"status": "unlinked"})
        return {
            "evidence_id": record["evidence_id"], "type": "ticket",
            "source_id": record["source_id"], "ticket_id": record["ticket_id"],
            "source_hash": record["source_hash"], "record_hash": record["record_hash"],
            "source_line": record["source_line"], "text": record["text"],
            "start_line": record["start_line"], "end_line": record["end_line"],
            "line_count": record["line_count"],
            "start_char": record["start_char"], "end_char": record["end_char"],
            "created_at": record["created_at"], "ticket_month": record["ticket_month"],
            "ticket_channel": record["ticket_channel"],
            "association": _public_verdict(verdict),
        }

    def close(self):
        self.index.close()


def _public_verdict(verdict: dict) -> dict:
    """Every public verdict carries the same nullable keys, always.

    ``status`` is always present; the reference fields are ``None`` when they do
    not apply, so a caller never has to distinguish "absent" from "null". A
    ``verified`` verdict names its authority source id and hash; an ``unlinked``
    or ``unverified`` verdict may legitimately have neither. ``duplicate_warning``
    is a plain per-ticket bool, ``False`` unless this very ticket has one.
    """
    status = verdict.get("status") or "unlinked"
    return {
        "status": status,
        "order_id": verdict.get("order_id"),
        "order_channel": verdict.get("order_channel"),
        "link_origin": verdict.get("link_origin"),
        "association_source_id": verdict.get("association_source_id"),
        "association_source_hash": verdict.get("association_source_hash"),
        "duplicate_warning": bool(verdict.get("duplicate_warning", False)),
        "detail": verdict.get("detail"),
        "reference_only": True,
        "financial_causality_claimed": False,
    }


# ------------------------------------------------------------- association

def _registered_columns(source: dict) -> dict:
    return {table: [c.get("name") for c in fields] for table, fields in (source.get("schema") or {}).items()}


# The only columns association ever reads. Each entry is an *identifier* that
# must be registered for that table before it may appear in the SQL, and it is
# quoted; nothing else -- and in particular no monetary payload such as
# ``paid_cents`` -- is ever selected.
ASSOC_COLUMNS = {
    "orders": ("order_id", "channel_code"),
    "support_ticket_links": ("ticket_id", "order_id"),
}


def _orders_sources(manifest: dict) -> list[dict]:
    eligible = []
    for source in manifest.get("sources", []):
        if source.get("kind") != "sqlite":
            continue
        columns = _registered_columns(source)
        # Only registered SQLite sources with a registered orders table and the two
        # columns a verification needs can ever be an orders authority.
        orders = columns.get("orders")
        if orders and set(ASSOC_COLUMNS["orders"]) <= set(orders):
            eligible.append({"source": source, "columns": columns})
    return eligible


def _read_rows(source: dict, table: str, columns: list) -> list[dict]:
    """Read only the registered identifier columns this table's check needs.

    ``columns`` is the registered schema for ``table``. The select list is
    intersected with :data:`ASSOC_COLUMNS` and re-validated against the
    registered names, so an unregistered or unexpected column can never be
    reaching the SQL -- and ``SELECT *`` (which would drag the whole monetary
    payload into memory) is never issued.
    """
    required = ASSOC_COLUMNS.get(table)
    if not required:
        raise ValueError(f"Table {table!r} is not an association reference table")
    registered = {c for c in columns if isinstance(c, str)}
    if not set(required) <= registered:
        raise ValueError(f"Source {source.get('source_id')!r} table {table!r} lacks a registered "
                         f"{'/'.join(sorted(set(required) - registered))} column")
    selected = [c for c in required if c in registered]
    quoted_table = table.replace('"', '""')
    select_list = ", ".join('"' + c.replace('"', '""') + '"' for c in selected)
    with closing(readonly_sqlite(Path(source["path"]))) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute(
            f'SELECT {select_list} FROM "{quoted_table}" LIMIT {MAX_ASSOC_ROWS + 1}')]
    if len(rows) > MAX_ASSOC_ROWS:
        raise ValueError(f"Source {source.get('source_id')!r} table {table!r} exceeds {MAX_ASSOC_ROWS} rows")
    return rows


def _reparse_ticket_rows(manifest: dict) -> list[dict]:
    """Standalone fallback for callers with no index: bounded, strict, streaming.

    It re-validates exactly like the index loader -- file present, registered
    hash honoured, strict UTF-8, strict JSON, object rows, non-empty string
    ``ticket_id``/string ``text``, no duplicates, ``MAX_RECORDS_PER_SOURCE`` and
    ``MAX_TEXT_CHARS`` caps -- and a violation rejects the whole source instead
    of silently continuing or breaking at the first bad record. The file is
    streamed line by line, never read as one whole buffer. A blank line is the
    one thing that may be skipped.
    """
    ticket_rows = []
    for source in manifest.get("sources", []):
        if source.get("kind") != "jsonl":
            continue
        if not {"ticket_id", "text"} <= TicketIndex._ticket_schema(source):
            continue
        source_id = source.get("source_id")
        path = Path(source.get("path", ""))
        if not path.is_file():
            raise ValueError(f"Registered source file is missing: {source_id}")
        try:
            digest = _sha256_file(path)
        except OSError:
            raise ValueError(f"Registered source file is unreadable: {source_id}") from None
        declared = source.get("sha256")
        if declared is not None and declared != digest:
            raise ValueError(f"Source {source_id!r} sha256 does not match its registered hash")
        rows, seen, count = [], set(), 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue  # the only tolerated skip: a blank line
                count += 1
                if count > MAX_RECORDS_PER_SOURCE:
                    raise ValueError(f"Source {source_id!r} exceeds {MAX_RECORDS_PER_SOURCE} records")
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    raise ValueError(f"Source {source_id!r} line {line_number} is malformed JSON") from None
                if not isinstance(row, dict):
                    raise ValueError(f"Source {source_id!r} line {line_number} is not a JSON object")
                ticket_id, body = row.get("ticket_id"), row.get("text")
                if not isinstance(ticket_id, str) or not ticket_id.strip():
                    raise ValueError(f"Source {source_id!r} line {line_number} has an empty or non-string ticket_id")
                if not isinstance(body, str):
                    raise ValueError(f"Source {source_id!r} line {line_number} has a non-string text")
                if len(body) > MAX_TEXT_CHARS:
                    raise ValueError(f"Source {source_id!r} line {line_number} text exceeds {MAX_TEXT_CHARS} characters")
                if ticket_id in seen:
                    raise ValueError(f"Source {source_id!r} has duplicate ticket_id values")
                seen.add(ticket_id)
                rows.append({"source_id": source_id, "ticket_id": ticket_id,
                             "raw_order_id": row.get("order_id"),
                             "raw_ticket_channel": (row.get("channel_code")
                                                    if isinstance(row.get("channel_code"), str) else None),
                             "source_line": line_number})
        ticket_rows.extend(rows)
    return ticket_rows


def _ticket_channel_conflicts(raw_channel, order_channel) -> bool:
    """A raw ticket channel that disagrees with the resolved order's channel.

    Only two real, non-empty strings can disagree; a ticket with no channel of
    its own is not a conflict, it is simply silent about the channel.
    """
    return (isinstance(raw_channel, str) and raw_channel != "" and raw_channel != order_channel)


def associate(manifest: dict, validated_ticket_rows: list[dict] | None = None) -> dict:
    """Reference-only ticket/order association over registered SQLite sources.

    Nothing here reads a caller path, invents a link from free text, classifies a
    ticket with a model, or aggregates amounts. A ``verified`` status means the
    ticket's explicit/raw order reference resolves to exactly one registered order.

    ``validated_ticket_rows`` are the already-validated ticket rows of the index
    (``source_id``/``ticket_id``/``raw_order_id``/``raw_ticket_channel``/
    ``source_line``). When they are supplied the JSONL is *not* parsed again; the
    standalone fallback parse below is retained for callers that have no index.
    """
    orders_sources = _orders_sources(manifest)
    if validated_ticket_rows is not None:
        ticket_rows = list(validated_ticket_rows)
    else:
        ticket_rows = _reparse_ticket_rows(manifest)

    # More than one eligible orders authority: never pick one arbitrarily.
    ambiguous = len(orders_sources) > 1
    orders_by_id, ambiguous_ids, links_by_ticket = {}, set(), {}
    duplicate_link_tickets = set()
    association_source_id = association_source_hash = None
    link_table = False
    if len(orders_sources) == 1:
        source = orders_sources[0]["source"]
        columns = orders_sources[0]["columns"]
        association_source_id = source.get("source_id")
        # The hash of the file actually read. A registered hash is verified
        # against it and a mismatch is a hard failure; either way the *actual*
        # hash is what the verdicts report, never a merely declared one.
        try:
            actual_hash = _sha256_file(Path(source["path"]))
        except OSError:
            raise ValueError(f"Orders source {association_source_id!r} is unreadable") from None
        registered_hash = source.get("sha256")
        if registered_hash is not None and registered_hash != actual_hash:
            raise ValueError(f"Source {association_source_id!r} sha256 does not match its registered hash")
        association_source_hash = actual_hash
        for row in _read_rows(source, "orders", columns["orders"]):
            order_id = row.get("order_id")
            if order_id is None:
                continue
            key = str(order_id)
            if key in orders_by_id:
                # The same order_id resolving to several distinct rows is
                # ambiguous: the reference is never arbitrated by keeping the
                # last (or the first) row, so this key can never be verified.
                if _canonical(orders_by_id[key]) != _canonical(row):
                    ambiguous_ids.add(key)
            else:
                orders_by_id[key] = row
        if "support_ticket_links" in columns:
            link_columns = columns["support_ticket_links"]
            if "ticket_id" in link_columns and "order_id" in link_columns:
                link_table = True
                for row in _read_rows(source, "support_ticket_links", link_columns):
                    ticket_id = row.get("ticket_id")
                    order_id = row.get("order_id")
                    if ticket_id is None or order_id is None:
                        continue
                    bucket = links_by_ticket.setdefault(str(ticket_id), [])
                    if str(order_id) in bucket:
                        # A repeated identical link is deduplicated, but the
                        # repeat is reported against THIS ticket only, so the
                        # warning never bleeds into unrelated verdicts.
                        duplicate_link_tickets.add(str(ticket_id))
                        continue
                    bucket.append(str(order_id))

    verdicts = []
    ticket_namespaces = Counter(t['ticket_id'] for t in ticket_rows)
    for ticket in ticket_rows:
        source_id, ticket_id = ticket["source_id"], ticket["ticket_id"]
        raw_ticket_channel = ticket.get("raw_ticket_channel")
        raw_reference = ticket["raw_order_id"]
        raw_reference = str(raw_reference) if isinstance(raw_reference, (str, int)) and str(raw_reference) else None
        raw_ambiguous = raw_reference is not None and raw_reference in ambiguous_ids
        raw_resolves = raw_reference is not None and raw_reference in orders_by_id and not raw_ambiguous
        raw_channel = orders_by_id[raw_reference].get("channel_code") if raw_resolves else None
        # Every verdict for this ticket starts from its own duplicate-link fact;
        # a verdict that is not about this ticket can never inherit it.
        per_ticket_duplicate = ticket_id in duplicate_link_tickets

        def verdict(status, _sid=source_id, _tid=ticket_id, _dup=per_ticket_duplicate, **fields):
            # Bind this ticket's identity as defaults so a later iteration can
            # never retroactively change an already-appended verdict.
            entry = {"status": status, "source_id": _sid, "ticket_id": _tid,
                     "order_id": None, "order_channel": None, "link_origin": None,
                     "association_source_id": None, "association_source_hash": None,
                     "duplicate_warning": _dup, "detail": None}
            entry.update(fields)
            verdicts.append(entry)

        if ambiguous:
            no_link_table = not any("support_ticket_links" in s["columns"] for s in orders_sources)
            if raw_reference is None and no_link_table:
                verdict("unlinked", detail="no raw reference and no registered link table")
            else:
                verdict("unverified", detail="multiple eligible orders sources; no authority selected")
            continue
        explicit = links_by_ticket.get(ticket_id, [])
        if explicit and ticket_namespaces[ticket_id] > 1:
            verdict("unverified", detail="explicit ticket ID is ambiguous across registered ticket sources")
            continue
        if len(set(explicit)) > 1:
            verdict("conflicting", detail="explicit links name multiple orders",
                    link_origin="explicit", association_source_id=association_source_id,
                    association_source_hash=association_source_hash)
            continue
        if explicit and raw_reference is not None and explicit[0] != raw_reference:
            # Ticket raw order_id must AGREE with the explicit link.
            verdict("conflicting", detail="raw order_id disagrees with explicit link",
                    link_origin="explicit", order_id=explicit[0],
                    association_source_id=association_source_id,
                    association_source_hash=association_source_hash)
            continue
        if explicit:
            order_id = explicit[0]
            if order_id in ambiguous_ids:
                verdict("unverified", detail="linked order_id is ambiguous in orders",
                        order_id=order_id, link_origin="explicit",
                        association_source_id=association_source_id,
                        association_source_hash=association_source_hash)
                continue
            if order_id not in orders_by_id:
                verdict("unverified", detail="linked order does not exist",
                        order_id=order_id, link_origin="explicit",
                        association_source_id=association_source_id,
                        association_source_hash=association_source_hash)
                continue
            channel = orders_by_id[order_id].get("channel_code")
            if _ticket_channel_conflicts(raw_ticket_channel, channel):
                verdict("conflicting", detail="raw ticket channel and linked order channel disagree",
                        order_id=order_id, order_channel=channel, link_origin="explicit",
                        association_source_id=association_source_id,
                        association_source_hash=association_source_hash)
                continue
            verdict("verified", order_id=order_id, order_channel=channel, link_origin="explicit",
                    association_source_id=association_source_id,
                    association_source_hash=association_source_hash)
            continue
        if raw_reference is None:
            # Neither explicit nor raw reference: never invented from a text mention.
            verdict("unlinked", detail="no explicit link and no raw order_id")
            continue
        if len(orders_sources) == 0:
            verdict("unverified", detail="raw order_id but no eligible orders source",
                    order_id=raw_reference, link_origin="raw")
            continue
        if raw_ambiguous:
            verdict("unverified", detail="raw order_id is ambiguous in orders",
                    order_id=raw_reference, link_origin="raw",
                    association_source_id=association_source_id,
                    association_source_hash=association_source_hash)
            continue
        if not raw_resolves:
            verdict("unverified", detail="raw order_id does not exist in orders",
                    order_id=raw_reference, link_origin="raw",
                    association_source_id=association_source_id,
                    association_source_hash=association_source_hash)
            continue
        if _ticket_channel_conflicts(raw_ticket_channel, raw_channel):
            verdict("conflicting", detail="raw ticket channel and resolved order channel disagree",
                    order_id=raw_reference, order_channel=raw_channel, link_origin="raw",
                    association_source_id=association_source_id,
                    association_source_hash=association_source_hash)
            continue
        verdict("verified", order_id=raw_reference, order_channel=raw_channel, link_origin="raw",
                association_source_id=association_source_id,
                association_source_hash=association_source_hash)

    return {"orders_source_ids": [s["source"].get("source_id") for s in orders_sources],
            "ambiguous_orders_sources": ambiguous,
            "explicit_link_table_used": link_table,
            "association_source_id": association_source_id,
            "association_source_hash": association_source_hash,
            "duplicate_warning": bool(duplicate_link_tickets),
            "duplicate_warning_ticket_ids": sorted(duplicate_link_tickets),
            "ticket_verdicts": verdicts,
            "reference_rows_only": ["orders", "support_ticket_links"],
            "columns_read": {t: sorted(c) for t, c in ASSOC_COLUMNS.items()},
            "financial_amounts_aggregated": False}
