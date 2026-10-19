"""Bounded local schema, value and document evidence retrieval."""
from __future__ import annotations
import csv
from contextlib import closing
from bisect import bisect_right
import hashlib
import json
import re
import sqlite3
from pathlib import Path

from .connectors import readonly_sqlite

ALIASES = {"收入": "现金净额 支付 退款 orders refunds", "净额": "CASH_NET_v1 paid_cents refund_cents", "退款": "refunds refunded_at refund_cents", "订单": "orders paid_at paid_cents", "官网": "web channels channel_name", "移动端": "app channels channel_name", "渠道": "channels channel_code channel_name", "工单": "tickets text ticket_id", "时区": "Australia/Sydney UTC 本地月"}

# Bounded evidence contract: an answer never receives more than a handful of
# contiguous exact snippets, and their combined original text stays inside a
# byte budget. Numeric data and approved contracts are never rewritten.
MAX_EVIDENCE_RECORDS = 8
MAX_EVIDENCE_TEXT_BYTES = 8000
MAX_CHUNKS_PER_DOCUMENT = 2
CHUNK_MAX_LINES = 12
CHUNK_MAX_CHARS = 1200
CHUNK_MAX_SPAN_LINES = 12  # trailing window: a chunk may not extend past starts[start_index + 12]


def source_rows(source: dict, table=None) -> list[dict]:
    path = Path(source["path"])
    if source["kind"] == "sqlite":
        if not table or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError("Registered table required")
        schema = source.get("schema", {})
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            tables = schema or {r[0]: [] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if table not in tables:
                raise ValueError("Unknown table")
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute(f'SELECT * FROM "{table}" LIMIT 100001')]
    if source["kind"] == "csv":
        with path.open(encoding="utf-8", newline="") as f:
            rows = []
            for i, row in enumerate(csv.DictReader(f)):
                if i > 100000:
                    raise ValueError("Source exceeds validation limit")
                rows.append(dict(row))
            return rows
    if source["kind"] == "jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 100000:
                    raise ValueError("Source exceeds validation limit")
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if source["kind"] == "parquet":
        import pyarrow.parquet as pq
        if pq.ParquetFile(path).metadata.num_rows > 100000:
            raise ValueError("Source exceeds validation limit")
        return pq.read_table(path).to_pylist()
    raise ValueError("Unsupported source kind")


def document_chunks(text: str, doc_id: str, max_chars: int = CHUNK_MAX_CHARS, max_lines: int = CHUNK_MAX_LINES) -> list[dict]:
    """Deterministic, nonoverlapping, contiguous coverage of the complete original text.

    The text is walked with a document-global cursor. Each chunk starts at the line
    containing the cursor and may not extend past the next ``max_lines`` original line
    starts, so a chunk never covers more than ``max_lines`` original lines; it also may
    not exceed ``max_chars`` characters (nor ``max_lines``). When the original text
    still has a newline inside that window, the last newline of the window is preferred
    so ordinary whole lines stay together -- unless it would fall exactly before the
    first line boundary, which would leave a leading fragment; the segment then ends at
    the earliest line boundary instead. A line with no newline in the window is segmented
    at the character cap. A CRLF pair is never split. Every chunk is exactly
    ``text[start_char:end_char]`` with document-global offsets, so consecutive chunks
    reassemble the original text with no gap and no overlap.
    """
    starts, offset = [], 0
    for line in text.splitlines(keepends=True):
        starts.append(offset)
        offset += len(line)
    chunks, cursor = [], 0
    while cursor < len(text):
        start_index = bisect_right(starts, cursor) - 1
        # Window end: the character cap first, then the line cap of the next
        # ``max_lines`` original line starts. No starts[start_index + 1] access:
        # a small document may legitimately be segmented inside its only line.
        end = min(len(text), cursor + max_chars)
        if start_index + max_lines < len(starts):
            end = min(end, starts[start_index + max_lines])
        if end < len(text):
            # Prefer the last newline inside the window so whole lines stay
            # together; when the window holds no usable newline the segment ends
            # at the character cap instead (a long line is split, never skipped).
            newline = text.rfind("\n", cursor, end)
            if newline >= cursor:
                end = newline + 1
        if end > cursor and end < len(text) and text[end - 1] == "\r" and text[end] == "\n":
            end -= 1
        if end <= cursor:
            end = min(len(text), cursor + max_chars)
        snippet = text[cursor:end]
        chunk = {"chunk_id": f"doc:{doc_id}:{start_index + 1}-{bisect_right(starts, end - 1)}:{cursor}-{end}",
                 "start_line": start_index + 1, "end_line": bisect_right(starts, end - 1),
                 "start_char": cursor, "end_char": end, "text": snippet,
                 "text_sha256": hashlib.sha256(snippet.encode("utf-8")).hexdigest()}
        chunks.append(chunk)
        cursor = end
    return chunks


class ContextRetriever:
    def __init__(self, run_dir: Path, manifest: dict):
        self.manifest = manifest
        self.run_dir = Path(run_dir)
        self.sources = {s["source_id"]: s for s in manifest["sources"]}
        self.records = []
        # metric_ids that carry an APPROVED contract: these are the only names worth looking
        # for inside a historical document's text, so no unrelated word can trigger a fallback.
        self.contract_metric_ids = [c["metric_id"] for c in manifest.get("metric_contracts", [])
                                    if c.get("status") == "APPROVED" and c.get("metric_id")]
        for source in manifest["sources"]:
            for table, fields in source.get("schema", {}).items():
                text = f"source_id={source['source_id']} table={table} " + " ".join(f"{f['name']}:{f.get('type','unknown')}" for f in fields)
                self.records.append({"evidence_id": f"schema:{source['source_id']}:{table}", "type": "schema", "source_id": source["source_id"], "table": table, "text": text, "schema": fields})
        for doc in manifest.get("documents", []):
            # Read bytes and decode UTF8 so the original full SHA256 covers the exact
            # on-disk bytes, including CRLF; splitlines(keepends=True) retains real
            # newline positions and no newline is ever normalized or fabricated.
            raw = Path(doc["path"]).read_bytes()
            text = raw.decode("utf-8")
            lines = text.splitlines(keepends=True)
            doc_id = doc["document_id"]
            common = {"document_id": doc_id, "type": "document", "status": doc.get("status", "DRAFT"),
                      "metric_id": doc.get("metric_id"), "doc_hash": hashlib.sha256(raw).hexdigest(),
                      "line_count": len(lines)}
            if doc.get("source_id") is not None:
                # A source-bound document keeps its origin so a source filter excludes it too;
                # documents without a source_id stay globally available.
                common["source_id"] = doc["source_id"]
            chunks = document_chunks(text, doc_id)
            if len(chunks) == 1:
                chunk = chunks[0]
                self.records.append(dict(common, evidence_id=f"doc:{doc_id}", text=chunk["text"],
                                         start_line=1, end_line=max(1, len(lines)),
                                         start_char=0, end_char=len(text),
                                         text_sha256=chunk["text_sha256"]))
                continue
            for chunk in chunks:
                last_line = chunk["end_line"]
                self.records.append(dict(common, evidence_id=f"doc:{doc_id}:{chunk['start_line']}-{last_line}:{chunk['start_char']}-{chunk['end_char']}",
                                         text=chunk["text"], start_line=chunk["start_line"], end_line=last_line,
                                         start_char=chunk["start_char"], end_char=chunk["end_char"],
                                         text_sha256=chunk["text_sha256"]))
        # This edition executes tools on LangChain worker threads. Its tool
        # middleware serializes access to this run-local index and domain state.
        self.index = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            self.index.execute("CREATE VIRTUAL TABLE evidence USING fts5(text, tokenize='trigram')")
            self.fts = True
            self.index.executemany("INSERT INTO evidence(rowid,text) VALUES(?,?)", [(i+1, r["text"]) for i, r in enumerate(self.records)])
        except sqlite3.OperationalError:
            self.fts = False

    def _candidates(self, query, source_ids, types) -> list[dict]:
        """Score, filter and deterministically rank every eligible indexed chunk.

        Every eligible candidate is returned -- there is no count limit before
        ``_shape``, so a strong late match in another document can never be dropped
        by an earlier top-N truncation. Records with no relevance at all are skipped, so
        an empty or unrelated query yields no evidence. Literal query word hits rank
        first, then the expanded relevance score, then the stable evidence id; the
        internal ranking fields never reach the caller.
        """
        expanded = query + " " + " ".join(v for k, v in ALIASES.items() if k in query)
        expanded_words = {w for w in re.findall(r"[\w/]+", expanded.lower()) if w}
        literal_words = {w for w in re.findall(r"[\w/]+", query.lower()) if w}
        words = {w for w in expanded_words if len(w) >= 3}
        scores = {}
        if self.fts:
            for word in words:
                for row in self.index.execute("SELECT rowid FROM evidence WHERE evidence MATCH ?", ('"' + word.replace('"', '""') + '"',)):
                    scores[row[0]-1] = scores.get(row[0]-1, 0) + (2 if word in literal_words else 1)
        lowered = query.lower().strip()
        candidates = []
        for i, record in enumerate(self.records):
            if source_ids and record.get("source_id") and record["source_id"] not in source_ids:
                continue
            if types and record["type"] not in types and not ("metric" in types and record["type"] == "document" and record.get("metric_id")):
                continue
            body = record["text"].lower()
            literal_hits = sum(1 for w in literal_words if w and w in body)
            relevance = scores.get(i, 0) + literal_hits + sum(1 for w in expanded_words if w and w in body)
            if len(lowered) >= 3 and lowered in body:
                relevance += 3
            if not relevance:
                # Nothing in this record matches the query: it is never evidence.
                continue
            item = dict(record, _literal_hits=literal_hits, _relevance=relevance)
            # An approved definition chunk is promoted only when the query actually
            # reaches it; the remaining document chunks stay ordinary ranked candidates.
            item["_approved_definition"] = bool(
                record["type"] == "document" and record.get("status") == "APPROVED" and record.get("metric_id"))
            candidates.append(item)
        candidates.sort(key=lambda r: (-r["_literal_hits"], -r["_relevance"], r["evidence_id"]))
        promoted = [r for r in candidates if r["_approved_definition"]][:1]
        promoted_ids = {r["evidence_id"] for r in promoted}
        return promoted + [r for r in candidates if r["evidence_id"] not in promoted_ids]

    def _shape(self, candidates: list[dict]) -> list[dict]:
        """Bounded selection: cap chunks per document, then honour the remaining UTF8 budget."""
        evidence, per_document, remaining = [], {}, MAX_EVIDENCE_TEXT_BYTES
        dropped = False
        for item in candidates:
            if len(evidence) >= MAX_EVIDENCE_RECORDS:
                dropped = True
                break
            document_id = item.get("document_id")
            if document_id is not None and per_document.get(document_id, 0) >= MAX_CHUNKS_PER_DOCUMENT:
                dropped = True
                continue
            # Only real document text is charged against the document byte budget: schema
            # (and any other non-document) payload is free, so it never evicts a document
            # chunk nor needs an out-of-band allowance.
            cost = len(item["text"].encode("utf-8")) if item["type"] == "document" else 0
            if cost > remaining:
                dropped = True
                continue
            remaining -= cost
            if document_id is not None:
                per_document[document_id] = per_document.get(document_id, 0) + 1
            evidence.append(item)
        if len(evidence) < len(candidates):
            self._last_shape_truncated = True
        elif dropped:
            self._last_shape_truncated = True
        else:
            self._last_shape_truncated = False
        for item in evidence:
            item.pop("_approved_definition", None)
        return evidence

    def _fallback_metric_ids(self, candidates) -> list[str]:
        """metric_ids named by a matched non-APPROVED document, however that document names them.

        A matched historical/draft document can name the metric it discusses either in its
        ``metric_id`` metadata or only inside its text (a historical note has no metadata but
        still says ``CASH_NET_v1``). Both spellings count. No matched non-approved document
        means no fallback at all, so an unrelated query returns only its own matches.
        """
        named, seen = [], set()
        for record in candidates:
            if record["type"] != "document" or record.get("status") == "APPROVED":
                continue
            metric_id = record.get("metric_id")
            if metric_id and metric_id not in seen:
                seen.add(metric_id)
                named.append(metric_id)
            body = record["text"]
            for known in self.contract_metric_ids:
                if known not in seen and known in body:
                    seen.add(known)
                    named.append(known)
        return named

    def _approved_definition_for(self, metric_id, source_ids, types):
        """The one stable eligible APPROVED definition chunk for ``metric_id``, or ``None``.

        The approved definition exists only to sit beside the historical evidence for the
        *same* metric. It must survive the caller's own source and type filters, exactly like
        any other record, so a filtered-out definition is never resurrected.
        """
        pool = [r for r in self.records
                if r["type"] == "document" and r.get("status") == "APPROVED"
                and r.get("metric_id") == metric_id
                and not (source_ids and r.get("source_id") and r["source_id"] not in source_ids)
                and not (types and r["type"] not in types and not ("metric" in types and r.get("metric_id")))]
        if not pool:
            return None
        return dict(min(pool, key=lambda r: r["evidence_id"]))

    def search(self, query, source_ids=None, types=None) -> dict:
        if not isinstance(query, str) or len(query) > 2000:
            raise ValueError("Search query must be a string of at most 2000 characters")
        if source_ids and any(s not in self.sources for s in source_ids):
            raise ValueError("Unknown source identifier")
        if types and any(t not in {"schema", "document", "metric"} for t in types):
            raise ValueError("Unsupported evidence type")
        self._last_shape_truncated = False
        candidates = self._candidates(query, source_ids, types)
        named_metric_ids = self._fallback_metric_ids(candidates)
        if named_metric_ids:
            # A matched historical document can contradict the approved definition for the
            # same metric, so at most one stable approved definition is placed ahead of it
            # and the whole selection is shaped exactly once: the ranked candidates still
            # fill the caps, so the extra record can never push the result over them.
            present = {r["evidence_id"] for r in candidates}
            for metric_id in named_metric_ids:
                fallback = self._approved_definition_for(metric_id, source_ids, types)
                if fallback is not None and fallback["evidence_id"] not in present:
                    present.add(fallback["evidence_id"])
                    candidates.insert(0, fallback)
        shaped = self._shape(candidates)
        # Only real document text counts against the document byte budget: schema and
        # approved-contract payloads are reported separately and never consume it.
        document_bytes = sum(len(r["text"].encode("utf-8")) for r in shaped if r["type"] == "document")
        evidence = [{k: v for k, v in r.items() if not k.startswith("_")} for r in shaped]
        # Every approved structured contract is returned independently of the evidence
        # type filter; contracts are never reconstructed from snippets.
        contracts = [c for c in self.manifest.get("metric_contracts", []) if c.get("status") == "APPROVED"]
        contract_bytes = len(json.dumps(contracts, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        dropped_candidates = len(evidence) < len(candidates)
        truncated = bool(self._last_shape_truncated or dropped_candidates)
        return {"query": query, "evidence": evidence, "metric_contracts": contracts,
                "missing_context": not evidence, "truncated": truncated,
                "retrieval_mode": "fts5-trigram+aliases" if self.fts else "bounded-alias-substring",
                "paths_exposed": False,
                "evidence_records_max": MAX_EVIDENCE_RECORDS,
                "evidence_records_returned": len(evidence),
                "evidence_text_bytes_max": MAX_EVIDENCE_TEXT_BYTES,
                "document_text_bytes": document_bytes,
                "metric_contract_bytes": contract_bytes,
                "budget_remaining_bytes": max(0, MAX_EVIDENCE_TEXT_BYTES - document_bytes),
                "chunks_indexed": len(self.records)}

    def _row_count(self, source, table) -> int:
        kind = source["kind"]
        if kind == "sqlite":
            # Exact, read-only count of the registered identifier; never materialize rows.
            # The name comes from the validated registered schema, so it is an identifier,
            # never caller SQL -- but an identifier may itself contain a quote, so escape it.
            quoted = table.replace('"', '""')
            with closing(readonly_sqlite(Path(source["path"]))) as db:
                return int(db.execute(f'SELECT count(*) FROM "{quoted}"').fetchone()[0])
        if kind == "parquet":
            import pyarrow.parquet as pq
            return int(pq.read_metadata(source["path"]).num_rows)
        with Path(source["path"]).open("r", encoding="utf-8-sig", newline="") as handle:
            if kind == "csv":
                return sum(1 for _ in csv.DictReader(handle))
            return sum(1 for line in handle if line.strip())

    def inspect_schema(self, source_id, table) -> dict:
        source = self.sources.get(source_id)
        if source is None or table not in source.get("schema", {}):
            raise ValueError("Unknown source or table")
        fields = source["schema"][table]
        relations = []
        if table == "refunds":
            relations.append({"column": "order_id", "references": "business.orders.order_id", "cardinality": "many-to-one"})
        if table == "orders":
            relations.append({"column": "channel_code", "references": "channels.channels.channel_code", "cardinality": "many-to-one"})
        rows = self._row_count(source, table)
        default_limit = 10000
        sqlite_source = source["kind"] == "sqlite"
        return {"source_id": source_id, "table": table, "schema": fields, "relationships": relations,
                "schema_hash": hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest(),
                "row_count": rows, "source_kind": source["kind"],
                # A file LOAD accepts no SQL: only the registered SQLite source can run
                # source-side SQL, so files advertise the DuckDB artifact dialect instead
                # and explicitly deny source SQL.
                "query_dialect": "sqlite" if sqlite_source else None,
                "artifact_query_dialect": "duckdb",
                "source_sql_allowed": sqlite_source,
                "default_load_output_rows": default_limit,
                "raw_load_exceeds_default_output_limit": rows > default_limit}

    def profile_values(self, source_id, table, column) -> dict:
        source = self.sources.get(source_id)
        if not source or table not in source.get("schema", {}) or column not in {f["name"] for f in source["schema"][table]}:
            raise ValueError("Unknown source/table/column")
        rows = source_rows(source, table)
        if len(rows) > 100000:
            raise ValueError("Source exceeds profile limit")
        values, nulls = {}, 0
        for row in rows:
            v = row.get(column)
            if v is None:
                nulls += 1
            else:
                key = json.dumps(v, ensure_ascii=False, sort_keys=True)
                values[key] = values.get(key, 0) + 1
        bounded = sorted(values, key=lambda v: (-values[v], v))[:20]
        return {"source_id": source_id, "table": table, "column": column, "row_count": len(rows), "distinct_count": len(values), "null_count": nulls, "values": [{"value": json.loads(v), "count": values[v]} for v in bounded], "truncated": len(values)>20}

    def close(self):
        self.index.close()
