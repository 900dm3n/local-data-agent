"""Behavioral tests for bounded document chunking and evidence retrieval.

These tests exercise observable behavior of ``ContextRetriever``: exact chunk slices
with correct global offsets and line bounds, bounded selection, source/type filtering,
ranking and contract availability. No test assumes that every indexed chunk is returned
by one bounded search.
"""
import hashlib
import json

import pytest

from local_data_agent.retrieval import (
    CHUNK_MAX_CHARS,
    CHUNK_MAX_LINES,
    MAX_EVIDENCE_RECORDS,
    MAX_EVIDENCE_TEXT_BYTES,
    ContextRetriever,
)

FILLER = "http://example.invalid/path?q="


def test_alias_substring_fallback_without_fts(tmp_path):
    retriever = retriever_for(tmp_path, [])
    try:
        retriever.fts = False
        result = retriever.search("订单", types=["schema"])
        assert any(r.get("table") == "orders" for r in result["evidence"])
        assert result["retrieval_mode"] == "bounded-alias-substring"
    finally:
        retriever.close()


def fill(length, tag=FILLER):
    return (tag * (length // len(tag) + 1))[:length]


def lines(count, tag="filler", width=60):
    body = (tag * width)[:width]
    return ["%03d %s\n" % (i, body) for i in range(count)]


def write_document(path, text):
    """Write the exact UTF-8 bytes of ``text`` (LF or CRLF exactly as given)."""
    path.write_bytes(text.encode("utf-8"))
    return path


def manifest_for(documents, contracts=None):
    return {"sources": [{"source_id": "business", "kind": "sqlite", "path": "absent.sqlite",
                         "schema": {"orders": [{"name": "paid_cents", "type": "INTEGER"}]}},
                        {"source_id": "channels", "kind": "csv", "path": "absent.csv",
                         "schema": {"channels": [{"name": "channel_code", "type": "VARCHAR"}]}}],
            "documents": documents,
            "metric_contracts": contracts or []}


def retriever_for(tmp_path, documents, contracts=None):
    (tmp_path / "run").mkdir(exist_ok=True)
    return ContextRetriever(tmp_path / "run", manifest_for(documents, contracts))


def doc_entry(path, document_id, status="REFERENCE", **extra):
    return {"document_id": document_id, "path": str(path), "status": status, **extra}


def contract(metric_id, **extra):
    return {"metric_id": metric_id, "version": 1, "status": "APPROVED", "unit": "CNY_cents",
            "formula": "paid minus refunded", "timezone": "Australia/Sydney", **extra}


def assert_chunks_exact(chunks, original):
    """Every returned chunk is an exact slice, char/line bounded, hashed and contiguous."""
    assert chunks, "no chunks returned"
    ordered = sorted(chunks, key=lambda c: c["start_char"])
    assert ordered[0]["start_char"] == 0
    cursor = 0
    for chunk in ordered:
        assert chunk["text"] == original[chunk["start_char"]:chunk["end_char"]]
        assert chunk["start_char"] == cursor, "returned chunks must be contiguous"
        assert len(chunk["text"]) <= CHUNK_MAX_CHARS
        assert chunk["end_line"] - chunk["start_line"] + 1 <= CHUNK_MAX_LINES
        assert chunk["text_sha256"] == hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
        cursor = chunk["end_char"]


def test_embedded_long_line_between_regular_lines_keeps_bounds_and_offsets(tmp_path):
    """Unicode + CRLF long line embedded between regular lines: exact slices and bounds."""
    long_line = "INCOME-" + fill(2500) + "-END"
    assert len(long_line) > 2400
    text = "first header line\r\n" + long_line + "\r\n尾行 closing line\r\n"
    write_document(tmp_path / "embedded.md", text)
    retriever = retriever_for(tmp_path, [doc_entry(tmp_path / "embedded.md", "embedded")])
    try:
        records = [r for r in retriever.records if r["type"] == "document"]
        assert records, "the document must be indexed"
        # The whole indexed set reassembles the exact original text.
        for record in records:
            assert record["text"] == text[record["start_char"]:record["end_char"]]
            assert record["doc_hash"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
            assert record["text_sha256"] == hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
            assert len(record["text"]) <= CHUNK_MAX_CHARS
            assert record["end_line"] - record["start_line"] + 1 <= CHUNK_MAX_LINES
        # Filter by record type first: only document records carry char offsets, so
        # exact coverage is asserted over exactly those records and nothing else.
        documents = [r for r in records if r["type"] == "document"]
        assert documents, "the document records must be indexed"
        assert all("start_char" in r for r in documents), "every document record carries start_char"
        ordered = sorted(documents, key=lambda r: r["start_char"])
        assert ordered[0]["start_char"] == 0 and ordered[-1]["end_char"] == len(text)
        assert "".join(r["text"] for r in ordered) == text, "indexed chunks reassemble the exact text"
        assert "\r\n" in "".join(r["text"] for r in ordered), "CRLF is never normalized"
        assert ordered[0]["start_line"] == 1, "the first line has correct line bounds"
        assert ordered[-1]["end_line"] == len(text.splitlines())
        # A single bounded search returns only part of the coverage -- that is expected.
        result = retriever.search("INCOME", types=["document"])
        returned = [r for r in result["evidence"] if r["type"] == "document"]
        assert returned, "the long line must be reachable"
        token_offset = text.index("INCOME")
        assert any(r["start_char"] <= token_offset < r["end_char"] for r in returned)
        assert all(r["text"] == text[r["start_char"]:r["end_char"]] for r in returned)
    finally:
        retriever.close()


def test_small_three_line_document_keeps_the_legacy_single_record_id(tmp_path):
    """A document inside one window stays one exact record with the legacy ``doc:<id>`` id."""
    text = "alpha line\nbeta line\ngamma line\n"
    write_document(tmp_path / "small.md", text)
    retriever = retriever_for(tmp_path, [doc_entry(tmp_path / "small.md", "small")])
    try:
        records = [r for r in retriever.records if r["type"] == "document"]
        assert len(records) == 1, "a small document must not be split into per-line chunks"
        record = records[0]
        assert record["evidence_id"] == "doc:small"
        assert record["text"] == text
        assert (record["start_char"], record["end_char"]) == (0, len(text))
        assert (record["start_line"], record["end_line"]) == (1, 3)
        assert record["text_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
        result = retriever.search("gamma", types=["document"])
        returned = [r for r in result["evidence"] if r["type"] == "document"]
        assert returned and returned[0]["evidence_id"] == "doc:small"
        assert returned[0]["text"] == text[returned[0]["start_char"]:returned[0]["end_char"]]
    finally:
        retriever.close()


def test_single_long_line_with_newline_at_tail_is_split_at_the_char_cap(tmp_path):
    """One overlong line ending in a newline: char-capped segments that still reassemble."""
    # This line uses a filler without the literal ``q=q=q`` sequence, so the token searched
    # below can only be found at its real offset -- never as an accident of the filler.
    text = fill(2500, tag="http://example.invalid/path?filler=") + "TAIL_TOKEN_42" \
        + fill(100, tag="http://example.invalid/path?filler=") + "\n"
    write_document(tmp_path / "oneline.md", text)
    retriever = retriever_for(tmp_path, [doc_entry(tmp_path / "oneline.md", "oneline")])
    try:
        records = [r for r in retriever.records if r["type"] == "document"]
        ordered = sorted(records, key=lambda r: r["start_char"])
        assert len(ordered) >= 3, "an overlong line must be segmented, not collapsed"
        assert ordered[0]["start_char"] == 0 and ordered[-1]["end_char"] == len(text)
        cursor = 0
        for record in ordered:
            assert record["text"] == text[record["start_char"]:record["end_char"]]
            assert record["start_char"] == cursor
            assert len(record["text"]) <= CHUNK_MAX_CHARS
            assert record["text_sha256"] == hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
            cursor = record["end_char"]
        assert "".join(r["text"] for r in ordered) == text
        assert "q=q=q" not in text, "the filler must not contain the searched token by accident"
        token_offset = text.index("TAIL_TOKEN_42")
        assert token_offset > 2000, "the token sits well past the char cap"
        result = retriever.search("TAIL_TOKEN_42", types=["document"])
        returned = [r for r in result["evidence"] if r["type"] == "document"]
        assert returned, "a token late in the only line must be reachable"
        hits = [r for r in returned if r["start_char"] <= token_offset < r["end_char"]]
        assert hits, "the returned chunk must carry the token's original offset"
        hit = hits[0]
        # The evidence is the real slice of the original text that contains the token, so
        # the offset is reported and no substring is ever invented.
        assert hit["text"] == text[hit["start_char"]:hit["end_char"]]
        assert "TAIL_TOKEN_42" in hit["text"]
        assert "q=q=q" not in hit["text"]
    finally:
        retriever.close()


def test_search_token_after_line_100_is_recalled_with_stable_ids(tmp_path):
    text = "".join(lines(120))
    text += "sentinel-narwhal appears only on this late line\n"
    text += "".join("tail filler %03d\n" % i for i in range(20))
    write_document(tmp_path / "late.md", text)
    retriever = retriever_for(tmp_path, [doc_entry(tmp_path / "late.md", "late")])
    try:
        offset = text.index("narwhal")
        first = retriever.search("narwhal", types=["document"])
        returned = [r for r in first["evidence"] if r["type"] == "document"]
        assert returned, "a token only after line 100 must still be recalled"
        assert all("narwhal" in r["text"] for r in returned)
        hit = returned[0]
        assert hit["start_char"] <= offset < hit["end_char"], "the chunk carries the true global range"
        assert hit["start_char"] >= text.index("sentinel-narwhal") - CHUNK_MAX_CHARS
        second = retriever.search("narwhal", types=["document"])
        assert [r["evidence_id"] for r in second["evidence"]] == [r["evidence_id"] for r in first["evidence"]]
    finally:
        retriever.close()


def test_token_only_in_long_line_tail_is_recalled_at_correct_offset(tmp_path):
    long_line = fill(2400) + " zephyrquartz tail marker"
    text = "intro line\n" + long_line + "\nfinal line\n"
    write_document(tmp_path / "long.md", text)
    retriever = retriever_for(tmp_path, [doc_entry(tmp_path / "long.md", "long")])
    try:
        token_offset = text.index("zephyrquartz")
        result = retriever.search("zephyrquartz", types=["document"])
        returned = [r for r in result["evidence"] if r["type"] == "document"]
        assert returned, "the tail of an overlong line must be retrievable"
        assert returned[0]["start_char"] <= token_offset < returned[0]["end_char"]
        assert returned[0]["text"] == text[returned[0]["start_char"]:returned[0]["end_char"]]
        assert result["missing_context"] is False
    finally:
        retriever.close()


def test_many_match_queries_stay_inside_the_bounded_contract(tmp_path):
    documents = []
    for index in range(6):
        text = "".join("aurora keyword payload %02d in doc %d\n" % (line, index) for line in range(60))
        path = tmp_path / ("aurora_%d.md" % index)
        write_document(path, text)
        documents.append(doc_entry(path, "aurora_%d" % index))
    contracts = [contract("CASH_NET_v1"), contract("GROSS_v2"), dict(contract("DRAFT_v3"), status="DRAFT")]
    retriever = retriever_for(tmp_path, documents, contracts)
    try:
        result = retriever.search("aurora keyword", types=["document"])
        evidence = result["evidence"]
        assert len(evidence) <= MAX_EVIDENCE_RECORDS
        counts = {}
        for record in evidence:
            counts[record["document_id"]] = counts.get(record["document_id"], 0) + 1
            original = (tmp_path / (record["document_id"] + ".md")).read_text(encoding="utf-8")
            assert record["text"] == original[record["start_char"]:record["end_char"]]
            assert record["doc_hash"] == hashlib.sha256(original.encode("utf-8")).hexdigest()
        assert counts and max(counts.values()) <= 2, "at most 2 chunks per document"
        assert result["document_text_bytes"] <= MAX_EVIDENCE_TEXT_BYTES
        assert result["truncated"] is True, "capping must be reported truthfully"
        assert result["document_text_bytes"] == sum(
            len(r["text"].encode("utf-8")) for r in evidence if r["type"] == "document")
        assert result["metric_contracts"] == [c for c in contracts if c["status"] == "APPROVED"]
    finally:
        retriever.close()


def test_ten_matching_chunks_keep_the_top_document_and_a_second_one(tmp_path):
    documents = []
    for index in range(10):
        text = "".join("kraken signal row %02d doc %02d\n" % (line, index) for line in range(40))
        path = tmp_path / ("kraken_%02d.md" % index)
        write_document(path, text)
        documents.append(doc_entry(path, "kraken_%02d" % index))
    retriever = retriever_for(tmp_path, documents)
    try:
        result = retriever.search("kraken", types=["document"])
        returned = [r for r in result["evidence"] if r["type"] == "document"]
        assert len(returned) >= 2
        assert {r["evidence_id"] for r in returned} == {r["evidence_id"] for r in result["evidence"]}
        counts = {}
        for record in returned:
            counts[record["document_id"]] = counts.get(record["document_id"], 0) + 1
        top_document = returned[0]["document_id"]
        assert counts[top_document] == 2, "the top document fills its per-document cap first"
        # A second genuinely matching document must not be squeezed out by the first.
        assert any(document_id != top_document for document_id in counts), "a second match is lost"
        assert result["truncated"] is True
    finally:
        retriever.close()


def test_late_approved_chunk_and_matching_draft_keep_their_statuses(tmp_path):
    approved_text = "".join("pad approved line %03d\n" % i for i in range(120))
    approved_text += "phoenix approved ruling reference\n"
    write_document(tmp_path / "approved.md", approved_text)
    draft_text = "phoenix draft proposal not yet approved\n"
    write_document(tmp_path / "draft.md", draft_text)
    documents = [doc_entry(tmp_path / "approved.md", "late_approved", status="APPROVED", metric_id="CASH_NET_v1"),
                 doc_entry(tmp_path / "draft.md", "draft_note", status="DRAFT")]
    retriever = retriever_for(tmp_path, documents, [contract("CASH_NET_v1")])
    try:
        result = retriever.search("phoenix", types=["document"])
        returned = [r for r in result["evidence"] if r["type"] == "document"]
        by_id = {r["document_id"]: r for r in returned}
        assert set(by_id) == {"late_approved", "draft_note"}
        assert by_id["late_approved"]["status"] == "APPROVED"
        assert by_id["draft_note"]["status"] == "DRAFT"
        assert "phoenix" in by_id["late_approved"]["text"]
        assert result["metric_contracts"] == [contract("CASH_NET_v1")]
    finally:
        retriever.close()


def test_source_bound_document_filters_while_global_document_stays(tmp_path):
    write_document(tmp_path / "bound.md", "bound ledger reconciliation note\n")
    write_document(tmp_path / "global.md", "global ledger reconciliation note\n")
    documents = [doc_entry(tmp_path / "bound.md", "bound", source_id="business"),
                 doc_entry(tmp_path / "global.md", "global")]
    retriever = retriever_for(tmp_path, documents)
    try:
        unscoped = retriever.search("ledger", types=["document"])
        assert {r["document_id"] for r in unscoped["evidence"]} == {"bound", "global"}
        scoped = retriever.search("ledger", source_ids=["business"], types=["document"])
        assert {r["document_id"] for r in scoped["evidence"]} == {"bound", "global"}
        foreign = retriever.search("ledger", source_ids=["channels"], types=["document"])
        assert {r["document_id"] for r in foreign["evidence"]} == {"global"}
        assert foreign["missing_context"] is False
        # The type filter still narrows evidence while contracts stay independent.
        typed = retriever.search("ledger", types=["schema"])
        assert all(r["type"] == "schema" for r in typed["evidence"])
    finally:
        retriever.close()


def test_unrelated_query_keeps_contracts_and_source_bytes_unchanged(tmp_path):
    write_document(tmp_path / "metric.md", "approved definition body\nsecond line\n")
    documents = [doc_entry(tmp_path / "metric.md", "metric", status="APPROVED", metric_id="CASH_NET_v1")]
    contracts = [contract("CASH_NET_v1"), dict(contract("DRAFT_v3"), status="DRAFT")]
    retriever = retriever_for(tmp_path, documents, contracts)
    try:
        before = (tmp_path / "metric.md").read_bytes()
        for query in ("", "zzzz no such token zzzz"):
            result = retriever.search(query, types=["document"])
            assert result["evidence"] == [], "an unrelated query yields no evidence"
            assert result["missing_context"] is True
            assert result["document_text_bytes"] == 0
            assert result["truncated"] is False
            assert result["metric_contracts"] == [contract("CASH_NET_v1")], "approved contracts stay present"
            assert result["metric_contract_bytes"] == len(
                json.dumps([contract("CASH_NET_v1")], ensure_ascii=False, sort_keys=True).encode("utf-8"))
        # Unknown sources are still rejected rather than silently ignored.
        with pytest.raises(ValueError):
            retriever.search("metric", source_ids=["nope"])
        assert (tmp_path / "metric.md").read_bytes() == before, "retrieval never rewrites the source"
    finally:
        retriever.close()
