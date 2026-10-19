"""Deterministic source and artifact checks, separate from evaluator-only gold."""
from __future__ import annotations
import hashlib
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .retrieval import source_rows

EVENT_ZONE = "Australia/Sydney"
INVALID = object()  # sentinel: a timestamp was supplied but is not a usable aware ISO-8601 value


def _check(name, passed, details=None):
    return {"name": name, "passed": bool(passed), "details": details or {}}


def _aware(value):
    """Parse an ISO-8601 timestamp; return it only when it is timezone-aware."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        raise ValueError("Unparseable timestamp")
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Naive timestamp")
    return stamp


def _safe_aware(value):
    """Like _aware, but report unusable timestamps as a sentinel so checks can record them."""
    try:
        return _aware(value)
    except ValueError:
        return INVALID


def _event_timestamp(row, time_col, status, success_status):
    """Return the datetime a CASH_NET_v1 event must be dated by, or a sentinel when unusable.

    Successful events must carry their own financial timestamp; a supplied-but-bad value is never
    rescued by created_at/attempted_at. Recognized non-success rows may omit it, but then need a
    valid aware attempted_at, falling back to created_at only when attempted_at was never supplied:
    an empty supplied attempted_at is invalid, not a hint to try created_at.
    """
    status = None if status is None else str(status).strip().lower()
    financial = row.get(time_col)
    if status == success_status:
        return _safe_aware(financial)
    if status in {"failed", "pending", "cancelled"}:
        supplied = str(financial).strip() if financial is not None else ""
        if supplied:
            return _safe_aware(financial)
        attempt = row.get("attempted_at")
        if attempt is not None:
            return _safe_aware(attempt)
        return _safe_aware(row.get("created_at"))
    return _safe_aware(financial)


def _derived_local_month(stamp):
    """Convert a parsed aware datetime to the business local month; None when there is no instant."""
    if stamp is None:
        return None
    if stamp is INVALID:
        return INVALID
    return stamp.astimezone(ZoneInfo(EVENT_ZONE)).strftime("%Y-%m")


def validate_sources(manifest) -> dict:
    checks = []
    sources = {s["source_id"]: s for s in manifest["sources"]}
    business = sources.get("business")
    if not business:
        return {"passed": False, "checks": [_check("business_source", False, {"error": "Required business source missing"})]}
    try:
        orders, refunds = source_rows(business, "orders"), source_rows(business, "refunds")
        channels = source_rows(sources["channels"])
    except (ValueError, KeyError) as exc:
        return {"passed": False, "checks": [_check("read_sources", False, {"error": str(exc)})]}
    checks.append(_check("source_row_limit", len(orders) <= 100000 and len(refunds) <= 100000 and len(channels) <= 100000))
    def unique(rows, column, name):
        seen, duplicates = set(), []
        for row in rows:
            key = row.get(column)
            if key in seen or key is None:
                duplicates.append(key)
            seen.add(key)
        checks.append(_check(name, not duplicates, {"duplicate_or_null_keys": duplicates[:10]}))
    unique(orders, "order_id", "unique_order_id")
    unique(refunds, "refund_id", "unique_refund_id")
    unique(channels, "channel_code", "unique_channel_mapping")
    order_ids = {r["order_id"] for r in orders}
    orphans = [r["refund_id"] for r in refunds if r["order_id"] not in order_ids]
    checks.append(_check("refund_foreign_key", not orphans, {"refund_ids": orphans[:10]}))
    channel_ids = {r["channel_code"] for r in channels}
    unmapped = [r["order_id"] for r in orders if r["channel_code"] not in channel_ids]
    checks.append(_check("mapped_channels", not unmapped, {"order_ids": unmapped[:10]}))
    invalid_money, invalid_time, invalid_status = [], [], []
    month_mismatch, month_invalid = [], []
    for rows, id_col, money_col, time_col, success, statuses in [
        (orders, "order_id", "paid_cents", "paid_at", "paid", {"paid", "failed", "pending", "cancelled"}),
        (refunds, "refund_id", "refund_cents", "refunded_at", "succeeded", {"succeeded", "failed", "pending", "cancelled"}),
    ]:
        for row in rows:
            record_id = row.get(id_col)
            money = row.get(money_col)
            if isinstance(money, bool) or not isinstance(money, int) or money < 0:
                invalid_money.append(record_id)
            status = row.get("status")
            if status not in statuses:
                invalid_status.append(record_id)
            event_time = _event_timestamp(row, time_col, status, success)
            if event_time is None or event_time is INVALID:
                invalid_time.append(record_id)
            # Absent or NULL local_month is legacy data and is not a month error; a supplied value
            # must still equal the independently derived month (invalid values are reported).
            if "local_month" in row and row.get("local_month") is not None:
                month = _derived_local_month(event_time)
                if month is INVALID:
                    month_invalid.append(record_id)
                elif month != row.get("local_month"):
                    month_mismatch.append(record_id)
    checks.append(_check("integer_nonnegative_cents", not invalid_money, {"record_ids": invalid_money[:10]}))
    checks.append(_check("aware_event_timestamps", not invalid_time, {"record_ids": invalid_time[:10]}))
    checks.append(_check("known_event_status", not invalid_status, {"record_ids": invalid_status[:10]}))
    derived_months = [{"check": "invalid_derived_timestamps", "record_ids": month_invalid[:10]}, {"check": "local_month_mismatch", "record_ids": month_mismatch[:10]}]
    checks.append(_check("derived_event_months", not month_invalid and not month_mismatch, {"mismatches": derived_months}))
    contracts = [c for c in manifest.get("metric_contracts", []) if c.get("status") == "APPROVED"]
    checks.append(_check("approved_metric_available", bool(contracts), {"needs_clarification": not contracts}))
    checks.append(_check("metric_unit_and_timezone", bool(contracts) and all(c.get("unit") == "CNY_cents" and c.get("timezone") == "Australia/Sydney" for c in contracts)))
    for source in manifest["sources"]:
        path = Path(source["path"])
        if source.get("sha256"):
            checks.append(_check("snapshot_hash:" + source["source_id"], hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]))
    return {"passed": all(c["passed"] for c in checks), "needs_clarification": not contracts, "checks": checks}


def _rows(artifact):
    if artifact.get("path"):
        import pyarrow.parquet as pq
        path = Path(artifact["path"])
        if pq.ParquetFile(path).metadata.num_rows > 10000:
            raise ValueError("Artifact exceeds check row limit")
        return pq.read_table(path).to_pylist()
    return artifact.get("rows", artifact.get("preview", []))


def check_artifacts(artifacts, check_ids) -> dict:
    allowed = {"schema", "nonempty", "finite", "unique_keys", "monthly_reconciliation", "metric_timezone", "semantic_evidence", "integrity", "row_count", "money_units", "source_integrity", "consistency"}
    unknown = set(check_ids) - allowed
    if unknown:
        raise ValueError("Unknown check IDs: " + ",".join(sorted(unknown)))
    checks, loaded = [], {}
    for name, artifact in artifacts.items():
        rows = _rows(artifact)
        loaded[name] = rows
        if "schema" in check_ids or "integrity" in check_ids:
            columns = {s["name"] for s in artifact.get("schema", [])}
            checks.append(_check(f"schema:{name}", not columns or all(set(r) == columns for r in rows)))
        if "row_count" in check_ids or "integrity" in check_ids:
            checks.append(_check(f"row_count:{name}", artifact.get("row_count", len(rows)) == len(rows)))
        if ("integrity" in check_ids or "source_integrity" in check_ids) and artifact.get("sha256") and artifact.get("path"):
            checks.append(_check(f"hash:{name}", hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]))
        if "nonempty" in check_ids:
            checks.append(_check(f"nonempty:{name}", bool(rows)))
        if "finite" in check_ids:
            checks.append(_check(f"finite:{name}", all(not isinstance(v, float) or math.isfinite(v) for r in rows for v in r.values())))
        if "money_units" in check_ids:
            checks.append(_check(f"money_units:{name}", all(isinstance(v, int) and not isinstance(v, bool) for r in rows for k,v in r.items() if k.endswith("_cents") and v is not None)))
        if "unique_keys" in check_ids:
            columns = {k for r in rows for k in r}
            keys = [k for k in ("month", "channel_code") if k in columns]
            if not keys:
                keys = [k for k in ("order_id", "refund_id", "ticket_id") if k in columns]
            checks.append(_check(f"unique_keys:{name}", bool(keys) and len({tuple(r.get(k) for k in keys) for r in rows}) == len(rows), {"keys": keys}))
        if "metric_timezone" in check_ids:
            # A result alone cannot prove that timezone conversion occurred.
            metadata = artifact.get("metric_contract", {})
            checks.append(_check(f"metric_timezone:{name}", metadata.get("timezone") == "Australia/Sydney", {"requires_metric_contract": True}))
        if "semantic_evidence" in check_ids:
            checks.append(_check(f"semantic_evidence:{name}", all(r.get("evidence_id") and isinstance(r.get("start_line"), int) and isinstance(r.get("end_line"), int) and 1 <= r["start_line"] <= r["end_line"] <= r.get("line_count", 0) for r in rows)))
    if "monthly_reconciliation" in check_ids or "consistency" in check_ids:
        total_rows = [r for rows in loaded.values() for r in rows if "month" in r and "net_cents" in r and "channel_code" not in r]
        channel_rows = [r for rows in loaded.values() for r in rows if "month" in r and "net_cents" in r and "channel_code" in r]
        if total_rows and channel_rows:
            grouped = {}
            for row in channel_rows:
                grouped[row["month"]] = grouped.get(row["month"], 0) + row["net_cents"]
            expected = {r["month"]: r["net_cents"] for r in total_rows}
            passed = len(expected) == len(total_rows) and expected == grouped
        else:
            passed = False
        checks.append(_check("monthly_reconciliation", passed, {"requires_month_totals_and_channel_totals": True}))
    return {"passed": bool(checks) and all(c["passed"] for c in checks), "checks": checks}


def evaluate_gold(artifacts, gold):
    """Only the external evaluator may call this with hidden expected amounts."""
    checks, candidates = [], []
    for name, artifact in artifacts.items():
        rows = _rows(artifact)
        if rows and all("month" in r and "net_cents" in r for r in rows):
            candidates.append((name, rows))
    monthly = [(name, rows) for name, rows in candidates if "channel_code" not in rows[0]]
    channel = [(name, rows) for name, rows in candidates if "channel_code" in rows[0]]
    if monthly:
        # All candidate totals must agree; don't mask a wrong final table by finding a correct intermediate.
        for name, rows in monthly:
            actual = {str(r["month"])[:7]: r["net_cents"] for r in rows}
            expected = gold["monthly_cents"]
            # A correct calendar scaffold may include zero-event months omitted by GROUP BY.
            matches = all(actual.get(month) == cents for month, cents in expected.items()) and all(cents == 0 for month, cents in actual.items() if month not in expected)
            checks.append(_check("gold_monthly:"+name, matches and len(actual) == len(rows), {"expected": expected, "actual": actual}))
    else:
        checks.append(_check("gold_monthly", False, {"error": "No monthly result artifact"}))
    for name, rows in channel:
        actual = sorted((str(r["month"])[:7], r["channel_code"], r["net_cents"]) for r in rows)
        expected = sorted((r["month"], r["channel_code"], r["net_cents"]) for r in gold["monthly_channel_cents"])
        actual_map = {(month, channel): cents for month, channel, cents in actual}
        expected_map = {(month, channel): cents for month, channel, cents in expected}
        matches = len(actual_map) == len(actual) and all(actual_map.get(key) == cents for key, cents in expected_map.items()) and all(cents == 0 for key, cents in actual_map.items() if key not in expected_map)
        checks.append(_check("gold_channels:"+name, matches))
    return {"passed": all(c["passed"] for c in checks), "checks": checks}
