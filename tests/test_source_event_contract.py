"""Source event semantics for CASH_NET_v1: mandatory success timestamps and attempt fallback.

These tests build minimal temporary manifests themselves; they never call the evaluator gold
helper or any model. All timestamps are aware UTC strings.
"""
import csv
import json
import sqlite3
from pathlib import Path

import pytest

from local_data_agent.validation import validate_sources

ZONE = "Australia/Sydney"

ORDER_COLUMNS = "order_id TEXT, paid_at TEXT, channel_code TEXT, paid_cents INTEGER, status TEXT, attempted_at TEXT, created_at TEXT, local_month TEXT"
REFUND_COLUMNS = "refund_id TEXT, order_id TEXT, refunded_at TEXT, refund_cents INTEGER, status TEXT, attempted_at TEXT, created_at TEXT, local_month TEXT"


def _order(order_id, status="paid", paid_at="2026-08-05T02:00:00Z", cents=10000, channel="web", **extra):
    return {"order_id": order_id, "paid_at": paid_at, "channel_code": channel, "paid_cents": cents, "status": status, "attempted_at": None, "created_at": None, "local_month": None, **extra}


def _refund(refund_id, order_id, status="succeeded", refunded_at="2026-08-25T02:00:00Z", cents=2000, **extra):
    return {"refund_id": refund_id, "order_id": order_id, "refunded_at": refunded_at, "refund_cents": cents, "status": status, "attempted_at": None, "created_at": None, "local_month": None, **extra}


def _manifest(tmp_path, orders, refunds, channels=(("web", "官网"),)) -> dict:
    """Write a minimal business SQLite source plus channels CSV and return a manifest."""
    folder = Path(tmp_path)
    folder.mkdir(parents=True, exist_ok=True)
    business = folder / "business.sqlite"
    with sqlite3.connect(business) as db:
        db.execute(f"CREATE TABLE orders({ORDER_COLUMNS})")
        db.execute(f"CREATE TABLE refunds({REFUND_COLUMNS})")
        keys = ["order_id", "paid_at", "channel_code", "paid_cents", "status", "attempted_at", "created_at", "local_month"]
        db.executemany(f"INSERT INTO orders VALUES({','.join('?' * len(keys))})", [[row.get(k) for k in keys] for row in orders])
        keys = ["refund_id", "order_id", "refunded_at", "refund_cents", "status", "attempted_at", "created_at", "local_month"]
        db.executemany(f"INSERT INTO refunds VALUES({','.join('?' * len(keys))})", [[row.get(k) for k in keys] for row in refunds])
        db.commit()
    csv_path = folder / "channels.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel_code", "channel_name"])
        writer.writerows(channels)
    schema = {"orders": [{"name": name, "type": "TEXT"} for name in ORDER_COLUMNS.split(", ")], "refunds": [{"name": name, "type": "TEXT"} for name in REFUND_COLUMNS.split(", ")]}
    return {
        "sources": [
            {"source_id": "business", "kind": "sqlite", "path": str(business), "tables": ["orders", "refunds"], "schema": schema},
            {"source_id": "channels", "kind": "csv", "path": str(csv_path)},
        ],
        "metric_contracts": [{"metric_id": "CASH_NET_v1", "status": "APPROVED", "unit": "CNY_cents", "timezone": ZONE}],
    }


def _failures(manifest):
    result = validate_sources(manifest)
    return result, {c["name"] for c in result["checks"] if not c["passed"]}


def test_success_timestamp_mandatory_aware_and_not_rescued(tmp_path):
    """A paid order needs its own aware paid_at; created_at/attempted_at cannot rescue it."""
    missing = _manifest(tmp_path / "missing", [_order("O1", paid_at=None, attempted_at="2026-08-05T02:00:00Z", created_at="2026-08-05T01:00:00Z")], [_refund("R1", "O1")])
    result, failures = _failures(missing)
    assert "aware_event_timestamps" in failures
    assert not result["passed"]
    naive = _manifest(tmp_path / "naive", [_order("O1", paid_at="2026-08-05T02:00:00", attempted_at="2026-08-05T02:00:00Z")], [_refund("R1", "O1")])
    _, failures = _failures(naive)
    assert "aware_event_timestamps" in failures


def test_pending_null_financial_time_with_aware_attempt_accepted(tmp_path):
    """Non-paid rows may omit paid_at/refunded_at when an aware attempted_at is supplied."""
    orders = [_order("O1"), _order("O2", status="pending", paid_at=None, paid_cents=0, attempted_at="2026-09-30T14:30:00Z")]
    refunds = [_refund("R1", "O1"), _refund("R2", "O1", status="failed", refunded_at=None, refund_cents=0, attempted_at="2026-09-30T14:30:00Z")]
    result, failures = _failures(_manifest(tmp_path, orders, refunds))
    assert failures == set(), result["checks"]
    assert result["passed"]


def test_failed_row_without_any_timestamp_rejected(tmp_path):
    """No timestamp-free records: a failed row with no financial/attempt/created time fails."""
    orders = [_order("O1"), _order("O2", status="failed", paid_at=None, paid_cents=0)]
    _, failures = _failures(_manifest(tmp_path, orders, [_refund("R1", "O1")]))
    assert "aware_event_timestamps" in failures


def test_invalid_supplied_financial_timestamp_rejected_on_non_success(tmp_path):
    """A nonempty but unparseable paid_at on a failed row is rejected outright."""
    orders = [_order("O1"), _order("O2", status="failed", paid_at="not-a-timestamp", paid_cents=0, attempted_at="2026-08-05T02:00:00Z")]
    _, failures = _failures(_manifest(tmp_path, orders, [_refund("R1", "O1")]))
    assert "aware_event_timestamps" in failures


def test_bad_attempted_at_does_not_fall_back_to_created_at(tmp_path):
    """A malformed attempted_at fails; created_at must not silently rescue the record."""
    orders = [_order("O1"), _order("O2", status="pending", paid_at=None, paid_cents=0, attempted_at="2026-08-05 02:00", created_at="2026-08-05T02:00:00Z")]
    _, failures = _failures(_manifest(tmp_path, orders, [_refund("R1", "O1")]))
    assert "aware_event_timestamps" in failures


def test_created_at_fallback_only_when_attempt_absent(tmp_path):
    """created_at is an accepted fallback when attempted_at was never supplied at all."""
    orders = [_order("O1"), _order("O2", status="cancelled", paid_at=None, paid_cents=0, created_at="2026-08-05T02:00:00Z")]
    result, failures = _failures(_manifest(tmp_path, orders, [_refund("R1", "O1")]))
    assert failures == set(), result["checks"]


def test_empty_attempted_at_rejects_even_with_valid_created_at(tmp_path):
    """A supplied but empty attempted_at is invalid; a valid created_at must not be used instead."""
    orders = [_order("O1"), _order("O2", status="pending", paid_at=None, paid_cents=0, attempted_at="", created_at="2026-08-05T02:00:00Z")]
    result, failures = _failures(_manifest(tmp_path, orders, [_refund("R1", "O1")]))
    assert "aware_event_timestamps" in failures
    details = next(c["details"] for c in result["checks"] if c["name"] == "aware_event_timestamps")
    assert details["record_ids"] == ["O2"]


def test_absent_local_month_accepted_when_column_present(tmp_path):
    """A declared local_month column holding NULL is legacy data, not a month failure."""
    orders = [_order("O1", paid_at="2026-08-31T14:30:00Z"), _order("O2", paid_at="2026-10-31T13:30:00Z")]
    result, failures = _failures(_manifest(tmp_path, orders, [_refund("R1", "O1")]))
    assert failures == set(), result["checks"]
    assert result["passed"]


def test_unknown_status_still_rejected(tmp_path):
    orders = [_order("O1", status="refunded", paid_at=None, attempted_at="2026-08-05T02:00:00Z")]
    _, failures = _failures(_manifest(tmp_path, orders, [_refund("R1", "O1")]))
    assert "known_event_status" in failures


def test_legacy_fixture_without_attempt_columns_accepted(tmp_path):
    """Legacy sources lacking attempted_at/created_at/local_month and with all-success rows pass."""
    folder = Path(tmp_path)
    business = folder / "legacy.sqlite"
    with sqlite3.connect(business) as db:
        db.execute("CREATE TABLE orders(order_id TEXT,paid_at TEXT,channel_code TEXT,paid_cents INTEGER,status TEXT)")
        db.execute("CREATE TABLE refunds(refund_id TEXT,order_id TEXT,refunded_at TEXT,refund_cents INTEGER,status TEXT)")
        db.execute("INSERT INTO orders VALUES('O1','2026-08-05T02:00:00Z','web',10000,'paid')")
        db.execute("INSERT INTO refunds VALUES('R1','O1','2026-08-25T02:00:00Z',2000,'succeeded')")
        db.commit()
    channels = folder / "channels.csv"
    channels.write_text("channel_code,channel_name\nweb,官网\n", encoding="utf-8")
    manifest = {"sources": [{"source_id": "business", "kind": "sqlite", "path": str(business), "tables": ["orders", "refunds"]}, {"source_id": "channels", "kind": "csv", "path": str(channels)}], "metric_contracts": [{"metric_id": "CASH_NET_v1", "status": "APPROVED", "unit": "CNY_cents", "timezone": ZONE}]}
    result, failures = _failures(manifest)
    assert failures == set(), result["checks"]
    assert result["passed"]


@pytest.mark.parametrize(
    "stamp,expected_month",
    [
        # AEST (UTC+10) before the October DST switch: 14:30Z is the next local day.
        ("2026-08-31T14:30:00Z", "2026-09"),
        # AEDT (UTC+11) after the switch: 13:30Z already rolls into the next local month.
        ("2026-10-31T13:30:00Z", "2026-11"),
        # DST ends 2026-04-05 in Sydney: the same UTC hour yields different local hours,
        # but both remain in April and must not be treated as May.
        ("2026-04-04T15:30:00Z", "2026-04"),
        ("2026-04-05T15:30:00Z", "2026-04"),
    ],
)
def test_local_month_derived_through_dst_boundaries(tmp_path, stamp, expected_month):
    """local_month is independently converted to Australia/Sydney, including DST changes."""
    orders = [_order("O1", paid_at=stamp, local_month=expected_month)]
    result, failures = _failures(_manifest(tmp_path, orders, [_refund("R1", "O1", refunded_at=stamp, local_month=expected_month)]))
    assert failures == set(), result["checks"]


def test_local_month_mismatch_and_invalid_rows_reported(tmp_path):
    """Any declared local_month that disagrees with the derived month is a bounded mismatch.

    The validator derives the month from the row's own timestamp; it does not parse the declared
    string as a month. So both a plausible-but-wrong month (O1) and an unparseable one (O2) are
    rejected under `local_month_mismatch`, and neither record may slip through as accepted.
    """
    orders = [
        _order("O1", paid_at="2026-08-31T14:30:00Z", local_month="2026-08"),
        _order("O2", paid_at="2026-08-05T02:00:00Z", local_month="not-a-month"),
    ]
    result, failures = _failures(_manifest(tmp_path, orders, [_refund("R1", "O1")]))
    assert "derived_event_months" in failures
    assert not result["passed"]
    details = next(c["details"] for c in result["checks"] if c["name"] == "derived_event_months")
    reported = {entry["check"]: entry["record_ids"] for entry in details["mismatches"]}
    # O1 declares 2026-08 but 14:30Z on 2026-08-31 is 2026-09 in Sydney; O2 declares a non-month.
    # Both are rejected as mismatches against the independently derived month.
    assert reported["local_month_mismatch"] == ["O1", "O2"]
    assert reported["invalid_derived_timestamps"] == []
    # The check itself must be reported as failing, not merely sampled in details.
    assert next(c for c in result["checks"] if c["name"] == "derived_event_months")["passed"] is False
    assert json.loads(json.dumps(details)) == details
