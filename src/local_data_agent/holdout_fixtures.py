"""Holdout fixture generator for calendar stress profiles.

Builds a holdout variant of an existing generated dataset by staging the
generator output inside a temporary directory, mutating the staged copy, and
publishing it atomically with ``os.replace``.

Profiles:
  * ``baseline``     - no database mutation, metadata only.
  * ``empty-base``   - move Feb-2026 Sydney successful orders/refunds across the
                       local January/February boundary.
  * ``channel-gap``  - same shift, but limited to ``channel_code='web'``.

No gold values, expected sums, or reference SQL are ever written.
"""

from __future__ import annotations

import os
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .dataset_v2 import DatasetError, generate_dataset

__all__ = ["generate_holdout", "PROFILES", "SAFE_IDENTIFIER"]

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

PROFILES = ("baseline", "empty-base", "channel-gap")

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]+$")

SYDNEY = "Australia/Sydney"
BASE_MONTH = "2026-02"

HOLDOUT_PROFILE_VERSION = "holdout-calendar/1"
HOLDOUT_SCOPE = (
    "financial calendar stress; nonfinancial operational dates preserved"
)

ORDER_MIN, ORDER_MAX = 100, 100000
TICKET_MIN, TICKET_MAX = 0, 10000
DOC_MIN, DOC_MAX = 40, 300

PAID_ORDER_STATUS = "paid"
SUCCEEDED_STATUS = "succeeded"

# local Jan-2026 boundary targets: the last two Sydney-local instants of
# 2026-01-31, i.e. just before 2026-02-01T00:00:00+11:00 (AEDT).  The tzinfo is
# the authoritative Sydney zone; _iso_utc converts the instant to UTC.  Using a
# fixed-offset/UTC tzinfo here would preserve the wrong instant and land on Feb 1.
_TARGET_ORDER_LOCAL = datetime(
    2026, 1, 31, 23, 59, 59, 999998, tzinfo=ZoneInfo(SYDNEY)
)
_TARGET_REFUND_LOCAL = datetime(
    2026, 1, 31, 23, 59, 59, 999999, tzinfo=ZoneInfo(SYDNEY)
)

# tables we touch; everything else must stay untouched
_ORDER_TABLE = "orders"
_PAYMENT_TABLE = "payments"
_REFUND_TABLE = "refunds"


# --------------------------------------------------------------------------
# timezone helpers (zoneinfo only; no fixed-offset fallback)
# --------------------------------------------------------------------------


def _sydney_tz() -> ZoneInfo:
    """Return the authoritative ``Australia/Sydney`` zone.

    There is deliberately no fixed-offset/UTC fallback: a missing tz database
    must fail loudly rather than silently relabel UTC instants as Sydney local
    time (which would move boundary rows back across the local month edge).
    """
    try:
        return ZoneInfo(SYDNEY)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise DatasetError(
            f"timezone database unavailable for {SYDNEY!r}; "
            "install tzdata (or the 'tzdata' package on Windows)"
        ) from exc


def _to_sydney(value: str) -> datetime:
    """Parse an aware timestamp string and convert it to Sydney local time."""
    dt = _parse_aware(value)
    return dt.astimezone(_sydney_tz())


def _parse_aware(value: Any) -> datetime:
    """Parse a timestamp, refusing naive or malformed values."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise DatasetError("empty timestamp value in source data")
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise DatasetError(f"malformed timestamp: {value!r}") from exc
    else:
        raise DatasetError(f"unsupported timestamp value: {value!r}")

    if dt.tzinfo is None or dt.utcoffset() is None:
        raise DatasetError(f"naive timestamp is not allowed: {value!r}")
    return dt


def _iso_utc(dt: datetime) -> str:
    """Serialise an aware datetime as a UTC ISO-8601 string."""
    return dt.astimezone(timezone.utc).isoformat()


def _shift_order_stamp(dt: datetime) -> str:
    """Return the ISO-UTC stamp for the Jan/Feb Sydney-local order boundary."""
    return _iso_utc(_TARGET_ORDER_LOCAL)


def _shift_refund_stamp(dt: datetime) -> str:
    """Return the ISO-UTC stamp for the Jan/Feb Sydney-local refund boundary."""
    return _iso_utc(_TARGET_REFUND_LOCAL)


def _is_target_month(dt: datetime) -> bool:
    """True when the *own* timestamp falls in Feb 2026 Sydney local time."""
    local = dt.astimezone(_sydney_tz())
    return local.year == 2026 and local.month == 2


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------


def _validate_counts(order_count: int, ticket_count: int, document_count: int) -> None:
    if not isinstance(order_count, int) or isinstance(order_count, bool):
        raise DatasetError("order_count must be an integer")
    if not ORDER_MIN <= order_count <= ORDER_MAX:
        raise DatasetError(
            f"order_count must be between {ORDER_MIN} and {ORDER_MAX}, got {order_count}"
        )
    if not isinstance(ticket_count, int) or isinstance(ticket_count, bool):
        raise DatasetError("ticket_count must be an integer")
    if not TICKET_MIN <= ticket_count <= TICKET_MAX:
        raise DatasetError(
            f"ticket_count must be between {TICKET_MIN} and {TICKET_MAX}, got {ticket_count}"
        )
    if not isinstance(document_count, int) or isinstance(document_count, bool):
        raise DatasetError("document_count must be an integer")
    if not DOC_MIN <= document_count <= DOC_MAX:
        raise DatasetError(
            f"document_count must be between {DOC_MIN} and {DOC_MAX}, got {document_count}"
        )


def _validate_fixture(fixture: str) -> str:
    if not isinstance(fixture, str) or not fixture:
        raise DatasetError("fixture must be a non-empty string")
    if not SAFE_IDENTIFIER.match(fixture):
        raise DatasetError(f"unsafe fixture identifier: {fixture!r}")
    if fixture in (".", ".."):
        raise DatasetError(f"unsafe fixture identifier: {fixture!r}")
    return fixture


def _validate_profile(profile: str) -> str:
    if profile not in PROFILES:
        raise DatasetError(
            f"unknown holdout profile {profile!r}; expected one of {PROFILES}"
        )
    return profile


def _path_exists(path: Path) -> bool:
    """True for real files/dirs and broken symlinks alike."""
    return os.path.lexists(str(path))


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    if not rows:
        raise DatasetError(f"missing table {table!r} in generated database")
    return [row[1] for row in rows]


def _has_column(columns: List[str], name: str) -> bool:
    if name in columns:
        return True
    lowered = {c.lower() for c in columns}
    return name.lower() in lowered


# --------------------------------------------------------------------------
# mutation
# --------------------------------------------------------------------------


def _apply_baseline(folder: Path) -> Dict[str, int]:
    """baseline performs no database mutation."""
    return {}


def _apply_empty_base(folder: Path) -> Dict[str, int]:
    return _apply_calendar_shift(folder, channel=None)


def _apply_channel_gap(folder: Path) -> Dict[str, int]:
    return _apply_calendar_shift(folder, channel="web")


def _apply_calendar_shift(folder: Path, channel: Optional[str]) -> Dict[str, int]:
    """Move qualifying Feb-2026 Sydney rows across the Jan/Feb local boundary."""
    db_path = folder / "business.sqlite"
    if not db_path.exists():
        raise DatasetError(f"staged database missing: {db_path}")

    conn = sqlite3.connect(str(db_path))
    counts = {
        "orders_shifted": 0,
        "payments_shifted": 0,
        "refunds_shifted": 0,
    }
    try:
        conn.execute("BEGIN IMMEDIATE")
        order_cols = _table_columns(conn, _ORDER_TABLE)
        payment_cols = _table_columns(conn, _PAYMENT_TABLE)
        refund_cols = _table_columns(conn, _REFUND_TABLE)

        counts["orders_shifted"] = _shift_orders(conn, order_cols, channel)
        counts["payments_shifted"] = _sync_payments(conn, payment_cols, channel)
        counts["refunds_shifted"] = _shift_refunds(
            conn, refund_cols, order_cols, channel
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return counts


def _shift_orders(
    conn: sqlite3.Connection, columns: List[str], channel: Optional[str]
) -> int:
    """Shift successful Feb orders; leave non-paid February orders untouched."""
    if not _has_column(columns, "order_id") or not _has_column(columns, "paid_at"):
        raise DatasetError("orders table lacks order_id/paid_at")
    has_attempted = _has_column(columns, "attempted_at")
    has_channel = _has_column(columns, "channel_code")

    available = [
        c
        for c in ("order_id", "status", "paid_at", "attempted_at", "order_date", "local_month")
        if _has_column(columns, c)
    ]

    where = ""
    params: List[Any] = []
    if channel is not None:
        if not has_channel:
            raise DatasetError("orders table lacks channel_code for channel-gap")
        where = "WHERE channel_code = ?"
        params.append(channel)

    rows = conn.execute(
        "SELECT " + ", ".join(dict.fromkeys(available)) + f" FROM orders {where}",
        params,
    ).fetchall()
    names = list(dict.fromkeys(available))

    updates = []
    for row in rows:
        record = dict(zip(names, row))
        status = record.get("status")
        if status != PAID_ORDER_STATUS:
            continue  # non-paid February orders remain unchanged
        paid_at = record.get("paid_at")
        if paid_at is None:
            continue
        dt = _parse_aware(paid_at)
        if not _is_target_month(dt):
            continue

        new_stamp = _shift_order_stamp(dt)
        assignments = ["paid_at = ?"]
        values: List[Any] = [new_stamp]
        if has_attempted:
            attempted = record.get("attempted_at")
            if attempted is not None:
                _parse_aware(attempted)
                assignments.append("attempted_at = ?")
                values.append(new_stamp)
        if _has_column(columns, "order_date"):
            assignments.append("order_date = ?")
            values.append("2026-01-31")
        if _has_column(columns, "local_month"):
            assignments.append("local_month = ?")
            values.append("2026-01")
        values.append(record["order_id"])
        updates.append(
            (f"UPDATE orders SET {', '.join(assignments)} WHERE order_id = ?", values)
        )

    for sql, values in updates:
        conn.execute(sql, values)
    return len(updates)


def _sync_payments(
    conn: sqlite3.Connection, columns: List[str], channel: Optional[str]
) -> int:
    """Synchronise paid payments whose order was shifted to the new stamp."""
    if not _has_column(columns, "order_id"):
        raise DatasetError("payments table lacks order_id")

    join = ""
    params: List[Any] = []
    if channel is not None:
        join = "JOIN orders o ON o.order_id = p.order_id"
        where = "WHERE o.channel_code = ? AND o.status = 'paid'"
        params.append(channel)
    else:
        where = "WHERE p.status = 'paid'"

    select = "p.payment_id, p.order_id, p.paid_at, p.status"
    if _has_column(columns, "attempted_at"):
        select += ", p.attempted_at"
    if _has_column(columns, "created_at"):
        select += ", p.created_at"

    rows = conn.execute(f"SELECT {select} FROM payments p {join} {where}", params).fetchall()

    assignments = ["paid_at = ?"]
    if _has_column(columns, "attempted_at"):
        assignments.append("attempted_at = ?")
    if _has_column(columns, "created_at"):
        assignments.append("created_at = ?")

    updates = []
    for row in rows:
        paid_at = row[2]
        if paid_at is None:
            continue
        dt = _parse_aware(paid_at)
        if not _is_target_month(dt):
            continue
        new_stamp = _shift_order_stamp(dt)
        values: List[Any] = [new_stamp]
        if _has_column(columns, "attempted_at"):
            values.append(new_stamp)
        if _has_column(columns, "created_at"):
            values.append(new_stamp)
        values.append(row[0])
        updates.append(
            (f"UPDATE payments SET {', '.join(assignments)} WHERE payment_id = ?", values)
        )

    for sql, values in updates:
        conn.execute(sql, values)
    return len(updates)


def _shift_refunds(
    conn: sqlite3.Connection,
    refund_columns: List[str],
    order_columns: List[str],
    channel: Optional[str],
) -> int:
    """Shift succeeded Feb refunds; channel derived via the order FK."""
    if not _has_column(refund_columns, "refund_id"):
        raise DatasetError("refunds table lacks refund_id")
    if not _has_column(refund_columns, "refunded_at"):
        raise DatasetError("refunds table lacks refunded_at")

    select = "r.refund_id, r.order_id, r.refunded_at, r.status"
    has_attempted = _has_column(refund_columns, "attempted_at")
    has_month = _has_column(refund_columns, "local_month")
    if has_attempted:
        select += ", r.attempted_at"

    join = "JOIN orders o ON o.order_id = r.order_id"
    where = "WHERE r.status = 'succeeded'"
    params: List[Any] = []
    if channel is not None:
        if not _has_column(order_columns, "channel_code"):
            raise DatasetError("orders table lacks channel_code for channel-gap")
        where += " AND o.channel_code = ?"
        params.append(channel)

    rows = conn.execute(f"SELECT {select} FROM refunds r {join} {where}", params).fetchall()

    assignments = ["refunded_at = ?"]
    if has_attempted:
        assignments.append("attempted_at = ?")
    if has_month:
        assignments.append("local_month = ?")

    updates = []
    for row in rows:
        refunded_at = row[2]
        if refunded_at is None:
            continue
        dt = _parse_aware(refunded_at)
        if not _is_target_month(dt):
            continue

        parent = conn.execute(
            "SELECT paid_at FROM orders WHERE order_id = ?", (row[1],)
        ).fetchone()
        if parent is None or parent[0] is None:
            raise DatasetError(
                f"succeeded refund {row[0]!r} has no payable parent order"
            )
        parent_dt = _parse_aware(parent[0])
        new_stamp = _shift_refund_stamp(dt)
        new_dt = _parse_aware(new_stamp)

        # invariant: paid_at <= refunded_at must survive the transform
        if parent_dt > new_dt:
            raise DatasetError(
                f"refund {row[0]!r} would precede its parent payment timestamps"
            )

        values: List[Any] = [new_stamp]
        if has_attempted:
            values.append(new_stamp)
        if has_month:
            values.append("2026-01")
        values.append(row[0])
        updates.append(
            (f"UPDATE refunds SET {', '.join(assignments)} WHERE refund_id = ?", values)
        )

    for sql, values in updates:
        conn.execute(sql, values)
    return len(updates)


_PROFILE_MUTATORS = {
    "baseline": _apply_baseline,
    "empty-base": _apply_empty_base,
    "channel-gap": _apply_channel_gap,
}


# --------------------------------------------------------------------------
# registry handling
# --------------------------------------------------------------------------


def _attach_profile_metadata(
    registry: Dict[str, Any], profile: str, seed: int, counts: Dict[str, int]
) -> Dict[str, Any]:
    registry["holdout_profile"] = {
        "version": HOLDOUT_PROFILE_VERSION,
        "profile": profile,
        "seed": seed,
        "base_month": BASE_MONTH,
        "scope": HOLDOUT_SCOPE,
    }
    if counts:
        registry["holdout_profile"]["transformed_counts"] = dict(sorted(counts.items()))
    return registry


def _write_registry(folder: Path, registry: Dict[str, Any]) -> None:
    import json

    registry_path = folder / "registry.json"
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _relativise_paths(registry: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite absolute source paths so they stay relative to the fixture folder."""
    sources = registry.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            path = source.get("path")
            if isinstance(path, str) and os.path.isabs(path):
                source["path"] = os.path.basename(path)
    return registry


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def generate_holdout(
    project_root: Path,
    fixture: str,
    seed: int,
    profile: str = "baseline",
    order_count: int = 12000,
    ticket_count: int = 600,
    document_count: int = 60,
) -> Dict[str, Any]:
    """Generate a holdout fixture and publish it atomically.

    Returns the registry dict for the published fixture, with source paths
    relative to the fixture folder.
    """
    root = Path(project_root).resolve()
    fixture = _validate_fixture(fixture)
    profile = _validate_profile(profile)
    _validate_counts(order_count, ticket_count, document_count)

    source_root = root / "var" / "sources"
    source_root.mkdir(parents=True, exist_ok=True)
    target = source_root / fixture

    # refuse overwrite of an existing fixture (including a broken symlink)
    if _path_exists(target):
        raise DatasetError(f"refusing to overwrite existing fixture: {target}")

    with tempfile.TemporaryDirectory(prefix=".holdout-", dir=str(source_root)) as tmp:
        tmp_root = Path(tmp)
        registry = generate_dataset(
            tmp_root,
            fixture,
            seed,
            order_count,
            ticket_count,
            document_count,
        )

        staged = tmp_root / "var" / "sources" / fixture
        if not staged.is_dir():
            raise DatasetError(f"generator did not produce staged fixture: {staged}")

        counts = _PROFILE_MUTATORS[profile](staged)

        _relativise_paths(registry)
        _attach_profile_metadata(registry, profile, seed, counts)
        _write_registry(staged, registry)

        # re-check immediately before publishing
        if _path_exists(target):
            raise DatasetError(f"refusing to overwrite existing fixture: {target}")

        os.replace(str(staged), str(target))

    return registry
