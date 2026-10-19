"""Tests for :mod:`local_data_agent.holdout_fixtures`.

These tests exercise the real generator on small fixtures.  They assert
*structural* properties (row counts, awareness, ordering, month labels) and
never compare source financial amounts against model results.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from local_data_agent.dataset_v2 import DatasetError, generate_dataset
from local_data_agent import holdout_fixtures as hf

ORDERS = 400
TICKETS = 120
DOCS = 60

SYDNEY = "Australia/Sydney"


def _sydney():
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(SYDNEY)
    except Exception:  # pragma: no cover
        return timezone.utc


def _aware(value: str) -> datetime:
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    assert dt.tzinfo is not None and dt.utcoffset() is not None, value
    return dt


def _fixture_dir(root: Path, fixture: str) -> Path:
    return root / "var" / "sources" / fixture


def _connect(root: Path, fixture: str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(_fixture_dir(root, fixture) / "business.sqlite"))
    conn.row_factory = sqlite3.Row
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> list:
    return [r[1] for r in conn.execute(f"PRAGMA table_info('{table}')")]


def _rows(conn: sqlite3.Connection, table: str) -> list:
    return [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]


def _tables(conn: sqlite3.Connection) -> list:
    return sorted(
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
            " AND name NOT LIKE 'sqlite_%'"
        )
    )


def _in_feb_2026_sydney(value) -> bool:
    dt = _aware(value)
    local = dt.astimezone(_sydney())
    return local.year == 2026 and local.month == 2


def _snapshot(root: Path, fixture: str) -> dict:
    """All-table raw row content, for determinism comparison."""
    conn = _connect(root, fixture)
    try:
        return {t: _rows(conn, t) for t in _tables(conn)}
    finally:
        conn.close()


def _manifest(root: Path, fixture: str) -> dict:
    """Build a source manifest from the registry, with absolute paths."""
    folder = _fixture_dir(root, fixture)
    registry = json.loads((folder / "registry.json").read_text(encoding="utf-8"))
    sources = []
    for source in registry["sources"]:
        entry = dict(source)
        path = Path(entry["path"])
        entry["path"] = str(path if path.is_absolute() else (folder / path).resolve())
        sources.append(entry)
    return {
        "sources": sources,
        "metric_contracts": registry.get("metric_contracts", []),
        "documents": registry.get("documents", []),
    }


# --------------------------------------------------------------------------
# parameter validation
# --------------------------------------------------------------------------


def test_unknown_profile_rejected(tmp_path):
    with pytest.raises(DatasetError):
        hf.generate_holdout(tmp_path, "fx-bad-profile", 7, profile="nope")
    assert not (tmp_path / "var" / "sources" / "fx-bad-profile").exists()


@pytest.mark.parametrize(
    "fixture",
    ["../escape", "a/b", "a\\b", "", "with space", ".", "..", "a.b", "a$b"],
)
def test_unsafe_identifier_rejected(tmp_path, fixture):
    with pytest.raises(DatasetError):
        hf.generate_holdout(tmp_path, fixture, 7)


def test_unsafe_identifier_creates_nothing(tmp_path):
    with pytest.raises(DatasetError):
        hf.generate_holdout(tmp_path, "bad/name", 7)
    sources = tmp_path / "var" / "sources"
    if sources.exists():
        assert list(sources.iterdir()) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order_count": 99},
        {"order_count": 100001},
        {"ticket_count": -1},
        {"ticket_count": 10001},
        {"document_count": 39},
        {"document_count": 301},
    ],
)
def test_bad_counts_rejected(tmp_path, kwargs):
    args = {"order_count": ORDERS, "ticket_count": TICKETS, "document_count": DOCS}
    args.update(kwargs)
    with pytest.raises(DatasetError):
        hf.generate_holdout(tmp_path, "fx-bad-count", 7, **args)


def test_refuse_overwrite_of_existing_fixture(tmp_path):
    registry = hf.generate_holdout(
        tmp_path, "fx-keep", 11, order_count=ORDERS, ticket_count=TICKETS, document_count=DOCS
    )
    assert registry["holdout_profile"]["profile"] == "baseline"

    marker = _fixture_dir(tmp_path, "fx-keep") / "registry.json"
    before = marker.read_text(encoding="utf-8")

    with pytest.raises(DatasetError):
        hf.generate_holdout(
            tmp_path, "fx-keep", 11, order_count=ORDERS, ticket_count=TICKETS, document_count=DOCS
        )
    assert marker.read_text(encoding="utf-8") == before


def test_refuse_overwrite_of_broken_symlink(tmp_path):
    sources = tmp_path / "var" / "sources"
    sources.mkdir(parents=True)
    target = sources / "fx-symlink"
    try:
        os.symlink(str(sources / "does-not-exist"), str(target))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    assert not target.exists()  # broken
    with pytest.raises(DatasetError):
        hf.generate_holdout(
            tmp_path, "fx-symlink", 7, order_count=ORDERS, ticket_count=TICKETS, document_count=DOCS
        )


def test_no_leftover_staging_dirs(tmp_path):
    hf.generate_holdout(
        tmp_path, "fx-clean", 3, order_count=ORDERS, ticket_count=TICKETS, document_count=DOCS
    )
    leftovers = [p for p in (tmp_path / "var" / "sources").iterdir() if p.name.startswith(".holdout-")]
    assert leftovers == []


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_determinism_same_seed_and_profile(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    hf.generate_holdout(a, "fx-a", 21, "empty-base", ORDERS, TICKETS, DOCS)
    hf.generate_holdout(b, "fx-b", 21, "empty-base", ORDERS, TICKETS, DOCS)

    snap_a = _snapshot(a, "fx-a")
    snap_b = _snapshot(b, "fx-b")
    assert set(snap_a) == set(snap_b)
    assert snap_a == snap_b


def test_different_fixtures_differ(tmp_path):
    hf.generate_holdout(tmp_path, "fx-one", 21, "baseline", ORDERS, TICKETS, DOCS)
    hf.generate_holdout(tmp_path, "fx-two", 22, "baseline", ORDERS, TICKETS, DOCS)
    assert _snapshot(tmp_path, "fx-one") != _snapshot(tmp_path, "fx-two")


# --------------------------------------------------------------------------
# empty-base
# --------------------------------------------------------------------------


def test_empty_base_has_no_successful_february_orders(tmp_path):
    hf.generate_holdout(
        tmp_path, "fx-eb", 31, "empty-base", ORDERS, TICKETS, DOCS
    )
    conn = _connect(tmp_path, "fx-eb")
    try:
        paid = [
            r
            for r in _rows(conn, "orders")
            if r["status"] == "paid" and r["paid_at"] is not None
        ]
        assert paid, "expected some paid orders"
        assert not [r for r in paid if _in_feb_2026_sydney(r["paid_at"])]

        succeeded = [
            r for r in _rows(conn, "refunds") if r["status"] == "succeeded"
        ]
        assert succeeded, "expected some succeeded refunds"
        assert not [
            r for r in succeeded if _in_feb_2026_sydney(r["refunded_at"])
        ]
    finally:
        conn.close()


def test_empty_base_keeps_non_paid_february_orders(tmp_path):
    """Non-paid Feb orders are untouched, so attempt coverage stays."""
    hf.generate_holdout(tmp_path, "fx-eb2", 31, "empty-base", ORDERS, TICKETS, DOCS)
    conn = _connect(tmp_path, "fx-eb2")
    try:
        cols = _columns(conn, "orders")
        assert "attempted_at" in cols
        non_paid_feb = [
            r
            for r in _rows(conn, "orders")
            if r["status"] != "paid"
            and r["attempted_at"] is not None
            and _in_feb_2026_sydney(r["attempted_at"])
        ]
        assert non_paid_feb, "non-paid February attempts must remain"
    finally:
        conn.close()


def test_empty_base_january_boundary_numeral(tmp_path):
    """Shifted rows land on 2026-01-31 local 23:59:59.999998 Sydney."""
    hf.generate_holdout(tmp_path, 'fx-original-orders', 31, 'baseline', ORDERS, TICKETS, DOCS)
    original = _connect(tmp_path, 'fx-original-orders')
    try:
        shifted_ids = {r['order_id'] for r in _rows(original, 'orders')
                       if r['status'] == 'paid' and _in_feb_2026_sydney(r['paid_at'])}
    finally:
        original.close()
    hf.generate_holdout(tmp_path, "fx-eb3", 31, "empty-base", ORDERS, TICKETS, DOCS)
    conn = _connect(tmp_path, "fx-eb3")
    try:
        jan = [
            r
            for r in _rows(conn, "orders")
            if r["status"] == "paid"
            and r["paid_at"] is not None
            and r['order_id'] in shifted_ids
        ]
        assert jan, "expected shifted orders in January"
        for r in jan:
            local = _aware(r["paid_at"]).astimezone(_sydney())
            assert (local.year, local.month, local.day) == (2026, 1, 31)
            assert local.hour == 23 and local.minute == 59 and local.second == 59
            assert local.microsecond == 999998
            assert r["order_date"] == "2026-01-31"
    finally:
        conn.close()


def test_empty_base_refund_boundary_numeral(tmp_path):
    hf.generate_holdout(tmp_path, 'fx-original-refunds', 31, 'baseline', ORDERS, TICKETS, DOCS)
    original = _connect(tmp_path, 'fx-original-refunds')
    try:
        shifted_ids = {r['refund_id'] for r in _rows(original, 'refunds')
                       if r['status'] == 'succeeded' and _in_feb_2026_sydney(r['refunded_at'])}
    finally:
        original.close()
    hf.generate_holdout(tmp_path, "fx-eb4", 31, "empty-base", ORDERS, TICKETS, DOCS)
    conn = _connect(tmp_path, "fx-eb4")
    try:
        shifted = [
            r
            for r in _rows(conn, "refunds")
            if r["status"] == "succeeded" and r['refund_id'] in shifted_ids
        ]
        assert shifted, "expected shifted refunds"
        for r in shifted:
            local = _aware(r["refunded_at"]).astimezone(_sydney())
            assert (local.year, local.month, local.day) == (2026, 1, 31)
            assert local.hour == 23 and local.minute == 59 and local.second == 59
            assert local.microsecond == 999999
    finally:
        conn.close()


def test_empty_base_keeps_march_events(tmp_path):
    """Other months keep real calendar events (nothing is blanket-cleared)."""
    hf.generate_holdout(tmp_path, "fx-eb5", 31, "empty-base", ORDERS, TICKETS, DOCS)
    conn = _connect(tmp_path, "fx-eb5")
    try:
        march = [
            r
            for r in _rows(conn, "orders")
            if r["status"] == "paid"
            and r["paid_at"] is not None
            and _aware(r["paid_at"]).astimezone(_sydney()).month == 3
        ]
        assert march, "March payments must survive"

        march_refunds = [
            r
            for r in _rows(conn, "refunds")
            if r["status"] == "succeeded"
            and _aware(r["refunded_at"]).astimezone(_sydney()).month == 3
        ]
        assert march_refunds, "March refunds must survive"
    finally:
        conn.close()


# --------------------------------------------------------------------------
# channel-gap
# --------------------------------------------------------------------------


def test_channel_gap_zeroes_only_web_february(tmp_path):
    hf.generate_holdout(
        tmp_path, "fx-cg", 41, "channel-gap", ORDERS, TICKETS, DOCS
    )
    conn = _connect(tmp_path, "fx-cg")
    try:
        orders = _rows(conn, "orders")
        paid = [r for r in orders if r["status"] == "paid" and r["paid_at"] is not None]

        web_feb = [
            r
            for r in paid
            if r["channel_code"] == "web" and _in_feb_2026_sydney(r["paid_at"])
        ]
        other_feb = [
            r
            for r in paid
            if r["channel_code"] != "web" and _in_feb_2026_sydney(r["paid_at"])
        ]
        assert not web_feb
        assert other_feb, "non-web February payments must remain"

        web_ids = {r["order_id"] for r in orders if r["channel_code"] == "web"}

        succeeded = [r for r in _rows(conn, "refunds") if r["status"] == "succeeded"]
        web_refunds_feb = [
            r
            for r in succeeded
            if r["order_id"] in web_ids and _in_feb_2026_sydney(r["refunded_at"])
        ]
        other_refunds_feb = [
            r
            for r in succeeded
            if r["order_id"] not in web_ids and _in_feb_2026_sydney(r["refunded_at"])
        ]
        assert not web_refunds_feb
        assert other_refunds_feb, "refunds of non-web orders must remain"
    finally:
        conn.close()


def test_channel_gap_baseline_untouched_for_other_channels(tmp_path):
    a = tmp_path / "cg-a"
    b = tmp_path / "cg-b"
    hf.generate_holdout(a, "fx-x", 51, "baseline", ORDERS, TICKETS, DOCS)
    hf.generate_holdout(b, "fx-y", 51, "channel-gap", ORDERS, TICKETS, DOCS)

    snap_a = _snapshot(a, "fx-x")
    snap_b = _snapshot(b, "fx-y")

    base_orders = {r["order_id"]: r for r in snap_a["orders"]}
    gap_orders = {r["order_id"]: r for r in snap_b["orders"]}
    assert set(base_orders) == set(gap_orders)

    for oid, before in base_orders.items():
        after = gap_orders[oid]
        if before["channel_code"] == "web":
            continue
        assert before == after, f"non-web order {oid} changed under channel-gap"


# --------------------------------------------------------------------------
# timestamp synchronisation & invariants
# --------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["baseline", "empty-base", "channel-gap"])
def test_all_raw_timestamps_aware_and_month_correct(tmp_path, profile):
    fixture = f"fx-ts-{profile}"
    hf.generate_holdout(tmp_path, fixture, 61, profile, ORDERS, TICKETS, DOCS)
    conn = _connect(tmp_path, fixture)
    try:
        for r in _rows(conn, "orders"):
            if r["paid_at"] is not None:
                assert _aware(r["paid_at"])
            if r["attempted_at"] is not None:
                assert _aware(r["attempted_at"])
            if r["local_month"] is not None:
                local = _aware(r["paid_at"] or r["attempted_at"]).astimezone(_sydney())
                assert r["local_month"] == f"{local.year:04d}-{local.month:02d}"

        for r in _rows(conn, "refunds"):
            assert _aware(r["refunded_at"])
            local = _aware(r["refunded_at"]).astimezone(_sydney())
            assert r["local_month"] == f"{local.year:04d}-{local.month:02d}"
    finally:
        conn.close()


@pytest.mark.parametrize("profile", ["empty-base", "channel-gap"])
def test_payment_and_order_timestamps_synchronised(tmp_path, profile):
    hf.generate_holdout(tmp_path, 'fx-original-sync', 71, 'baseline', ORDERS, TICKETS, DOCS)
    original = _connect(tmp_path, 'fx-original-sync')
    try:
        before_payments = {r['payment_id']: r for r in _rows(original, 'payments')}
        shifted_ids = {r['order_id'] for r in _rows(original, 'orders')
                       if r['status'] == 'paid' and _in_feb_2026_sydney(r['paid_at'])
                       and (profile == 'empty-base' or r['channel_code'] == 'web')}
    finally:
        original.close()
    fixture = f"fx-sync-{profile}"
    hf.generate_holdout(tmp_path, fixture, 71, profile, ORDERS, TICKETS, DOCS)
    conn = _connect(tmp_path, fixture)
    try:
        orders = {r["order_id"]: r for r in _rows(conn, "orders")}
        for p in _rows(conn, "payments"):
            if p["status"] != "paid":
                continue
            order = orders.get(p["order_id"])
            if order is None or order["paid_at"] is None:
                continue
            assert _aware(p["paid_at"]) == _aware(order["paid_at"])
            if "attempted_at" in p and p["attempted_at"] is not None:
                assert _aware(p["attempted_at"]) == _aware(order["attempted_at"])
            if p['order_id'] not in shifted_ids:
                assert p == before_payments[p['payment_id']]
            elif "created_at" in p and p["created_at"] is not None:
                assert _aware(p["created_at"]) == _aware(order["paid_at"])
    finally:
        conn.close()


def test_succeeded_refund_parent_ordering_and_amounts(tmp_path):
    fixture = "fx-order"
    hf.generate_holdout(tmp_path, fixture, 81, "empty-base", ORDERS, TICKETS, DOCS)
    conn = _connect(tmp_path, fixture)
    try:
        orders = {r["order_id"]: r for r in _rows(conn, "orders")}
        succeeded = [r for r in _rows(conn, "refunds") if r["status"] == "succeeded"]
        assert succeeded, "expected succeeded refunds"
        for r in succeeded:
            parent = orders[r["order_id"]]
            assert _aware(parent["paid_at"]) <= _aware(r["refunded_at"])
            assert r["refund_cents"] <= parent["paid_cents"]
            assert r["status"] == "succeeded"
            assert parent["status"] == "paid"
    finally:
        conn.close()


_PK = {
    "orders": "order_id",
    "payments": "payment_id",
    "refunds": "refund_id",
}


def test_ids_and_amounts_preserved(tmp_path):
    a = tmp_path / "id-a"
    b = tmp_path / "id-b"
    hf.generate_holdout(a, "fx-base", 91, "baseline", ORDERS, TICKETS, DOCS)
    hf.generate_holdout(b, "fx-eb", 91, "empty-base", ORDERS, TICKETS, DOCS)

    snap_a, snap_b = _snapshot(a, "fx-base"), _snapshot(b, "fx-eb")
    for table, pk in _PK.items():
        base = {r[pk]: r for r in snap_a[table]}
        shifted = {r[pk]: r for r in snap_b[table]}
        assert set(base) == set(shifted), f"{table} ids changed"
        for key, before in base.items():
            after = shifted[key]
            for field in ("paid_cents", "amount_cents", "refund_cents", "status"):
                if field in before:
                    assert before[field] == after[field], f"{table}.{field} for {key}"


def test_other_tables_unchanged(tmp_path):
    a = tmp_path / "ot-a"
    b = tmp_path / "ot-b"
    hf.generate_holdout(a, "fx-base", 101, "baseline", ORDERS, TICKETS, DOCS)
    hf.generate_holdout(b, "fx-eb", 101, "empty-base", ORDERS, TICKETS, DOCS)

    snap_a, snap_b = _snapshot(a, "fx-base"), _snapshot(b, "fx-eb")
    assert set(snap_a) == set(snap_b)
    for table in snap_a:
        if table in ("orders", "payments", "refunds"):
            continue
        assert snap_a[table] == snap_b[table], f"table {table} changed"


# --------------------------------------------------------------------------
# registry / metadata
# --------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["baseline", "empty-base", "channel-gap"])
def test_registry_metadata_and_relative_paths(tmp_path, profile):
    fixture = f"fx-reg-{profile}"
    registry = hf.generate_holdout(tmp_path, fixture, 111, profile, ORDERS, TICKETS, DOCS)

    meta = registry["holdout_profile"]
    assert meta["version"] == "holdout-calendar/1"
    assert meta["profile"] == profile
    assert meta["seed"] == 111
    assert meta["base_month"] == "2026-02"
    assert "scope" in meta

    folder = _fixture_dir(tmp_path, fixture)
    for source in registry["sources"]:
        assert not os.path.isabs(source["path"])
        assert (folder / source["path"]).exists()

    assert registry["sources"], "registry must list sources"
    assert any(s["source_id"] == "business" for s in registry["sources"])

    on_disk = json.loads((folder / "registry.json").read_text(encoding="utf-8"))
    assert on_disk == registry


_LEAKAGE_KEYS = {
    "gold",
    "reference_sql",
    "expected_total",
    "expected_totals",
    "ledger",
}


def _leaked_keys(node, path="") -> list:
    """Every mapping key anywhere in ``node`` that names gold/reference data."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and key.strip().lower() in _LEAKAGE_KEYS:
                found.append(here)
            found.extend(_leaked_keys(value, here))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_leaked_keys(value, f"{path}[{index}]"))
    return found


def test_registry_has_no_gold_or_totals(tmp_path):
    fixture = "fx-private-check"
    registry = hf.generate_holdout(
        tmp_path, fixture, 121, "empty-base", ORDERS, TICKETS, DOCS
    )

    # exact key names, not substrings: schema columns such as
    # ``order_total_cents`` and ids containing "gold" are legitimate.
    assert _leaked_keys(registry) == []

    folder = _fixture_dir(tmp_path, fixture)
    files = [p for p in folder.rglob("*") if p.is_file()]
    assert any(p.name == "registry.json" for p in files)
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        try:
            payload = json.loads(text)
        except ValueError:
            continue  # non-JSON source artefact: no key namespace to inspect
        assert _leaked_keys(payload) == [], f"leaked keys in {path.name}"


def test_holdout_profile_has_no_computed_financials(tmp_path):
    fixture = "fx-private-meta"
    registry = hf.generate_holdout(
        tmp_path, fixture, 121, "empty-base", ORDERS, TICKETS, DOCS
    )
    profile = registry["holdout_profile"]

    assert set(profile) <= {"version", "profile", "seed", "base_month", "scope", "transformed_counts"}
    assert set(profile) >= {"version", "profile", "seed", "base_month", "scope"}

    # the only added block is count metadata: no sums, no amounts, no gold
    counts = profile.get("transformed_counts")
    assert isinstance(counts, dict) and counts
    assert all(key.endswith("_shifted") for key in counts)
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in counts.values())

    def _check_scalars_only(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                _check_scalars_only(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                _check_scalars_only(value, f"{path}[{index}]")
        else:
            assert isinstance(node, (str, int, bool)) or node is None, (
                f"holdout_profile{path} carries a computed value: {node!r}"
            )

    _check_scalars_only(profile, "")
    assert _leaked_keys(profile) == []


def test_transformed_counts_are_counts_not_sums(tmp_path):
    registry = hf.generate_holdout(
        tmp_path, "fx-counts", 131, "empty-base", ORDERS, TICKETS, DOCS
    )
    counts = registry["holdout_profile"].get("transformed_counts")
    assert isinstance(counts, dict) and counts
    for key, value in counts.items():
        assert key.endswith("_shifted")
        assert isinstance(value, int) and not isinstance(value, bool)


# --------------------------------------------------------------------------
# source integrity
# --------------------------------------------------------------------------


def test_sources_validate(tmp_path):
    fixture = "fx-valid"
    hf.generate_holdout(tmp_path, fixture, 141, "empty-base", ORDERS, TICKETS, DOCS)
    manifest = _manifest(tmp_path, fixture)

    folder = _fixture_dir(tmp_path, fixture)
    assert (folder / "channels.csv").exists()
    assert (folder / "tickets.jsonl").exists()
    assert (folder / "business.sqlite").exists()
    assert (folder / "documents").is_dir()

    for source in manifest["sources"]:
        assert os.path.isabs(source["path"])
        assert Path(source["path"]).exists()

    assert manifest["metric_contracts"], "metric_contracts must be preserved"

    try:
        from local_data_agent.validation import validate_sources
    except ImportError:  # pragma: no cover - validator not available
        pytest.skip("validate_sources unavailable")
    result = validate_sources(manifest)
    assert result is None or result is True or bool(result)


def test_source_files_unchanged_by_profile(tmp_path):
    """Non-DB artefacts are byte-identical across profiles."""
    a = tmp_path / "sf-a"
    b = tmp_path / "sf-b"
    hf.generate_holdout(a, "fx-base", 151, "baseline", ORDERS, TICKETS, DOCS)
    hf.generate_holdout(b, "fx-eb", 151, "empty-base", ORDERS, TICKETS, DOCS)

    fa, fb = _fixture_dir(a, "fx-base"), _fixture_dir(b, "fx-eb")
    for name in ("channels.csv", "tickets.jsonl"):
        assert (fa / name).read_bytes() == (fb / name).read_bytes()

    docs_a = sorted(p.name for p in (fa / "documents").iterdir())
    docs_b = sorted(p.name for p in (fb / "documents").iterdir())
    assert docs_a == docs_b
    for name in docs_a:
        assert (fa / "documents" / name).read_bytes() == (
            fb / "documents" / name
        ).read_bytes()


# --------------------------------------------------------------------------
# atomicity under injected failure
# --------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["baseline", "empty-base", "channel-gap"])
def test_injected_mutation_failure_never_publishes(tmp_path, monkeypatch, profile):
    fixture = f"fx-fail-{profile}"
    sources = tmp_path / "var" / "sources"

    def boom(folder):
        raise RuntimeError("injected mutation failure")

    monkeypatch.setitem(hf._PROFILE_MUTATORS, profile, boom)

    with pytest.raises(RuntimeError):
        hf.generate_holdout(tmp_path, fixture, 161, profile, ORDERS, TICKETS, DOCS)

    assert not (sources / fixture).exists(), "target must not be published"
    leftovers = [p for p in sources.iterdir() if p.name.startswith(".holdout-")]
    assert leftovers == [], "staging directory must be cleaned up"


def test_injected_registry_failure_never_publishes(tmp_path, monkeypatch):
    fixture = "fx-fail-reg"
    sources = tmp_path / "var" / "sources"

    def explode(*args, **kwargs):
        raise RuntimeError("injected registry failure")

    monkeypatch.setattr(hf, "_write_registry", explode)

    with pytest.raises(RuntimeError):
        hf.generate_holdout(tmp_path, fixture, 171, "empty-base", ORDERS, TICKETS, DOCS)

    assert not (sources / fixture).exists()
    assert [p for p in sources.iterdir() if p.name.startswith(".holdout-")] == []


# --------------------------------------------------------------------------
# generator sanity (real, installed package)
# --------------------------------------------------------------------------


def test_real_generator_shape(tmp_path):
    registry = generate_dataset(tmp_path, "fx-plain", 181, ORDERS, TICKETS, DOCS)
    folder = _fixture_dir(tmp_path, "fx-plain")
    assert (folder / "business.sqlite").exists()
    assert (folder / "registry.json").exists()
    assert any(s["source_id"] == "business" for s in registry["sources"])

    conn = _connect(tmp_path, "fx-plain")
    try:
        assert {"orders", "payments", "refunds"}.issubset(set(_tables(conn)))
    finally:
        conn.close()


def test_malformed_timestamp_rejected(tmp_path):
    fixture = "fx-malformed"
    hf.generate_holdout(tmp_path, fixture, 191, "baseline", ORDERS, TICKETS, DOCS)
    folder = _fixture_dir(tmp_path, fixture)

    conn = sqlite3.connect(str(folder / "business.sqlite"))
    try:
        conn.execute("UPDATE orders SET paid_at = 'not-a-timestamp' WHERE status = 'paid'")
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(DatasetError):
        hf._parse_aware("not-a-timestamp")
    with pytest.raises(DatasetError):
        hf._parse_aware("2026-02-10 12:00:00")  # naive
