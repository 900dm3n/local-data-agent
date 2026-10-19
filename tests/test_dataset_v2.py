"""DATA-001 generator checks.

These tests are deliberately small: they generate seed-controlled mini fixtures
and verify the structural, deterministic and contractual properties that the
independent evaluator will later re-check at full size.  They never use
``local_data_agent.fixtures.compute_gold`` as reference truth.

The suite encodes the money semantics the generator must satisfy: ``paid_at`` is
written only for genuinely successful orders, promotion discounts are allocated
into order lines, and inventory outflows mirror the purchased lines.
"""
from __future__ import annotations

import csv
import json
import random
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from local_data_agent.dataset_v2 import (
    AMOUNT_UNIT, BUSINESS_MONTHS, PROMOTION_TAKE_RATE, PROMOTION_WINDOWS, PROMOTION_DEFS,
    TABLE_ORDER, DatasetError, generate_dataset, local_month,
)

ZONE = ZoneInfo("Australia/Sydney")
SMALL = {"order_count": 400, "ticket_count": 120, "document_count": 60}
#: Larger sample used only by the distribution test, which needs enough paid
#: orders per (month, channel) cell to compare refund shares meaningfully.
DISTRIBUTION_ORDERS = 12000


def _folder(tmp_path: Path, fixture: str = "commerce-small") -> Path:
    return tmp_path / "var" / "sources" / fixture


def _query(folder: Path, sql: str) -> list[tuple]:
    with closing(sqlite3.connect(folder / "business.sqlite")) as db:
        return list(db.execute(sql))


def _rows(folder: Path, table: str) -> list[dict]:
    with closing(sqlite3.connect(folder / "business.sqlite")) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(f'SELECT * FROM "{table}"')]


def _tickets(folder: Path) -> list[dict]:
    return [json.loads(line) for line in (folder / "tickets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp("dataset-v2-base")
    registry = generate_dataset(root, fixture="commerce-base", seed=73, **SMALL)
    return root, registry, _folder(root, "commerce-base")


# --------------------------------------------------------------------------
# schema, contents and relations
# --------------------------------------------------------------------------

def test_exactly_24_meaningful_tables(base):
    _, registry, folder = base
    names = {row[0] for row in _query(folder, "SELECT name FROM sqlite_master WHERE type='table'")}
    assert names == set(TABLE_ORDER)
    assert len(names) == 24
    views = _query(folder, "SELECT name FROM sqlite_master WHERE type='view'")
    assert views == []
    business = next(s for s in registry["sources"] if s["source_id"] == "business")
    assert set(business["tables"]) == set(TABLE_ORDER)
    assert registry["manifest"]["row_counts"]["orders"] == SMALL["order_count"]


def test_every_table_has_rows_and_real_typed_columns(base):
    _, _, folder = base
    for table in TABLE_ORDER:
        info = _query(folder, f'PRAGMA table_info("{table}")')
        assert len(info) >= 3, table
        assert all(row[2] for row in info), table
        count = _query(folder, f'SELECT COUNT(*) FROM "{table}"')[0][0]
        # support_ticket_links is only empty when ticket_count is zero; this
        # fixture asks for 120 tickets, so every table must carry data.
        assert count > 0, table


def test_indexes_cover_transaction_time_and_order_relations(base):
    _, _, folder = base
    indexes = {row[0]: row[1] for row in _query(folder, "SELECT name, tbl_name FROM sqlite_master WHERE type='index'")}
    assert indexes.get("idx_orders_paid_at") == "orders"
    assert indexes.get("idx_refunds_refunded_at") == "refunds"
    assert indexes.get("idx_order_items_order") == "order_items"
    assert indexes.get("idx_refunds_order") == "refunds"
    assert indexes.get("idx_payments_order") == "payments"
    assert indexes.get("idx_shipments_order") == "shipments"


def test_entity_variety_is_not_placeholder_repetition(base):
    _, _, folder = base
    customers, products = _rows(folder, "customers"), _rows(folder, "products")
    categories, suppliers = _rows(folder, "categories"), _rows(folder, "suppliers")
    assert len({row["full_name"] for row in customers}) == len(customers)
    assert len({row["email"] for row in customers}) == len(customers)
    assert len({row["city"] for row in customers}) > 1
    assert len({row["segment"] for row in customers}) > 1
    assert len({row["product_name"] for row in products}) == len(products)
    assert len({row["unit_price_cents"] for row in products}) > 5
    assert len({row["category_name"] for row in categories}) == len(categories)
    assert len({row["supplier_name"] for row in suppliers}) == len(suppliers)
    assert len({row["category_id"] for row in products}) > 3


def test_order_items_money_and_foreign_keys(base):
    _, _, folder = base
    orders = {row["order_id"]: row for row in _rows(folder, "orders")}
    items = _rows(folder, "order_items")
    per_order: dict[str, int] = {}
    for item in items:
        assert item["order_id"] in orders
        assert item["quantity"] >= 1
        assert item["unit_price_cents"] > 0
        assert item["discount_cents"] >= 0
        assert item["discount_cents"] <= item["unit_price_cents"] * item["quantity"]
        # net line = gross line - allocated promotion discount.
        assert item["line_total_cents"] == item["unit_price_cents"] * item["quantity"] - item["discount_cents"]
        assert item["line_total_cents"] >= 0
        per_order[item["order_id"]] = per_order.get(item["order_id"], 0) + item["line_total_cents"]
    assert per_order
    assert all(1 <= sum(1 for item in items if item["order_id"] == oid) <= 3 for oid in per_order)
    for oid, total in per_order.items():
        order = orders[oid]
        # order_total_cents is the net line sum for every order, paid or not;
        # paid_cents only reflects it when the order actually succeeded.
        assert order["order_total_cents"] == total
        assert order["paid_cents"] == (total if order["status"] == "paid" else 0)
        assert order["order_total_cents"] > 0
    product_ids = {row["product_id"] for row in _rows(folder, "products")}
    assert all(item["product_id"] in product_ids for item in items)
    customer_ids = {row["customer_id"] for row in _rows(folder, "customers")}
    assert {row["customer_id"] for row in orders.values()} <= customer_ids


def test_non_paid_orders_keep_null_paid_at_and_zero_paid_cents(base):
    """The generator must not fabricate success timestamps to look complete."""
    _, _, folder = base
    orders = _rows(folder, "orders")
    unattempted = [row for row in orders if row["status"] != "paid"]
    assert unattempted, "the fixture must contain non-paid orders"
    for order in unattempted:
        assert order["paid_at"] is None, order["order_id"]
        assert order["paid_cents"] == 0, order["order_id"]
        # The attempt stamp still exists, is aware, and is what the month falls
        # back to when there is no payment event.
        attempted = datetime.fromisoformat(order["attempted_at"].replace("Z", "+00:00"))
        assert attempted.tzinfo is not None and attempted.utcoffset().total_seconds() == 0
        assert order["local_month"] == local_month(attempted)
    for order in orders:
        assert order["attempted_at"], order["order_id"]
        attempted = datetime.fromisoformat(order["attempted_at"].replace("Z", "+00:00"))
        created = datetime.fromisoformat(order["created_at"].replace("Z", "+00:00"))
        assert created <= attempted
        # Every order is attributed to a month inside the covered window, and the
        # attempt itself is the chosen event stamp, so it lies inside that window
        # too (whatever the order's eventual status).
        assert order["local_month"] in BUSINESS_MONTHS
        assert local_month(attempted) in BUSINESS_MONTHS
        if order["status"] == "paid":
            paid = datetime.fromisoformat(order["paid_at"].replace("Z", "+00:00"))
            assert attempted == paid
            assert order["local_month"] == local_month(paid)
            assert order["order_date"] == paid.astimezone(ZONE).date().isoformat()
            assert order["paid_cents"] == order["order_total_cents"] > 0
        else:
            assert order["local_month"] == local_month(attempted)
            assert order["order_date"] == attempted.astimezone(ZONE).date().isoformat()
    # Non-paid orders still persist the attempt in the payments ledger.
    payments = {row["order_id"]: row for row in _rows(folder, "payments")}
    for order in unattempted:
        payment = payments[order["order_id"]]
        assert payment["paid_at"] is None
        assert payment["amount_cents"] == 0
        assert payment["attempted_at"]


def test_promotions_are_genuinely_applied_to_lines(base):
    """order_promotions must describe a discount that really moved the totals."""
    _, _, folder = base
    orders = {row["order_id"]: row for row in _rows(folder, "orders")}
    items = _rows(folder, "order_items")
    promotions = {row["promotion_id"]: row for row in _rows(folder, "promotions")}
    applied = _rows(folder, "order_promotions")
    assert applied, "the fixture must contain at least one genuinely applied promotion"

    allocated: dict[str, int] = {}
    gross: dict[str, int] = {}
    for item in items:
        allocated[item["order_id"]] = allocated.get(item["order_id"], 0) + item["discount_cents"]
        gross[item["order_id"]] = gross.get(item["order_id"], 0) + item["unit_price_cents"] * item["quantity"]

    assert len({row["order_id"] for row in applied}) == len(applied), "at most one promotion per order"
    for row in applied:
        order = orders[row["order_id"]]
        promotion = promotions[row["promotion_id"]]
        assert row["discount_cents"] > 0
        # The recorded discount equals the sum of the line allocations.
        assert row["discount_cents"] == allocated[row["order_id"]]
        # ... and equals what discount_bps implies on the gross line sum.
        assert row["discount_cents"] == gross[row["order_id"]] * promotion["discount_bps"] // 10_000
        # ... and the promotion was actually live for this order's month/channel.
        start, end = PROMOTION_WINDOWS[row["promotion_id"]]
        assert start <= order["local_month"] <= end
        assert promotion["channel_code"] == order["channel_code"]
        # The discount is a real reduction, never larger than the order itself.
        assert 0 < row["discount_cents"] < order["order_total_cents"] + row["discount_cents"]
    # The discount really moved the money for every row, paid or not: the order
    # total is gross minus discount, and paid_cents mirrors it only when the
    # order actually settled (zero otherwise).
    for order_id, order in orders.items():
        discount = allocated.get(order_id, 0)
        assert order["order_total_cents"] == gross[order_id] - discount, order_id
        expected_paid = order["order_total_cents"] if order["status"] == "paid" else 0
        assert order["paid_cents"] == expected_paid, order_id
    # Partial take-up: not every eligible order uses the promotion.
    assert len(applied) < len(orders)
    # Orders with no promotion row were not discounted at all.
    undisounted = set(orders) - {row["order_id"] for row in applied}
    assert undisounted
    assert all(allocated.get(oid, 0) == 0 for oid in undisounted)
    # The take-up rate is a documented constant, not an arbitrary value.
    eligible = [oid for oid, order in orders.items()
                if any(code == order["channel_code"] and PROMOTION_WINDOWS[pid][0] <= order["local_month"]
                       <= PROMOTION_WINDOWS[pid][1] for pid, _, code, _ in PROMOTION_DEFS)]
    assert eligible
    assert 0 < len(applied) / len(eligible) < 1
    assert abs(len(applied) / len(eligible) - PROMOTION_TAKE_RATE) < 0.12


def test_payments_one_row_per_order_with_matching_amount(base):
    _, _, folder = base
    orders = {row["order_id"]: row for row in _rows(folder, "orders")}
    payments = _rows(folder, "payments")
    assert len(payments) == len(orders)
    assert len({row["order_id"] for row in payments}) == len(payments)
    for payment in payments:
        order = orders[payment["order_id"]]
        assert payment["status"] == order["status"]
        assert payment["amount_cents"] == order["paid_cents"] >= 0
        assert payment["attempted_at"] == order["attempted_at"]
        if order["status"] == "paid":
            assert payment["paid_at"] == order["paid_at"]
        else:
            # Unsuccessful attempts keep a NULL settlement time and zero amount.
            assert payment["paid_at"] is None
            assert payment["amount_cents"] == 0
    assert {row["status"] for row in orders.values()} <= {"paid", "failed", "pending", "cancelled"}
    assert {row["status"] for row in payments} <= {"paid", "failed", "pending", "cancelled"}
    # Both paid and unsuccessful attempts are represented.
    assert {row["status"] for row in payments} != {"paid"}


def test_inventory_outflows_match_purchased_lines_and_shipment_warehouse(base):
    _, _, folder = base
    orders = {row["order_id"]: row for row in _rows(folder, "orders")}
    items = _rows(folder, "order_items")
    shipments = {row["order_id"]: row for row in _rows(folder, "shipments")}
    movements = _rows(folder, "inventory_movements")

    by_order: dict[str, list[dict]] = {}
    for movement in movements:
        by_order.setdefault(movement["reference_id"], []).append(movement)

    # Only paid, shipped orders produce sale_out; the movements mirror the exact
    # purchased lines and the warehouse that actually shipped the order.
    assert set(by_order) == set(shipments)
    assert set(shipments) <= {oid for oid, order in orders.items() if order["status"] == "paid"}
    for order_id, rows in by_order.items():
        lines = [item for item in items if item["order_id"] == order_id]
        assert rows, order_id
        assert len(rows) == len(lines), order_id
        assert sorted((r["product_id"], r["quantity"]) for r in rows) == \
               sorted((item["product_id"], -item["quantity"]) for item in lines)
        for row in rows:
            assert row["movement_type"] == "sale_out"
            assert row["quantity"] < 0
            assert row["warehouse_id"] == shipments[order_id]["warehouse_id"]
            assert datetime.fromisoformat(row["moved_at"].replace("Z", "+00:00")).tzinfo is not None
            assert row["moved_at"] == shipments[order_id]["shipped_at"]
    # Non-paid orders move no inventory.
    for order_id, order in orders.items():
        if order["status"] != "paid":
            assert order_id not in by_order
    # The outflow really tracks purchased products rather than random ones.
    purchased = {item["product_id"] for item in items}
    assert {row["product_id"] for row in movements} <= purchased


def test_shipments_cover_exactly_the_paid_orders(base):
    _, _, folder = base
    orders = {row["order_id"]: row for row in _rows(folder, "orders")}
    shipments = _rows(folder, "shipments")
    paid = {oid for oid, order in orders.items() if order["status"] == "paid"}
    assert {row["order_id"] for row in shipments} == paid
    assert len(shipments) == len(paid)
    for row in shipments:
        assert row["warehouse_id"] in {w["warehouse_id"] for w in _rows(folder, "warehouses")}
        shipped = datetime.fromisoformat(row["shipped_at"].replace("Z", "+00:00"))
        paid_at = datetime.fromisoformat(orders[row["order_id"]]["paid_at"].replace("Z", "+00:00"))
        assert shipped >= paid_at, "an order cannot ship before it is paid"
        delivered = datetime.fromisoformat(row["delivered_at"].replace("Z", "+00:00"))
        assert delivered >= shipped
        assert row["freight_cents"] > 0


def test_orders_cover_twelve_business_months_and_four_channels(base):
    _, registry, folder = base
    orders = _rows(folder, "orders")
    assert {row["local_month"] for row in orders} == set(BUSINESS_MONTHS)
    with (folder / "channels.csv").open(encoding="utf-8", newline="") as handle:
        channels = list(csv.DictReader(handle))
    assert sorted(row["channel_code"] for row in channels) == ["app", "marketplace", "partner", "web"]
    assert {row["channel_code"] for row in orders} <= {row["channel_code"] for row in channels}
    # currency_code is the ISO 4217 currency; the amount unit is separate.
    assert all(row["currency_code"] == "CNY" for row in channels)
    assert all(row["currency_code"] != AMOUNT_UNIT for row in channels)
    # metric_timezone must be the full IANA zone, never a bare city name.
    assert all(row["metric_timezone"] == "Australia/Sydney" for row in channels)
    assert all(row["metric_timezone"] != "Sydney" for row in channels)
    assert registry["config"]["channels"] == ["web", "app", "marketplace", "partner"]
    assert registry["config"]["currency_code"] == "CNY"
    assert registry["config"]["amount_unit"] == AMOUNT_UNIT
    # Every order declares its currency and its amount unit in separate columns;
    # the unit is never stored in the column named ``currency``.
    assert {row["currency"] for row in orders} == {"CNY"}
    assert {row["amount_unit"] for row in orders} == {AMOUNT_UNIT}
    assert {row["currency"] for row in orders}.isdisjoint({AMOUNT_UNIT})


def test_utc_timestamps_are_aware_and_local_months_follow_event_time(base):
    _, _, folder = base
    order_stamps = _query(folder, "SELECT paid_at, local_month, order_date FROM orders WHERE status='paid'")
    assert order_stamps
    for paid_at, month, order_date in order_stamps:
        stamp = datetime.fromisoformat(paid_at.replace("Z", "+00:00"))
        assert stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0
        assert local_month(stamp) == month
        assert stamp.astimezone(ZONE).date().isoformat() == order_date
    # Non-paid orders have no paid_at at all; their derived month/date come from
    # the attempt stamp instead, and are re-derived here rather than trusted.
    unattempted = _query(folder, "SELECT attempted_at, local_month, order_date FROM orders WHERE paid_at IS NULL")
    assert unattempted
    for attempted_at, month, order_date in unattempted:
        stamp = datetime.fromisoformat(attempted_at.replace("Z", "+00:00"))
        assert stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0
        assert local_month(stamp) == month
        assert stamp.astimezone(ZONE).date().isoformat() == order_date
    for refunded_at, month in _query(folder, "SELECT refunded_at, local_month FROM refunds"):
        stamp = datetime.fromisoformat(refunded_at.replace("Z", "+00:00"))
        assert stamp.tzinfo is not None
        assert local_month(stamp) == month
    assert local_month(datetime.fromisoformat("2026-08-31T14:30:00+00:00")) == "2026-09"


# --------------------------------------------------------------------------
# refund rules
# --------------------------------------------------------------------------

def test_refunds_reference_paid_orders_and_respect_caps_and_order(base):
    _, _, folder = base
    orders = {row["order_id"]: row for row in _rows(folder, "orders")}
    refunds = _rows(folder, "refunds")
    assert refunds
    assert len({row["refund_id"] for row in refunds}) == len(refunds)
    succeeded: dict[str, int] = {}
    for refund in refunds:
        order = orders.get(refund["order_id"])
        assert order is not None, "refund must reference an existing order"
        assert order["status"] == "paid", "refund must reference a paid order"
        assert refund["status"] in {"succeeded", "failed", "pending", "cancelled"}
        assert refund["refund_cents"] >= 0
        paid_at = datetime.fromisoformat(order["paid_at"].replace("Z", "+00:00"))
        refunded_at = datetime.fromisoformat(refund["refunded_at"].replace("Z", "+00:00"))
        assert refunded_at >= paid_at
        assert refund["local_month"] in BUSINESS_MONTHS
        if refund["status"] == "succeeded":
            succeeded[refund["order_id"]] = succeeded.get(refund["order_id"], 0) + refund["refund_cents"]
    for order_id, total in succeeded.items():
        assert total <= orders[order_id]["paid_cents"], order_id


def test_refunds_show_statuses_repeats_and_cross_month_events(base):
    _, _, folder = base
    orders = {row["order_id"]: row for row in _rows(folder, "orders")}
    refunds = _rows(folder, "refunds")
    assert {row["status"] for row in refunds} == {"succeeded", "failed", "pending"}
    by_order: dict[str, list[dict]] = {}
    for refund in refunds:
        by_order.setdefault(refund["order_id"], []).append(refund)
    assert any(len(rows) > 1 for rows in by_order.values()), "expected repeated refunds"
    assert any(len({row["local_month"] for row in rows}) > 1 for rows in by_order.values()), "expected cross-month refunds"
    successful = {oid: rows for oid, rows in by_order.items() if any(r["status"] == "succeeded" for r in rows)}
    assert any(
        sum(row["refund_cents"] for row in rows if row["status"] == "succeeded") < orders[oid]["paid_cents"]
        for oid, rows in successful.items()
    ), "expected partial refunds"
    assert any(
        row["local_month"] != orders[row["order_id"]]["local_month"]
        for row in refunds if row["status"] == "succeeded"
    ), "expected refunds booked outside the payment month"
    assert any(row["reversal_of"] for row in refunds), "expected linked repeated refund events"
    refund_ids = {row["refund_id"] for row in refunds}
    assert all(row["reversal_of"] is None or row["reversal_of"] in refund_ids for row in refunds)


#: Per-cell sample floor for the distribution comparison, stated explicitly:
#: a share computed from fewer paid orders than this is not comparable.
DISTRIBUTION_MIN_CELL = 100


@pytest.fixture(scope="module")
def distribution(tmp_path_factory):
    """A larger, dedicated sample for the channel x month refund comparison.

    The base fixture only holds 400 orders spread over 12 months x 4 channels,
    so individual cells carry far too few paid orders for a share comparison to
    mean anything.  Instead of weakening the assertion, this test generates its
    own bigger sample and states the per-cell floor it actually needs.
    """
    root = tmp_path_factory.mktemp("dataset-v2-distribution")
    registry = generate_dataset(root, fixture="commerce-distribution", seed=73,
                                order_count=DISTRIBUTION_ORDERS,
                                ticket_count=SMALL["ticket_count"],
                                document_count=SMALL["document_count"])
    return root, registry, _folder(root, "commerce-distribution")


def test_channel_and_month_refund_distribution_varies(distribution):
    """2026-09 web has a deliberately raised refund-attempt share.

    This is a deterministic property of the fixture (a documented generation
    setting), and the refund/money invariants are checked elsewhere, so raising
    the sample size here does not relax any correctness check.
    """
    _, _, folder = distribution
    orders = {row["order_id"]: row for row in _rows(folder, "orders")}
    attempts: dict[tuple[str, str], int] = {}
    totals: dict[tuple[str, str], int] = {}
    for order in orders.values():
        if order["status"] != "paid":
            continue
        key = (order["local_month"], order["channel_code"])
        attempts[key] = attempts.get(key, 0) + 1
    for refund in _rows(folder, "refunds"):
        key = (orders[refund["order_id"]]["local_month"], orders[refund["order_id"]]["channel_code"])
        totals[key] = totals.get(key, 0) + 1
    shares = {key: totals.get(key, 0) / count
              for key, count in attempts.items() if count >= DISTRIBUTION_MIN_CELL}
    assert len(shares) >= 6, sorted(shares)
    september_web = shares.get(("2026-09", "web"))
    assert september_web is not None
    assert attempts[("2026-09", "web")] >= DISTRIBUTION_MIN_CELL
    # not a published constant: the setting only has to be visibly different.
    assert september_web == max(shares.values())
    assert september_web > min(shares.values())


def test_support_ticket_links_are_genuinely_linked(base):
    _, _, folder = base
    tickets = {row["ticket_id"]: row for row in _tickets(folder)}
    orders = {row["order_id"] for row in _rows(folder, "orders")}
    links = _rows(folder, "support_ticket_links")
    assert links
    assert len({row["link_id"] for row in links}) == len(links)
    assert len({row["ticket_id"] for row in links}) == len(links)
    for link in links:
        assert link["ticket_id"] in tickets
        assert link["order_id"] in orders
        assert tickets[link["ticket_id"]]["order_id"] == link["order_id"]
        assert 0 < link["confidence"] <= 1
        assert link["local_month"] in BUSINESS_MONTHS
    linked_tickets = {row["ticket_id"] for row in links}
    assert linked_tickets == {tid for tid, row in tickets.items() if row["order_id"]}
    for ticket_id, ticket in tickets.items():
        if ticket_id in linked_tickets:
            assert ticket["order_id"] in orders
        else:
            assert ticket["order_id"] is None


def test_tickets_have_varied_contexts_and_partial_association(base):
    _, _, folder = base
    tickets = _tickets(folder)
    assert len(tickets) == SMALL["ticket_count"]
    linked = sum(1 for row in tickets if row["order_id"])
    assert 0.4 <= linked / len(tickets) <= 0.6
    assert len({row["text"] for row in tickets}) > len(tickets) // 4
    assert len({row["context"] for row in tickets}) >= 5
    assert {row["channel_code"] for row in tickets} <= {"web", "app", "marketplace", "partner"}
    for row in tickets:
        assert set(row) == {"ticket_id", "text", "created_at", "order_id", "channel_code", "context"}
        assert datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")).tzinfo is not None


# --------------------------------------------------------------------------
# documents and registry contract
# --------------------------------------------------------------------------

def test_documents_are_diverse_sourced_and_status_labelled(base):
    _, registry, folder = base
    documents = registry["documents"]
    assert len(documents) == SMALL["document_count"]
    bodies = {}
    for document in documents:
        path = folder / document["path"]
        assert path.is_file()
        text = path.read_text(encoding="utf-8")
        bodies[document["document_id"]] = text
        assert any(f"状态：{status}" in text for status in ("APPROVED", "REFERENCE", "STALE", "DRAFT"))
        assert text.startswith("# ")
        assert document["lines"] == len(text.splitlines())
    assert len(set(bodies.values())) == len(bodies), "documents must not differ only by title/index"
    assert {d["status"] for d in documents} == {"APPROVED", "REFERENCE", "STALE", "DRAFT"}
    approved = [d for d in documents if d["status"] == "APPROVED"]
    assert approved == [d for d in documents if d.get("metric_id")]
    assert len(approved) == 1 and approved[0]["metric_id"] == "CASH_NET_v1"
    assert "CASH_NET_v1" in bodies[approved[0]["document_id"]]
    assert any(len(text.splitlines()) > 120 for text in bodies.values())
    # A rule that matters sits past the current 80-line retrieval cut.
    long_docs = [text for text in bodies.values() if len(text.splitlines()) > 120]
    assert any("关键" in line for text in long_docs for line in text.splitlines()[80:])


def test_documents_contain_no_answer_sql_or_gold_totals(base):
    _, registry, folder = base
    for document in registry["documents"]:
        text = (folder / document["path"]).read_text(encoding="utf-8").lower()
        for banned in ("select ", "group by", "gold", "标准答案", "评测结果"):
            assert banned not in text, (document["document_id"], banned)


def test_registry_is_relative_compatible_and_complete(base):
    _, registry, folder = base
    assert registry["fixture"] == "commerce-base"
    assert registry["seed"] == 73
    assert registry["generator"]
    assert [s["source_id"] for s in registry["sources"]] == ["business", "channels", "tickets"]
    for source in registry["sources"]:
        path = Path(source["path"])
        assert not path.is_absolute()
        assert (folder / path).is_file()
        assert ".." not in path.parts
    for document in registry["documents"]:
        assert not Path(document["path"]).is_absolute()
        assert (folder / document["path"]).is_file()
    contracts = [c for c in registry["metric_contracts"] if c["status"] == "APPROVED"]
    assert len(contracts) == 1
    contract = contracts[0]
    assert contract["metric_id"] == "CASH_NET_v1" and contract["version"] == 1
    assert contract["unit"] == "CNY_cents" and contract["timezone"] == "Australia/Sydney"
    assert contract["source_ids"] == ["business", "channels"]
    assert contract["formula"] == "successful paid events minus successful refund events in each event's own local calendar month"
    assert contract["payment_status"] == "paid" and contract["refund_status"] == "succeeded"
    assert contract["event_time"] == {"orders": "paid_at", "refunds": "refunded_at"}
    assert contract["missing_channel_policy"] == "reject"
    serialized = json.dumps(registry, ensure_ascii=False).lower()
    for banned in ("gold", "expected_cents", "answer"):
        assert banned not in serialized


def test_registry_sources_match_existing_snapshot_loader_contract(base):
    """Mirror the shapes snapshot_sources/discover_schema/validate_sources expect."""
    _, registry, folder = base
    business = next(s for s in registry["sources"] if s["source_id"] == "business")
    with closing(sqlite3.connect(folder / business["path"])) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        schema = {table: [{"name": r[1], "type": r[2]} for r in db.execute(f'PRAGMA table_info("{table}")')]
                  for table in business["tables"]}
    assert tables == set(TABLE_ORDER)
    assert set(schema) == set(TABLE_ORDER)
    orders = schema["orders"] and {field["name"] for field in schema["orders"]}
    refunds = {field["name"] for field in schema["refunds"]}
    assert {"order_id", "paid_at", "channel_code", "paid_cents", "status"} <= orders
    assert {"refund_id", "order_id", "refunded_at", "refund_cents", "status"} <= refunds
    # New required fields for the money semantics are part of the published schema.
    assert {"attempted_at", "order_total_cents", "local_month", "customer_id"} <= orders
    # The currency and the amount unit are separate published columns.
    assert {"currency", "amount_unit"} <= orders
    assert {"attempted_at"} <= {field["name"] for field in schema["payments"]}
    # validation.py reads these types through _sql_type; money must map to BIGINT.
    assert schema["orders"][[f["name"] for f in schema["orders"]].index("paid_cents")]["type"] == "INTEGER"
    assert schema["orders"][[f["name"] for f in schema["orders"]].index("order_total_cents")]["type"] == "INTEGER"
    channels = folder / next(s for s in registry["sources"] if s["source_id"] == "channels")["path"]
    with channels.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        channel_rows = list(reader)
    assert "channel_code" in header and "channel_name" in header
    # The IANA zone is published per channel, and the currency/unit split is
    # recorded as data rather than implied by the column name.
    assert "metric_timezone" in header and "currency_code" in header
    assert {row["metric_timezone"] for row in channel_rows} == {"Australia/Sydney"}
    assert {row["currency_code"] for row in channel_rows} == {"CNY"}
    channels_source = next(s for s in registry["sources"] if s["source_id"] == "channels")
    assert channels_source["amount_unit"] == AMOUNT_UNIT
    assert channels_source["currency_code"] == "CNY"
    tickets = folder / next(s for s in registry["sources"] if s["source_id"] == "tickets")["path"]
    first = json.loads(tickets.read_text(encoding="utf-8").splitlines()[0])
    assert {"ticket_id", "text", "created_at", "order_id", "channel_code"} <= set(first)


def test_registry_declares_amount_unit_apart_from_currency(base):
    """CNY (ISO 4217 currency) and CNY_cents (amount unit) must stay distinct."""
    _, registry, folder = base
    config = registry["config"]
    assert config["currency_code"] == "CNY"
    assert config["amount_unit"] == AMOUNT_UNIT
    assert config["currency_code"] != config["amount_unit"]
    manifest = registry["manifest"]
    assert manifest["amount_unit"] == AMOUNT_UNIT
    assert manifest["currency_code"] == "CNY"
    # The derived-column contract documents the paid_at vs attempted_at split.
    derived = manifest["derived_columns"]
    assert "attempted_at" in derived["orders.local_month"]
    assert "paid_at" in derived["orders.local_month"]
    assert "paid_at" in derived["orders.order_date"]
    assert "paid_at" in manifest["paid_at_semantics"]
    # Every order row stores the currency and the unit in separate, correctly
    # named columns: ``currency`` holds CNY, ``amount_unit`` holds CNY_cents.
    order_rows = _rows(folder, "orders")
    assert {row["currency"] for row in order_rows} == {config["currency_code"]}
    assert {row["amount_unit"] for row in order_rows} == {config["amount_unit"]}
    # A unit is never published as a currency, in either direction.
    assert config["amount_unit"] not in {row["currency"] for row in order_rows}
    # Money columns are integer cents throughout, never floats.
    for table in ("orders", "order_items", "payments", "refunds", "order_promotions",
                  "inventory_movements", "shipments"):
        for row in _rows(folder, table):
            for key, value in row.items():
                if key.endswith("_cents"):
                    assert value is None or (isinstance(value, int) and not isinstance(value, bool)), (table, key)


# --------------------------------------------------------------------------
# determinism, refusal and validation
# --------------------------------------------------------------------------

def test_same_parameters_produce_identical_content(tmp_path):
    """The fixture *name* is a label, not an input to the random stream.

    Two different fixture names generated with the same size parameters must
    produce byte-identical logical business data under different roots; only the
    registry's own ``fixture`` field may differ.
    """
    first = generate_dataset(tmp_path, fixture="repeat-a", seed=11, **SMALL)
    second = generate_dataset(tmp_path, fixture="repeat-b", seed=11, **SMALL)
    a, b = _folder(tmp_path, "repeat-a"), _folder(tmp_path, "repeat-b")
    assert a != b
    for table in TABLE_ORDER:
        assert _rows(a, table) == _rows(b, table), table
    assert (a / "channels.csv").read_bytes() == (b / "channels.csv").read_bytes()
    assert (a / "tickets.jsonl").read_bytes() == (b / "tickets.jsonl").read_bytes()
    assert (a / "business.sqlite").read_bytes() == (b / "business.sqlite").read_bytes()
    assert [d["lines"] for d in first["documents"]] == [d["lines"] for d in second["documents"]]
    for document in first["documents"]:
        name = document["document_id"] + ".md"
        assert (a / "documents" / name).read_bytes() == (b / "documents" / name).read_bytes()
    # registry differs only by fixture name, never by content or wall clock.
    left = {k: v for k, v in first.items() if k != "fixture"}
    right = {k: v for k, v in second.items() if k != "fixture"}
    assert left == right
    assert first["fixture"] == "repeat-a" and second["fixture"] == "repeat-b"
    # Regenerating under the same name/root is refused rather than duplicated.
    with pytest.raises(DatasetError):
        generate_dataset(tmp_path, fixture="repeat-a", seed=11, **SMALL)


def test_same_parameters_under_a_different_project_root_match(tmp_path):
    """Portability: the data depends on parameters, not on the target root."""
    left_root = tmp_path / "root-left"
    right_root = tmp_path / "root-right"
    generate_dataset(left_root, fixture="portable", seed=29, **SMALL)
    generate_dataset(right_root, fixture="portable-elsewhere", seed=29, **SMALL)
    left = _folder(left_root, "portable")
    right = _folder(right_root, "portable-elsewhere")
    assert left != right
    assert _rows(left, "orders") == _rows(right, "orders")
    assert _rows(left, "order_promotions") == _rows(right, "order_promotions")
    assert _rows(left, "inventory_movements") == _rows(right, "inventory_movements")


def test_different_seed_changes_business_content(tmp_path):
    generate_dataset(tmp_path, fixture="seed-a", seed=1, **SMALL)
    generate_dataset(tmp_path, fixture="seed-b", seed=2, **SMALL)
    assert _rows(_folder(tmp_path, "seed-a"), "orders") != _rows(_folder(tmp_path, "seed-b"), "orders")


def test_seed_is_part_of_the_random_stream_but_fixture_name_is_not(tmp_path):
    """Guards both halves of the reproducibility contract at once."""
    generate_dataset(tmp_path, fixture="stream-a", seed=13, **SMALL)
    generate_dataset(tmp_path, fixture="stream-b", seed=13, **SMALL)
    generate_dataset(tmp_path, fixture="stream-c", seed=14, **SMALL)
    a, b, c = (_folder(tmp_path, name) for name in ("stream-a", "stream-b", "stream-c"))
    assert _rows(a, "orders") == _rows(b, "orders")
    assert _rows(b, "orders") != _rows(c, "orders")


def test_no_wall_clock_timestamps_are_embedded(tmp_path):
    registry = generate_dataset(tmp_path, fixture="no-clock", seed=5, **SMALL)
    folder = _folder(tmp_path, "no-clock")
    # every business timestamp is inside the frozen calendar window.
    for row in _rows(folder, "orders"):
        for column in ("created_at", "attempted_at"):
            assert row[column][:4] in {"2025", "2026"}, (row["order_id"], column)
        if row["paid_at"]:
            assert row["paid_at"][:4] in {"2025", "2026"}
    for row in _rows(folder, "payments"):
        assert row["attempted_at"][:4] in {"2025", "2026"}
    for row in _rows(folder, "refunds"):
        assert row["refunded_at"][:4] in {"2025", "2026"}
        if row["settled_at"]:
            assert row["settled_at"][:4] in {"2025", "2026"}
    text = json.dumps(registry, ensure_ascii=False)
    assert str(datetime.now().year + 1) not in text


def test_existing_fixture_is_refused_and_previous_fixture_untouched(tmp_path):
    generate_dataset(tmp_path, fixture="keep-me", seed=3, **SMALL)
    before = (_folder(tmp_path, "keep-me") / "business.sqlite").read_bytes()
    with pytest.raises(DatasetError):
        generate_dataset(tmp_path, fixture="keep-me", seed=3, **SMALL)
    assert (_folder(tmp_path, "keep-me") / "business.sqlite").read_bytes() == before
    assert not list((tmp_path / "var" / "sources").glob(".*tmp*"))


@pytest.mark.parametrize("fixture", ["../escape", "with space", "bad/slash", "", "a.b", "ünicode"])
def test_unsafe_fixture_identifiers_are_refused(tmp_path, fixture):
    with pytest.raises(DatasetError):
        generate_dataset(tmp_path, fixture=fixture, seed=1, **SMALL)
    assert not _folder(tmp_path, str(fixture)).exists()
    sources = tmp_path / "var" / "sources"
    assert not sources.exists() or not list(sources.iterdir())


@pytest.mark.parametrize("field,value", [
    ("order_count", 99), ("order_count", 100001), ("ticket_count", -1),
    ("ticket_count", 10001), ("document_count", 39), ("document_count", 301),
    ("order_count", 100.5), ("ticket_count", 1.5), ("document_count", "60"),
    ("order_count", True), ("ticket_count", True), ("document_count", True),
])
def test_out_of_range_parameters_are_refused(tmp_path, field, value):
    kwargs = {"seed": 1, **SMALL, field: value}
    with pytest.raises(DatasetError):
        generate_dataset(tmp_path, fixture="range-check", **kwargs)
    assert not _folder(tmp_path, "range-check").exists()


@pytest.mark.parametrize("seed", [-1, -2147483649, 2 ** 31, 1.0, "1", None, True, False])
def test_seed_must_be_a_nonnegative_integer(tmp_path, seed):
    """Seed validation must actually run: pass it once, as the only seed.

    ``bool`` is rejected explicitly even though ``isinstance(True, int)`` holds,
    so ``True`` can never silently mean seed 1.
    """
    with pytest.raises(DatasetError):
        generate_dataset(tmp_path, fixture="seed-check", seed=seed, **SMALL)
    assert not _folder(tmp_path, "seed-check").exists()
    sources = tmp_path / "var" / "sources"
    assert not sources.exists() or not list(sources.iterdir())


@pytest.mark.parametrize("seed", [0, 73, 2 ** 31 - 1])
def test_seed_boundary_values_are_accepted(tmp_path, seed):
    registry = generate_dataset(tmp_path, fixture=f"seed-{seed}", seed=seed, **SMALL)
    assert registry["seed"] == seed
    assert _query(_folder(tmp_path, f"seed-{seed}"), "SELECT COUNT(*) FROM orders")[0][0] == SMALL["order_count"]


def test_boundary_limits_are_accepted(tmp_path):
    registry = generate_dataset(tmp_path, fixture="limits", seed=7,
                                order_count=100, ticket_count=0, document_count=40)
    assert registry["manifest"]["row_counts"]["orders"] == 100
    assert registry["manifest"]["row_counts"]["tickets"] == 0
    assert len(registry["documents"]) == 40
    folder = _folder(tmp_path, "limits")
    assert (folder / "tickets.jsonl").read_text(encoding="utf-8") == ""
    assert _query(folder, "SELECT COUNT(*) FROM support_ticket_links")[0][0] == 0
    assert _query(folder, "SELECT COUNT(*) FROM orders")[0][0] == 100
    # All 24 tables still exist; only support_ticket_links is naturally empty,
    # because there is no real ticket to link.  Nothing is invented to fill it.
    names = {row[0] for row in _query(folder, "SELECT name FROM sqlite_master WHERE type='table'")}
    assert names == set(TABLE_ORDER)
    populated = [t for t in TABLE_ORDER
                 if _query(folder, f'SELECT COUNT(*) FROM "{t}"')[0][0] > 0]
    assert set(populated) == set(TABLE_ORDER) - {"support_ticket_links"}
    # The money and promotion semantics still hold at the minimum size.
    assert _query(folder, "SELECT COUNT(*) FROM order_promotions")[0][0] > 0
    assert _query(folder, "SELECT COUNT(*) FROM inventory_movements")[0][0] > 0
    # Exactly the paid orders are shipped, and their outflows mirror the lines.
    paid = _query(folder, "SELECT COUNT(*) FROM orders WHERE status='paid'")[0][0]
    assert paid > 0
    assert _query(folder, "SELECT COUNT(*) FROM shipments")[0][0] == paid
    assert _query(folder, "SELECT COUNT(*) FROM orders WHERE paid_at IS NULL")[0][0] > 0
    assert _query(folder, "SELECT COUNT(*) FROM orders WHERE status!='paid' AND paid_cents=0")[0][0] > 0
    assert _query(folder, "SELECT COUNT(*) FROM orders WHERE status!='paid' AND paid_at IS NOT NULL")[0][0] == 0
    assert any(d.get("metric_id") == "CASH_NET_v1" for d in registry["documents"])


def test_failure_cleans_only_its_own_temporary_folder(tmp_path, monkeypatch):
    import local_data_agent.dataset_v2 as module
    sources = tmp_path / "var" / "sources"
    sources.mkdir(parents=True)
    keep = sources / "existing"
    keep.mkdir()
    (keep / "marker.txt").write_text("keep", encoding="utf-8")
    # A stray temporary-looking folder from an unrelated run must survive.
    stray = sources / ".unrelated.tmp-deadbeef"
    stray.mkdir()
    (stray / "marker.txt").write_text("stray", encoding="utf-8")

    def boom(*args, **kwargs):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(module, "_write_sqlite", boom)
    with pytest.raises(RuntimeError):
        generate_dataset(tmp_path, fixture="broken", seed=1, **SMALL)
    assert not _folder(tmp_path, "broken").exists()
    assert sorted(p.name for p in sources.iterdir()) == sorted(["existing", ".unrelated.tmp-deadbeef"])
    assert (keep / "marker.txt").read_text(encoding="utf-8") == "keep"
    assert (stray / "marker.txt").read_text(encoding="utf-8") == "stray"


def test_temporary_folder_is_private_and_under_the_resolved_source_root(tmp_path, monkeypatch):
    """The cleanup target must be provably this run's folder under source_root."""
    import local_data_agent.dataset_v2 as module
    seen: list[Path] = []
    original = module._generate_into

    def capture(folder, *args, **kwargs):
        seen.append(folder)
        return original(folder, *args, **kwargs)
    monkeypatch.setattr(module, "_generate_into", capture)
    generate_dataset(tmp_path, fixture="private-temp", seed=4, **SMALL)
    assert len(seen) == 1
    temporary = seen[0]
    resolved_root = (tmp_path / "var" / "sources").resolve()
    assert temporary.resolve().parent == resolved_root
    assert temporary.name.startswith(".private-temp.tmp-")
    # Published atomically: the target exists and no temporary folder is left.
    assert _folder(tmp_path, "private-temp").is_dir()
    assert not list(resolved_root.glob(".*tmp*"))


def test_discard_temporary_refuses_paths_outside_the_source_root(tmp_path):
    """A computed cleanup target is verified before any recursive delete."""
    import local_data_agent.dataset_v2 as module
    source_root = tmp_path / "var" / "sources"
    source_root.mkdir(parents=True)
    outside = tmp_path / "victim"
    outside.mkdir()
    (outside / "marker.txt").write_text("keep", encoding="utf-8")
    nested = source_root / "nested"
    nested.mkdir()
    (nested / "marker.txt").write_text("keep", encoding="utf-8")

    # A sibling of source_root is not under it.
    with pytest.raises(DatasetError):
        module._discard_temporary(outside, source_root, "broken")
    assert (outside / "marker.txt").read_text(encoding="utf-8") == "keep"
    # A nested directory is under source_root but not a direct child.
    with pytest.raises(DatasetError):
        module._discard_temporary(nested, source_root, "broken")
    assert (nested / "marker.txt").read_text(encoding="utf-8") == "keep"
    # A direct child with the wrong name is refused.
    wrong = source_root / "not-a-temp"
    wrong.mkdir()
    (wrong / "marker.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(DatasetError):
        module._discard_temporary(wrong, source_root, "broken")
    assert (wrong / "marker.txt").read_text(encoding="utf-8") == "keep"
    # The source root itself is never removed.
    with pytest.raises(DatasetError):
        module._discard_temporary(source_root, source_root, "broken")
    assert source_root.is_dir()


def test_discard_temporary_removes_a_genuine_temporary_folder(tmp_path):
    import local_data_agent.dataset_v2 as module
    source_root = tmp_path / "var" / "sources"
    source_root.mkdir(parents=True)
    genuine = source_root / ".thing.tmp-0123456789ab"
    genuine.mkdir()
    (genuine / "partial.txt").write_text("partial", encoding="utf-8")
    module._discard_temporary(genuine, source_root, "thing")
    assert not genuine.exists()
    assert source_root.is_dir()


def test_support_ticket_links_empty_only_when_nothing_can_be_linked(tmp_path):
    """Zero tickets is supported; any linkable ticket count must populate it."""
    import local_data_agent.dataset_v2 as module
    # Zero tickets: the table is naturally empty and nothing is invented.
    generate_dataset(tmp_path, fixture="zero-tickets", seed=21,
                     order_count=150, ticket_count=0, document_count=40)
    folder = _folder(tmp_path, "zero-tickets")
    assert _query(folder, "SELECT COUNT(*) FROM support_ticket_links")[0][0] == 0
    assert (folder / "tickets.jsonl").read_text(encoding="utf-8") == ""
    # ticket_count = 1 cannot yield a linked ticket either (only odd ids link,
    # and the first odd id is 1 with no linkable partner count), so it is also
    # allowed to be empty rather than failing generation.
    generate_dataset(tmp_path, fixture="one-ticket", seed=21,
                     order_count=150, ticket_count=1, document_count=40)
    assert _query(_folder(tmp_path, "one-ticket"), "SELECT COUNT(*) FROM support_ticket_links")[0][0] == 0
    # Two tickets is the first count that must produce a genuine link.
    generate_dataset(tmp_path, fixture="two-tickets", seed=21,
                     order_count=150, ticket_count=2, document_count=40)
    assert _query(_folder(tmp_path, "two-tickets"), "SELECT COUNT(*) FROM support_ticket_links")[0][0] == 1
    assert module._may_be_empty("support_ticket_links", 0)
    assert module._may_be_empty("support_ticket_links", 1)
    for count in (2, 3, 120, 3000):
        assert not module._may_be_empty("support_ticket_links", count), count
    # No other table is ever allowed to be empty, at any ticket count.
    for table in TABLE_ORDER:
        if table != "support_ticket_links":
            for count in (0, 1, 3000):
                assert not module._may_be_empty(table, count), (table, count)


def test_positive_ticket_counts_always_produce_links(base):
    """The default configuration keeps every table populated."""
    _, registry, folder = base
    assert SMALL["ticket_count"] > 0
    assert registry["manifest"]["row_counts"]["support_ticket_links"] > 0
    assert _query(folder, "SELECT COUNT(*) FROM support_ticket_links")[0][0] > 0


def test_discount_allocation_is_exact_integer_cents():
    """Allocation must sum back exactly, in integer cents, for every split."""
    from local_data_agent.dataset_v2 import _allocate_discount
    cases = [
        ([1000, 2000, 3000], 0), ([1000, 2000, 3000], 1), ([1000, 2000, 3000], 7),
        ([1000, 2000, 3000], 599), ([1000, 2000, 3000], 6000),
        ([999], 999), ([1, 1, 1], 1), ([1, 1, 1], 2), ([5, 5], 3),
        ([12345, 1], 100), ([7, 11, 13, 17], 41),
    ]
    for gross, discount in cases:
        parts = _allocate_discount(discount, gross)
        assert len(parts) == len(gross)
        assert all(isinstance(part, int) for part in parts)
        assert sum(parts) == discount, (gross, discount, parts)
        assert all(0 <= part <= line for part, line in zip(parts, gross)), (gross, discount, parts)
    # Deterministic: the same inputs always give the same split.
    assert _allocate_discount(7, [1000, 2000, 3000]) == _allocate_discount(7, [1000, 2000, 3000])
    # A discount larger than the gross is refused rather than creating a negative line.
    with pytest.raises(DatasetError):
        _allocate_discount(6001, [1000, 2000, 3000])
    with pytest.raises(DatasetError):
        _allocate_discount(-1, [1000, 2000, 3000])


def test_promotion_windows_match_the_published_promotions_table(base):
    """The discount windows used at generation time are the published ones."""
    _, _, folder = base
    promotions = {row["promotion_id"]: row for row in _rows(folder, "promotions")}
    assert set(promotions) == set(PROMOTION_WINDOWS)
    for promotion_id, (start, end) in PROMOTION_WINDOWS.items():
        row = promotions[promotion_id]
        # starts_on/ends_on are inclusive calendar-day bounds inside the window.
        assert row["starts_on"][:7] <= start
        assert row["ends_on"][:7] >= end
        assert row["discount_bps"] > 0
    applied = _rows(folder, "order_promotions")
    assert applied
    for row in applied:
        assert row["promotion_id"] in promotions


def test_promotion_window_covers_interior_months(base):
    """An inclusive (start, end) window must cover every month *between* them.

    ``p03`` spans 2025-10..2026-03, so 2025-12 is an interior month: the window
    test must accept it.  A ``month not in (start, end)`` check would reject it
    and silently reduce multi-month promotions to their two endpoints.
    """
    from local_data_agent.dataset_v2 import _promotion_discount, _promotion_months
    start, end = _promotion_months("p03")
    assert (start, end) == ("2025-10", "2026-03")
    interior = "2025-12"
    assert start < interior < end
    # Many independent deterministic draws: the interior month must be able to
    # produce a real, positive discount at the declared bps rate.
    applied = [_promotion_discount(random.Random(seed), ("p03", 1200), interior, "marketplace", 100_000)
               for seed in range(400)]
    assert any(promotion_id == "p03" for promotion_id, _ in applied)
    for promotion_id, discount in applied:
        if promotion_id:
            assert promotion_id == "p03"
            assert discount == 100_000 * 1200 // 10_000
    # The month before the window and the month after it are still refused.
    for outside in ("2025-09", "2026-04"):
        assert _promotion_discount(random.Random(1), ("p03", 1200), outside, "marketplace", 100_000) == ("", 0)


def test_september_web_refunds_exist_and_are_not_before_payment(distribution):
    """2026-09 web must actually produce refunds, all at/after the real payment.

    Same-month refunds are eligible whenever the payment month still has room
    after the payment instant, so September must not collapse to zero share.
    """
    _, _, folder = distribution
    orders = {row["order_id"]: row for row in _rows(folder, "orders")}
    paid_web_september = [oid for oid, row in orders.items()
                          if row["status"] == "paid" and row["local_month"] == "2026-09"
                          and row["channel_code"] == "web"]
    assert len(paid_web_september) >= DISTRIBUTION_MIN_CELL
    refunded = {row["order_id"] for row in _rows(folder, "refunds")}
    september_refunds = [oid for oid in paid_web_september if oid in refunded]
    assert september_refunds, "2026-09 web payments must allow same-month refunds"
    for row in _rows(folder, "refunds"):
        order = orders[row["order_id"]]
        if order["local_month"] != "2026-09" or order["channel_code"] != "web":
            continue
        paid_at = datetime.fromisoformat(order["paid_at"].replace("Z", "+00:00"))
        refunded_at = datetime.fromisoformat(row["refunded_at"].replace("Z", "+00:00"))
        assert refunded_at > paid_at, "a refund may never precede its payment"
        assert row["local_month"] in BUSINESS_MONTHS


def test_orders_currency_column_holds_a_currency_not_a_unit(base):
    """``orders.currency`` must be an ISO 4217 code; the unit has its own column.

    Storing ``CNY_cents`` (a *unit*) in a column named ``currency`` mislabels it.
    The unit is published in ``orders.amount_unit`` instead, and the two columns
    must never carry the same value.
    """
    _, _, folder = base
    for row in _rows(folder, "orders"):
        assert row["currency"] == "CNY"
        assert row["amount_unit"] == AMOUNT_UNIT
        assert row["currency"] != row["amount_unit"]
        # A currency code is three uppercase letters; a unit name is not.
        assert row["currency"].isalpha() and row["currency"].isupper()
        assert row["currency"].isascii()
        assert "_" in row["amount_unit"]


def test_self_check_uses_prebuilt_order_item_index(base):
    """The promotion/shipment checks must read the indexed pass, not rescan items.

    Each item row may be visited only a small constant number of times.  The old
    code rescanned the whole ``order_items`` table once per recorded promotion
    and once per shipment, which is O(orders x items) at 50k; this budget fails
    loudly if that pattern returns.
    """
    from local_data_agent.dataset_v2 import _build_business, _self_check
    business, _ = _build_business(random.Random("self-check-index"), 73, 400)

    class Counting(list):
        """Counts how many item rows the self check actually walks."""

        def __init__(self, rows):
            super().__init__(rows)
            self.yielded = 0

        def __iter__(self):
            for row in list.__iter__(self):
                self.yielded += 1
                yield row

    counted = Counting(business["order_items"])
    business["order_items"] = counted
    # ``support_ticket_links`` is normally attached by the caller; with no
    # tickets it is legitimately empty, which is the one allowed empty table.
    business["support_ticket_links"] = []
    _self_check(business, [], 0)
    # A single linear sweep visits each item row once (twice at most: the
    # null-primary-key guard plus the indexed pass).  The old O(orders*items)
    # code walked the whole item table once per recorded promotion and once per
    # shipment, i.e. many multiples of the row count on this fixture.
    assert counted.yielded <= 3 * len(counted), (counted.yielded, len(counted))
    assert business["order_promotions"], "fixture must exercise the promotion path"


def test_order_promotion_discount_never_exceeds_the_order(base):
    """A genuinely applied discount is always a real, bounded reduction."""
    _, _, folder = base
    orders = {row["order_id"]: row for row in _rows(folder, "orders")}
    for row in _rows(folder, "order_promotions"):
        order = orders[row["order_id"]]
        assert 0 < row["discount_cents"] < order["order_total_cents"] + row["discount_cents"]
        assert order["order_total_cents"] > 0
        # The stored net total is the gross total minus the recorded discount.
        gross = sum(item["unit_price_cents"] * item["quantity"]
                    for item in _rows(folder, "order_items") if item["order_id"] == row["order_id"])
        assert order["order_total_cents"] == gross - row["discount_cents"]


def test_default_parameters_match_the_documented_contract(tmp_path):
    import inspect
    signature = inspect.signature(generate_dataset)
    assert signature.parameters["fixture"].default == "commerce-v2"
    assert signature.parameters["seed"].default == 73
    assert signature.parameters["order_count"].default == 50000
    assert signature.parameters["ticket_count"].default == 3000
    assert signature.parameters["document_count"].default == 120


def test_cli_parser_exposes_documented_flags():
    from local_data_agent.dataset_v2 import build_parser, main
    args = build_parser().parse_args([
        "--project-root", "X", "--fixture", "cli-check", "--seed", "9",
        "--orders", "150", "--tickets", "20", "--documents", "45"])
    assert (args.fixture, args.seed, args.orders, args.tickets, args.documents) == ("cli-check", 9, 150, 20, 45)
    assert callable(main)
