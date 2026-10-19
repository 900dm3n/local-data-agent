"""Reproducible commerce source generator for the v2 data environment.

Design notes
------------
* Standard library only, no imports from the rest of this package, so the
  generator can run before any agent wiring exists.
* Generation is a pure function of ``(seed, order_count, ticket_count,
  document_count)``.  The fixture identifier is a publication label and does
  *not* enter the random stream, so the same size parameters published under two
  different fixture names produce identical logical data.  Nothing reads the
  wall clock: every timestamp is derived from the fixed business calendar
  2025-10 .. 2026-09.
* Money semantics: ``paid_at`` is written only for orders that actually
  succeeded and stays NULL otherwise; ``attempted_at`` records when the order
  was attempted.  Promotion discounts are genuinely allocated into order lines,
  and inventory outflows mirror the purchased lines.
* An order's ``attempted_at`` is the order's own event stamp, and for paid
  orders ``paid_at`` repeats that same stamp, so the event a row's
  ``local_month`` is derived from is always inside the covered calendar.  A
  non-paid row keeps ``paid_at = NULL`` while still carrying a real, aware
  attempt time.
* Output is written to a sibling temporary folder and published with a single
  atomic ``os.replace`` only after every artifact is complete.  A failed run
  cleans the temporary folder it created itself and nothing else: the folder is
  re-verified to resolve directly under the resolved ``source_root`` first.
* Two order fields are derived from event UTC timestamps under the approved
  timezone: ``order_date`` (business date) and ``local_month``.  For paid orders
  both come from ``paid_at``; for orders that never succeeded they come from
  ``attempted_at``.  The raw timestamp stays authoritative; the derived columns
  exist only so that source-side aggregation can stay under the
  ten-thousand-row artifact cap.

    python -m local_data_agent.dataset_v2 --project-root ROOT --fixture ID \
        --seed 73 --orders 50000 --tickets 3000 --documents 120
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import shutil
import sqlite3
import sys
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ZONE = "Australia/Sydney"
GENERATOR_VERSION = "dataset-v2/1.0.0"
DEFAULT_FIXTURE = "commerce-v2"
BUSINESS_MONTHS = (
    "2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03",
    "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09",
)
#: (channel_code, channel_name, currency_code, metric_timezone).
#: ``currency_code`` is the ISO 4217 currency (CNY); it is deliberately *not*
#: the amount unit used inside SQLite.  Every ``*_cents`` column stores integer
#: minor units of that currency (CNY_cents), which is a different concept and is
#: declared separately in the registry/metric contract.
CHANNELS = (
    ("web", "官网", "CNY", "Australia/Sydney"),
    ("app", "移动端", "CNY", "Australia/Sydney"),
    ("marketplace", "第三方平台", "CNY", "Australia/Sydney"),
    ("partner", "合作伙伴", "CNY", "Australia/Sydney"),
)
#: amount unit for every integer ``*_cents`` column in this fixture.
AMOUNT_UNIT = "CNY_cents"
#: ISO 4217 currency code of every money column.  This is a *currency*, not a
#: unit: ``CNY`` says which money, ``AMOUNT_UNIT`` says how it is counted.
CURRENCY_CODE = "CNY"
CHANNEL_CODES = tuple(code for code, _, _, _ in CHANNELS)
ORDER_STATUSES = ("paid", "failed", "pending", "cancelled")
REFUND_STATUSES = ("succeeded", "failed", "pending", "cancelled")
LIMITS = {"orders": (100, 100000), "tickets": (0, 10000), "documents": (40, 300)}
STATUS_VALUES = ("APPROVED", "REFERENCE", "STALE", "DRAFT")
SAFE_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")

#: exactly 24 SQLite tables, declared in a stable order.
TABLE_SCHEMAS: dict[str, list[tuple[str, str]]] = {
    "orders": [
        # ``paid_at`` stays NULL for any order that did not succeed.  It is the
        # only field that proves a successful payment and must never be faked.
        ("order_id", "TEXT PRIMARY KEY"), ("customer_id", "TEXT NOT NULL"),
        ("paid_at", "TEXT"), ("attempted_at", "TEXT NOT NULL"),
        ("order_date", "TEXT"), ("local_month", "TEXT"),
        ("channel_code", "TEXT NOT NULL"), ("status", "TEXT NOT NULL"),
        ("paid_cents", "INTEGER NOT NULL"), ("order_total_cents", "INTEGER NOT NULL"),
        # ``currency`` is the ISO 4217 currency of the money columns (CNY).
        # ``amount_unit`` is the *unit those columns are counted in*
        # (CNY_cents, i.e. integer minor units).  A unit is not a currency:
        # storing "CNY_cents" in a column named ``currency`` mislabels the
        # column, so the two concepts are published side by side instead.
        ("currency", "TEXT NOT NULL"), ("amount_unit", "TEXT NOT NULL"),
        ("sales_agent_id", "TEXT"), ("coupon_code", "TEXT"), ("created_at", "TEXT NOT NULL"),
    ],
    "refunds": [
        ("refund_id", "TEXT PRIMARY KEY"), ("order_id", "TEXT NOT NULL"),
        ("refunded_at", "TEXT NOT NULL"), ("local_month", "TEXT NOT NULL"),
        ("refund_cents", "INTEGER NOT NULL"), ("status", "TEXT NOT NULL"),
        ("reason_code", "TEXT NOT NULL"), ("reversal_of", "TEXT"), ("settled_at", "TEXT"),
    ],
    "order_items": [
        ("order_item_id", "TEXT PRIMARY KEY"), ("order_id", "TEXT NOT NULL"),
        ("product_id", "TEXT NOT NULL"), ("quantity", "INTEGER NOT NULL"),
        ("unit_price_cents", "INTEGER NOT NULL"), ("line_total_cents", "INTEGER NOT NULL"),
        ("discount_cents", "INTEGER NOT NULL DEFAULT 0"),
    ],
    "payments": [
        # ``paid_at`` is NULL for every unsuccessful attempt (pending/failed/
        # cancelled); only status='paid' attempts carry a settlement timestamp.
        ("payment_id", "TEXT PRIMARY KEY"), ("order_id", "TEXT NOT NULL"),
        ("paid_at", "TEXT"), ("attempted_at", "TEXT NOT NULL"), ("created_at", "TEXT NOT NULL"),
        ("amount_cents", "INTEGER NOT NULL"),
        ("status", "TEXT NOT NULL"), ("method", "TEXT NOT NULL"), ("provider", "TEXT NOT NULL"),
    ],
    "customers": [
        ("customer_id", "TEXT PRIMARY KEY"), ("full_name", "TEXT NOT NULL"),
        ("email", "TEXT NOT NULL"), ("city", "TEXT NOT NULL"), ("province", "TEXT NOT NULL"),
        ("segment", "TEXT NOT NULL"), ("registered_at", "TEXT NOT NULL"), ("is_active", "INTEGER NOT NULL"),
    ],
    "products": [
        ("product_id", "TEXT PRIMARY KEY"), ("category_id", "TEXT NOT NULL"),
        ("product_name", "TEXT NOT NULL"), ("unit_price_cents", "INTEGER NOT NULL"),
        ("cost_cents", "INTEGER NOT NULL"), ("is_active", "INTEGER NOT NULL"), ("launched_at", "TEXT NOT NULL"),
    ],
    "categories": [
        ("category_id", "TEXT PRIMARY KEY"), ("category_name", "TEXT NOT NULL"),
        ("parent_category_id", "TEXT"), ("margin_band", "TEXT NOT NULL"),
    ],
    "addresses": [
        ("address_id", "TEXT PRIMARY KEY"), ("customer_id", "TEXT NOT NULL"),
        ("address_type", "TEXT NOT NULL"), ("city", "TEXT NOT NULL"),
        ("province", "TEXT NOT NULL"), ("postcode", "TEXT NOT NULL"), ("updated_at", "TEXT NOT NULL"),
    ],
    "shipments": [
        ("shipment_id", "TEXT PRIMARY KEY"), ("order_id", "TEXT NOT NULL"),
        ("warehouse_id", "TEXT NOT NULL"), ("carrier", "TEXT NOT NULL"),
        ("shipped_at", "TEXT"), ("delivered_at", "TEXT"), ("freight_cents", "INTEGER NOT NULL"),
        ("status", "TEXT NOT NULL"),
    ],
    "promotions": [
        ("promotion_id", "TEXT PRIMARY KEY"), ("promotion_name", "TEXT NOT NULL"),
        ("channel_code", "TEXT NOT NULL"), ("discount_bps", "INTEGER NOT NULL"),
        ("starts_on", "TEXT NOT NULL"), ("ends_on", "TEXT NOT NULL"), ("status", "TEXT NOT NULL"),
    ],
    "order_promotions": [
        ("order_promotion_id", "TEXT PRIMARY KEY"), ("order_id", "TEXT NOT NULL"),
        ("promotion_id", "TEXT NOT NULL"), ("discount_cents", "INTEGER NOT NULL"),
        ("applied_at", "TEXT NOT NULL"),
    ],
    "inventory_movements": [
        ("movement_id", "TEXT PRIMARY KEY"), ("product_id", "TEXT NOT NULL"),
        ("warehouse_id", "TEXT NOT NULL"), ("movement_type", "TEXT NOT NULL"),
        ("quantity", "INTEGER NOT NULL"), ("moved_at", "TEXT NOT NULL"), ("reference_id", "TEXT"),
    ],
    "channel_daily_spend": [
        ("daily_spend_id", "TEXT PRIMARY KEY"), ("channel_code", "TEXT NOT NULL"),
        ("spend_date", "TEXT NOT NULL"), ("local_month", "TEXT NOT NULL"),
        ("campaign_name", "TEXT NOT NULL"), ("impressions", "INTEGER NOT NULL"),
        ("spend_cents", "INTEGER NOT NULL"),
    ],
    "support_ticket_links": [
        ("link_id", "TEXT PRIMARY KEY"), ("ticket_id", "TEXT NOT NULL"),
        ("order_id", "TEXT NOT NULL"), ("local_month", "TEXT NOT NULL"),
        ("link_reason", "TEXT NOT NULL"), ("confidence", "REAL NOT NULL"), ("linked_at", "TEXT NOT NULL"),
    ],
    "warehouses": [
        ("warehouse_id", "TEXT PRIMARY KEY"), ("warehouse_name", "TEXT NOT NULL"),
        ("city", "TEXT NOT NULL"), ("region", "TEXT NOT NULL"), ("capacity_units", "INTEGER NOT NULL"),
    ],
    "suppliers": [
        ("supplier_id", "TEXT PRIMARY KEY"), ("supplier_name", "TEXT NOT NULL"),
        ("country", "TEXT NOT NULL"), ("lead_time_days", "INTEGER NOT NULL"),
        ("payment_terms", "TEXT NOT NULL"), ("is_preferred", "INTEGER NOT NULL"),
    ],
    "product_suppliers": [
        ("product_supplier_id", "TEXT PRIMARY KEY"), ("product_id", "TEXT NOT NULL"),
        ("supplier_id", "TEXT NOT NULL"), ("unit_cost_cents", "INTEGER NOT NULL"),
        ("moq", "INTEGER NOT NULL"), ("is_primary", "INTEGER NOT NULL"),
    ],
    "channel_history": [
        ("history_id", "TEXT PRIMARY KEY"), ("channel_code", "TEXT NOT NULL"),
        ("effective_from", "TEXT NOT NULL"), ("effective_to", "TEXT"),
        ("version", "INTEGER NOT NULL"), ("status", "TEXT NOT NULL"),
        ("commission_bps", "INTEGER NOT NULL"), ("change_note", "TEXT NOT NULL"),
    ],
    "daily_fx_rates": [
        ("rate_id", "TEXT PRIMARY KEY"), ("rate_date", "TEXT NOT NULL"),
        ("base_currency", "TEXT NOT NULL"), ("quote_currency", "TEXT NOT NULL"),
        ("rate", "REAL NOT NULL"), ("source", "TEXT NOT NULL"),
    ],
    "sales_targets": [
        ("target_id", "TEXT PRIMARY KEY"), ("local_month", "TEXT NOT NULL"),
        ("channel_code", "TEXT NOT NULL"), ("target_net_cents", "INTEGER NOT NULL"),
        ("target_orders", "INTEGER NOT NULL"), ("owner", "TEXT NOT NULL"), ("version", "INTEGER NOT NULL"),
    ],
    "historical_orders": [
        ("historical_order_id", "TEXT PRIMARY KEY"), ("order_id", "TEXT NOT NULL"),
        ("recognized_at", "TEXT NOT NULL"), ("recognized_month", "TEXT NOT NULL"),
        ("revenue_cents", "INTEGER NOT NULL"), ("recognition_policy", "TEXT NOT NULL"),
        ("status", "TEXT NOT NULL"),
    ],
    "legacy_refunds": [
        ("legacy_refund_id", "TEXT PRIMARY KEY"), ("order_id", "TEXT NOT NULL"),
        ("recorded_at", "TEXT NOT NULL"), ("recorded_month", "TEXT NOT NULL"),
        ("amount_cents", "INTEGER NOT NULL"), ("allocation_policy", "TEXT NOT NULL"),
        ("status", "TEXT NOT NULL"),
    ],
    "marketing_clicks": [
        ("click_id", "TEXT PRIMARY KEY"), ("channel_code", "TEXT NOT NULL"),
        ("campaign_name", "TEXT NOT NULL"), ("clicked_at", "TEXT NOT NULL"),
        ("local_month", "TEXT NOT NULL"), ("click_count", "INTEGER NOT NULL"), ("cost_cents", "INTEGER NOT NULL"),
    ],
    "metric_audit_log": [
        ("audit_id", "TEXT PRIMARY KEY"), ("metric_id", "TEXT NOT NULL"),
        ("contract_version", "INTEGER NOT NULL"), ("action", "TEXT NOT NULL"),
        ("effective_from", "TEXT NOT NULL"), ("actor", "TEXT NOT NULL"), ("note", "TEXT NOT NULL"),
    ],
}
TABLE_ORDER = tuple(TABLE_SCHEMAS)
TABLE_NAMES = frozenset(TABLE_ORDER)
INDEX_STATEMENTS = (
    "CREATE INDEX idx_orders_paid_at ON orders(paid_at)",
    "CREATE INDEX idx_orders_attempted_at ON orders(attempted_at)",
    "CREATE INDEX idx_orders_local_month ON orders(local_month)",
    "CREATE INDEX idx_orders_customer ON orders(customer_id)",
    "CREATE INDEX idx_orders_channel ON orders(channel_code)",
    "CREATE INDEX idx_refunds_refunded_at ON refunds(refunded_at)",
    "CREATE INDEX idx_refunds_order ON refunds(order_id)",
    "CREATE INDEX idx_order_items_order ON order_items(order_id)",
    "CREATE INDEX idx_order_items_product ON order_items(product_id)",
    "CREATE INDEX idx_payments_order ON payments(order_id)",
    "CREATE INDEX idx_shipments_order ON shipments(order_id)",
    "CREATE INDEX idx_order_promotions_order ON order_promotions(order_id)",
    "CREATE INDEX idx_inventory_movements_product ON inventory_movements(product_id)",
    "CREATE INDEX idx_channel_daily_spend_month ON channel_daily_spend(local_month)",
    "CREATE INDEX idx_support_ticket_links_order ON support_ticket_links(order_id)",
    "CREATE INDEX idx_support_ticket_links_ticket ON support_ticket_links(ticket_id)",
    "CREATE INDEX idx_addresses_customer ON addresses(customer_id)",
    "CREATE INDEX idx_product_suppliers_product ON product_suppliers(product_id)",
    "CREATE INDEX idx_marketing_clicks_month ON marketing_clicks(local_month)",
)

CITIES = (("Sydney", "NSW"), ("Melbourne", "VIC"), ("Brisbane", "QLD"), ("Perth", "WA"), ("Adelaide", "SA"), ("Canberra", "ACT"))
SEGMENTS = ("consumer", "smb", "enterprise", "vip", "partner")
SURNAMES = ("陈", "林", "黄", "张", "李", "王", "吴", "刘", "蔡", "杨", "许", "郑", "谢", "洪", "周", "叶")
GIVEN_NAMES = ("伟", "芳", "娜", "敏", "静", "磊", "洋", "勇", "艳", "杰", "涛", "明", "超", "霞", "平", "刚")
CATEGORY_DEFS = (
    ("c01", "家居生活", None, "mid"), ("c02", "消费电子", None, "high"), ("c03", "服饰鞋包", None, "mid"),
    ("c04", "食品饮料", None, "low"), ("c05", "运动户外", None, "mid"), ("c06", "美妆个护", None, "high"),
    ("c07", "母婴玩具", None, "mid"), ("c08", "图书文具", None, "low"),
    ("c011", "厨房用具", "c01", "mid"), ("c012", "床上用品", "c01", "mid"),
    ("c021", "手机配件", "c02", "high"), ("c022", "影音设备", "c02", "high"),
    ("c031", "男装", "c03", "mid"), ("c032", "女装", "c03", "mid"),
    ("c041", "咖啡茶饮", "c04", "low"), ("c051", "健身器材", "c05", "mid"),
)
PRODUCT_NOUNS = {
    "c01": ("收纳箱", "折叠桌", "香薰灯", "落地衣架"),
    "c02": ("蓝牙耳机", "便携充电宝", "智能手表", "无线音箱"),
    "c03": ("通勤背包", "羊毛大衣", "速干T恤", "帆布鞋"),
    "c04": ("挂耳咖啡", "乌龙茶包", "坚果礼盒", "气泡水"),
    "c05": ("瑜伽垫", "登山杖", "哑铃组", "骑行头盔"),
    "c06": ("精华液", "防晒霜", "面膜套装", "洗发水"),
    "c07": ("婴儿推车", "积木套装", "儿童水杯", "婴儿湿巾"),
    "c08": ("笔记本", "中性笔套装", "儿童绘本", "桌面台历"),
}
PRODUCT_QUALIFIERS = ("基础款", "进阶款", "轻量版", "旗舰版", "经典款", "限定款", "环保版", "专业版")
SUPPLIER_NAMES = ("华南智造供应链", "长三角日用制造", "珠三角电子科技", "华北食品贸易", "西南纺织集团", "东南家居实业", "中原包装材料", "海外直采联盟")
WAREHOUSE_DEFS = (("w01", "悉尼西区仓", "Sydney", "NSW", 48000), ("w02", "墨尔本南仓", "Melbourne", "VIC", 36000), ("w03", "布里斯班北仓", "Brisbane", "QLD", 21000), ("w04", "珀斯转运仓", "Perth", "WA", 12000))
CARRIERS = ("AusPost", "StarTrack", "Toll", "Sendle", "DHL eCommerce")
PAYMENT_METHODS = ("card", "wallet", "bank_transfer", "voucher")
PAYMENT_PROVIDERS = ("acquirer_a", "acquirer_b", "gateway_c")
REFUND_REASONS = ("damaged_item", "wrong_item", "late_delivery", "customer_remorse", "duplicate_charge", "price_adjustment")
FREIGHT_BY_CHANNEL = {"web": (900, 1800), "app": (800, 1700), "marketplace": (600, 1400), "partner": (1200, 2400)}
#: channel mix drifts upward over the fiscal year; web recedes, marketplace grows.
CHANNEL_BASE_WEIGHTS = ((58, 24, 11, 7), (52, 25, 15, 8))
#: daily order volume multiplier by month (seasonality, never published as an answer).
MONTH_SEASONALITY = (0.86, 1.12, 1.34, 0.78, 0.82, 0.94, 0.97, 0.99, 1.02, 1.05, 1.08, 1.14)
#: base share of a month's orders that get any refund attempt.
BASE_REFUND_SHARE = 0.30
#: September web refund attempt share is deliberately raised so that a channel
#: and month comparison has an explainable driver; the document set only states
#: the policy change, never the resulting amounts.
SEPTEMBER_WEB_REFUND_SHARE = 0.66
CHANNEL_REFUND_SHARE = {"web": 1.0, "app": 0.78, "marketplace": 0.70, "partner": 0.60}
#: Minimum delay between the successful payment instant and a refund event.
#: A refund month is eligible only while ``month_end > paid_at + delay``, which
#: keeps same-month refunds possible (the payment month stays eligible as long as
#: any time is left in it after the payment) and never places a refund at or
#: before the payment it reverses.
REFUND_REQUIRED_DELAY = timedelta(hours=2)
#: September uses a shifted event mix: more reversals, fewer partial reversals.
#: Event-shape table: (cumulative threshold, number of refund events for the order).
REVERSAL_TABLE = ((0.72, 1), (1.00, 2))
REVERSAL_TABLE_SEPTEMBER_WEB = ((0.48, 1), (0.85, 2), (1.00, 3))
PROMOTION_DEFS = (
    ("p01", "金秋家居焕新", "web", 800), ("p02", "移动端专享立减", "app", 500),
    ("p03", "平台大促满减", "marketplace", 1200), ("p04", "合作伙伴联合券", "partner", 600),
    ("p05", "春季新品首发", "web", 700), ("p06", "会员积分抵扣", "app", 400),
)
#: Inclusive business-month window in which each promotion may be applied.
PROMOTION_WINDOWS = {
    "p01": ("2025-10", "2025-11"), "p02": ("2025-10", "2025-12"),
    "p03": ("2025-10", "2026-03"), "p04": ("2025-11", "2026-06"),
    "p05": ("2026-02", "2026-04"), "p06": ("2025-10", "2026-09"),
}
#: Deterministic take-up rate: not every eligible order actually applies the
#: promotion, so ``order_promotions`` never covers all orders.
PROMOTION_TAKE_RATE = 0.32
CAMPAIGNS = ("brand_search", "social_feed", "retargeting", "marketplace_banner", "partner_newsletter", "affiliate_network")
TICKET_CONTEXTS = (
    ("payment UI", ("付款按钮点击后没有反应，换了浏览器也一样。", "支付页面一直转圈，最后提示超时，我不确定有没有扣款。", "同一笔订单重复弹出支付确认框，页面刷新后才消失。", "支付页面提示网络异常，但我这边网络是正常的。")),
    ("delivery", ("下单后一周都没有物流更新，想确认什么时候能发货。", "快递显示已签收但我没有收到，希望帮忙查一下配送。", "配送地址想改成公司前台，包裹还在途中。", "包裹外包装破损，里面的固定件松了。")),
    ("product description", ("商品详情里的尺寸和实际收到的标签不一致。", "页面描述的材质与实物手感差别较大。", "说明书是英文的，找不到中文版本。", "图片颜色和实物色差明显，想确认是不是同一款。")),
    ("refund timing", ("退款申请提交三天了，还没看到到账记录。", "客服说退款已处理，但银行账单里还没有这笔。", "部分退款到账了，剩下的金额不清楚什么时候退。", "退回的运费是否包含在退款金额里？")),
    ("unrelated", ("想问一下你们是否支持企业批量采购开票。", "希望增加周末配送的选项。", "你们在珀斯有线下自提点吗？", "注册会员的积分规则在哪里查看？")),
)


class DatasetError(ValueError):
    """Raised for invalid parameters or refused targets."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def local_month(stamp: datetime) -> str:
    """Approved business month for an event timestamp, in Australia/Sydney."""
    if stamp.tzinfo is None:
        raise ValueError("Event timestamps must be timezone-aware")
    return stamp.astimezone(ZoneInfo(ZONE)).strftime("%Y-%m")


def _order_month(status: str, paid_stamp: datetime, attempted_stamp: datetime) -> str:
    """Business month attributed to an order row.

    Paid orders are attributed to the month of the *successful payment event*
    (``paid_at``), which is what the approved CASH_NET_v1 contract aggregates on.
    Orders that never succeeded have no ``paid_at`` at all, so their month is
    derived from the attempt stamp instead.  This fallback is a descriptive
    attribution for non-paid rows only; it never substitutes for a payment
    timestamp and non-paid rows still carry ``paid_at = NULL``.
    """
    return local_month(paid_stamp if status == "paid" else attempted_stamp)


def _iso(stamp: datetime) -> str:
    return stamp.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_month(month: str) -> tuple[int, int]:
    return int(month[:4]), int(month[5:7])


def _month_bounds(month: str) -> tuple[datetime, datetime]:
    year, mon = _parse_month(month)
    start = datetime(year, mon, 1, tzinfo=ZoneInfo(ZONE))
    end = datetime(year + (mon == 12), (mon % 12) + 1, 1, tzinfo=ZoneInfo(ZONE))
    return start, end


def _month_starts() -> dict[str, datetime]:
    return {month: _month_bounds(month)[0] for month in BUSINESS_MONTHS}


def _month_days(month: str) -> int:
    start, end = _month_bounds(month)
    return (end - start).days


def _random_month_day(rng: random.Random, month: str) -> datetime:
    start, _ = _month_bounds(month)
    return start + timedelta(days=rng.randrange(_month_days(month)), seconds=rng.randrange(86400))


def _weighted_key(rng: random.Random, weights: dict[str, float]) -> str:
    total = sum(weights.values())
    point = rng.random() * total
    for key in sorted(weights):
        point -= weights[key]
        if point <= 0:
            return key
    return sorted(weights)[-1]


def _business_days(month: str) -> list[str]:
    start, end = _month_bounds(month)
    days = []
    cursor = start
    while cursor < end:
        if cursor.weekday() < 5:
            days.append(cursor.date().isoformat())
        cursor += timedelta(days=1)
    return days


def _fixture_folder(project_root: Path, fixture: str) -> Path:
    return Path(project_root).resolve() / "var" / "sources" / fixture


def _rng_seed(seed: int, order_count: int, ticket_count: int, document_count: int) -> str:
    """Random-stream seed built from generation parameters only.

    The fixture *name* is deliberately excluded: it is a publication label, not
    an input to the data.  Two different fixture names generated with the same
    size parameters therefore produce identical logical business data, which is
    what makes the fixture portable and independently reproducible.
    """
    return f"dataset-v2|{seed}|{order_count}|{ticket_count}|{document_count}"


def _validate_parameters(fixture, seed, order_count, ticket_count, document_count) -> None:
    if not isinstance(fixture, str) or not fixture or any(c not in SAFE_CHARACTERS for c in fixture):
        raise DatasetError("Invalid fixture identifier: use letters, digits, hyphen and underscore only")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DatasetError("Seed must be a nonnegative integer between 0 and 2147483647")
    if not 0 <= seed <= 2 ** 31 - 1:
        raise DatasetError("Seed must be a nonnegative integer between 0 and 2147483647")
    for name, value in (("orders", order_count), ("tickets", ticket_count), ("documents", document_count)):
        low, high = LIMITS[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise DatasetError(f"{name} must be an integer")
        if not low <= value <= high:
            raise DatasetError(f"{name} must be between {low} and {high}")


# --------------------------------------------------------------------------
# entity construction
# --------------------------------------------------------------------------

def _build_customers(rng: random.Random, count: int) -> list[dict]:
    rows, used = [], set()
    for i in range(count):
        city, province = CITIES[rng.randrange(len(CITIES))]
        candidate = rng.choice(SURNAMES) + rng.choice(GIVEN_NAMES)
        if rng.random() < 0.35:
            candidate += rng.choice(GIVEN_NAMES)
        # disambiguate rather than loop forever once the name space is exhausted.
        name, suffix = candidate, 2
        while name in used:
            name = f"{candidate}{suffix}"
            suffix += 1
        used.add(name)
        # Registrations sit well before the first covered business month
        # (2025-10): a customer that registers after its own first order would be
        # an impossible lifecycle record.
        registered = datetime(2021, 6, 1, tzinfo=timezone.utc) + timedelta(days=rng.randrange(1200), seconds=rng.randrange(86400))
        rows.append({
            "customer_id": f"C{i + 1:05d}", "full_name": name,
            "email": f"customer{i + 1:05d}@example-{province.lower()}.test",
            "city": city, "province": province, "segment": rng.choice(SEGMENTS),
            "registered_at": _iso(registered), "is_active": 1 if rng.random() < 0.94 else 0,
        })
    return rows


def _build_categories() -> list[dict]:
    return [{"category_id": cid, "category_name": name, "parent_category_id": parent, "margin_band": band}
            for cid, name, parent, band in CATEGORY_DEFS]


def _build_products(rng: random.Random, count: int) -> list[dict]:
    parents = {cid for cid, _, _, _ in CATEGORY_DEFS if _category_parent(cid) is None}
    leaves = [cid for cid, _, _, _ in CATEGORY_DEFS if cid not in parents]
    base = {"low": (1900, 5900), "mid": (4900, 24900), "high": (19900, 129900)}
    bands = {row["category_id"]: row["margin_band"] for row in _build_categories()}
    rows, used = [], set()
    for i in range(count):
        category = leaves[i % len(leaves)]
        low, high = base[bands[category]]
        candidate = f"{rng.choice(PRODUCT_NOUNS[category[:3]])} {rng.choice(PRODUCT_QUALIFIERS)}"
        name, suffix = candidate, 2
        while name in used:
            name = f"{candidate} {suffix}"
            suffix += 1
        used.add(name)
        price = rng.randrange(low, high + 1, 100)
        rows.append({
            "product_id": f"P{i + 1:04d}", "category_id": category, "product_name": name,
            "unit_price_cents": price, "cost_cents": int(price * rng.uniform(0.45, 0.78) // 100 * 100),
            "is_active": 1 if rng.random() < 0.92 else 0,
            "launched_at": _iso(datetime(2023, 6, 1, tzinfo=timezone.utc) + timedelta(days=rng.randrange(800), seconds=rng.randrange(86400))),
        })
    return rows


def _category_parent(category_id: str) -> str | None:
    for cid, _, parent, _ in CATEGORY_DEFS:
        if cid == category_id:
            return parent
    return None


def _build_suppliers(rng: random.Random) -> list[dict]:
    countries = ("CN", "VN", "MY", "AU", "TH", "KR")
    return [{"supplier_id": f"S{i + 1:03d}", "supplier_name": name, "country": countries[i % len(countries)],
             "lead_time_days": rng.randrange(7, 61), "payment_terms": rng.choice(("NET30", "NET45", "NET60", "PREPAID")),
             "is_preferred": 1 if i < 3 else 0} for i, name in enumerate(SUPPLIER_NAMES)]


def _build_warehouses() -> list[dict]:
    return [{"warehouse_id": wid, "warehouse_name": name, "city": city, "region": region, "capacity_units": capacity}
            for wid, name, city, region, capacity in WAREHOUSE_DEFS]


def _build_addresses(rng: random.Random, customers: list[dict]) -> list[dict]:
    rows, counter = [], 0
    stamps = {"shipping": 0.85, "billing": 0.45}
    for customer in customers:
        for address_type, probability in stamps.items():
            if rng.random() > probability:
                continue
            counter += 1
            rows.append({
                "address_id": f"A{counter:06d}", "customer_id": customer["customer_id"],
                "address_type": address_type, "city": customer["city"], "province": customer["province"],
                "postcode": f"{rng.randrange(2000, 2999)}",
                "updated_at": _iso(datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=rng.randrange(600), seconds=rng.randrange(86400))),
            })
    return rows


def _build_promotions() -> list[dict]:
    starts = {"p01": "2025-09-20", "p02": "2025-10-01", "p03": "2025-10-15",
              "p04": "2025-11-01", "p05": "2026-02-01", "p06": "2025-10-01"}
    ends = {"p01": "2025-11-30", "p02": "2025-12-31", "p03": "2026-03-31",
            "p04": "2026-06-30", "p05": "2026-04-30", "p06": "2026-09-30"}
    statuses = {"p01": "ended", "p02": "ended", "p03": "ended", "p04": "ended", "p05": "ended", "p06": "active"}
    return [{"promotion_id": pid, "promotion_name": name, "channel_code": channel, "discount_bps": bps,
             "starts_on": starts[pid], "ends_on": ends[pid], "status": statuses[pid]}
            for pid, name, channel, bps in PROMOTION_DEFS]


def _build_product_suppliers(rng: random.Random, products: list[dict], suppliers: list[dict]) -> list[dict]:
    rows = []
    for product in products:
        count = rng.choice((1, 1, 1, 2))
        chosen = rng.sample(suppliers, count)
        for index, supplier in enumerate(chosen):
            rows.append({
                "product_supplier_id": f"PS{len(rows) + 1:05d}", "product_id": product["product_id"],
                "supplier_id": supplier["supplier_id"],
                "unit_cost_cents": max(100, int(product["cost_cents"] * rng.uniform(0.85, 1.15) // 100 * 100)),
                "moq": rng.choice((50, 100, 200, 500)), "is_primary": 1 if index == 0 else 0,
            })
    return rows


def _build_channel_history() -> list[dict]:
    rows = []
    commission = {"web": 250, "app": 180, "marketplace": 1200, "partner": 900}
    for code, _, _, _ in CHANNELS:
        rows.append({"history_id": f"CH{len(rows) + 1:03d}", "channel_code": code,
                     "effective_from": "2020-01-01", "effective_to": "2025-09-30", "version": 1,
                     "status": "SUPERSEDED", "commission_bps": commission[code] * 2,
                     "change_note": "初始渠道合同版本，仅用于历史对比。"})
        rows.append({"history_id": f"CH{len(rows) + 1:03d}", "channel_code": code,
                     "effective_from": "2025-10-01", "effective_to": None, "version": 2,
                     "status": "CURRENT", "commission_bps": commission[code],
                     "change_note": "第二版合同自本财年首月起生效；不可用于回溯第一版期间。"})
    return rows


def _build_channel_daily_spend(rng: random.Random, seed: int) -> list[dict]:
    rows = []
    for month, channel in itertools.product(BUSINESS_MONTHS, CHANNEL_CODES):
        for index, day in enumerate(_business_days(month)):
            if rng.random() < 0.35:
                continue
            campaign = CAMPAIGNS[(seed + len(rows) + index) % len(CAMPAIGNS)]
            rows.append({
                "daily_spend_id": f"DS{len(rows) + 1:06d}", "channel_code": channel, "spend_date": day,
                "local_month": month, "campaign_name": campaign,
                "impressions": rng.randrange(4000, 240000),
                "spend_cents": rng.randrange(5000, 480000),
            })
    return rows


def _build_marketing_clicks(rng: random.Random) -> list[dict]:
    rows = []
    for month, channel in itertools.product(BUSINESS_MONTHS, CHANNEL_CODES):
        for campaign in CAMPAIGNS[:3]:
            day = rng.randrange(1, _month_days(month) + 1)
            stamp = datetime(*_parse_month(month), day, rng.randrange(24), tzinfo=ZoneInfo(ZONE))
            rows.append({
                "click_id": f"MC{len(rows) + 1:05d}", "channel_code": channel, "campaign_name": campaign,
                "clicked_at": _iso(stamp), "local_month": month,
                "click_count": rng.randrange(120, 9000), "cost_cents": rng.randrange(2000, 90000),
            })
    return rows


def _build_daily_fx_rates() -> list[dict]:
    rows, day = [], datetime(2025, 9, 1, tzinfo=timezone.utc)
    while day.date().isoformat() <= "2026-09-30":
        rows.append({"rate_id": f"FX{len(rows) + 1:05d}", "rate_date": day.date().isoformat(),
                     "base_currency": "USD", "quote_currency": "CNY",
                     "rate": round(7.05 + (len(rows) % 37) * 0.004, 4), "source": "synthetic-desk"})
        day += timedelta(days=1)
    return rows


def _build_metric_audit_log() -> list[dict]:
    return [
        {"audit_id": "MA001", "metric_id": "CASH_NET_v1", "contract_version": 1, "action": "APPROVED",
         "effective_from": "2025-10-01", "actor": "synthetic-commerce-owner",
         "note": "按事件自身 UTC 时间转换到 Australia/Sydney 后的日历月汇总成功支付减成功退款。"},
        {"audit_id": "MA002", "metric_id": "CASH_NET_v1", "contract_version": 1, "action": "SCOPE_CONFIRMED",
         "effective_from": "2025-10-01", "actor": "synthetic-commerce-owner",
         "note": "单位 CNY 整数分；渠道取原订单 channel_code；缺失或重复渠道映射必须报错。"},
        {"audit_id": "MA003", "metric_id": "CASH_NET_v1", "contract_version": 1, "action": "RETAINED",
         "effective_from": "2026-07-01", "actor": "synthetic-commerce-owner",
         "note": "本财年继续沿用 version 1；未批准任何替代口径。"},
        {"audit_id": "MA004", "metric_id": "CASH_NET_v1", "contract_version": 0, "action": "SUPERSEDED",
         "effective_from": "2024-07-01", "actor": "legacy-finance",
         "note": "旧版按支付月回溯扣减退款，已被 version 1 取代，不得用于当前口径。"},
    ]


def _build_sales_targets(rng: random.Random) -> list[dict]:
    owners = {code: f"growth-{code}" for code in CHANNEL_CODES}
    rows = []
    for month, channel in itertools.product(BUSINESS_MONTHS, CHANNEL_CODES):
        base = {"web": 4_200_000, "app": 2_600_000, "marketplace": 1_400_000, "partner": 900_000}[channel]
        rows.append({"target_id": f"ST{len(rows) + 1:04d}", "local_month": month, "channel_code": channel,
                     "target_net_cents": base + rng.randrange(0, 40) * 25_000,
                     "target_orders": rng.randrange(300, 3200), "owner": owners[channel],
                     "version": 2 if month >= "2026-07" else 1})
    return rows


def _promotion_months(promotion_id: str) -> tuple[str, str]:
    return PROMOTION_WINDOWS[promotion_id]


def _promotion_channel(promotion_id: str) -> str:
    for pid, _, code, _ in PROMOTION_DEFS:
        if pid == promotion_id:
            return code
    raise DatasetError(f"Unknown promotion: {promotion_id}")


def _build_promotion_plan() -> dict[str, dict[str, tuple[str, int]]]:
    """Map each (month, channel) to the (promotion_id, discount_bps) live there.

    At most one promotion per channel is live in any month by construction; the
    window is re-checked at order time so a plan entry can never leak a
    promotion outside its declared ``starts_on``/``ends_on`` span.
    """
    channel = {pid: code for pid, _, code, _ in PROMOTION_DEFS}
    bps = {pid: value for pid, _, _, value in PROMOTION_DEFS}
    plan: dict[str, dict[str, tuple[str, int]]] = {}
    for month in BUSINESS_MONTHS:
        for pid, (start, end) in PROMOTION_WINDOWS.items():
            if start <= month <= end:
                plan.setdefault(month, {}).setdefault(channel[pid], (pid, bps[pid]))
    return plan


def _order_days(rng: random.Random, order_count: int) -> list[tuple[str, datetime]]:
    """Assign each order a business month and an aware UTC timestamp."""
    weights = {month: MONTH_SEASONALITY[index] * _month_days(month) for index, month in enumerate(BUSINESS_MONTHS)}
    total = sum(weights.values())
    quota = {month: int(order_count * weights[month] / total) for month in BUSINESS_MONTHS}
    # largest-remainder apportionment keeps the total exact and deterministic.
    remainder = order_count - sum(quota.values())
    ranked = sorted(BUSINESS_MONTHS, key=lambda month: (-(order_count * weights[month] / total - quota[month]), month))
    for month in ranked[:remainder]:
        quota[month] += 1
    days = [month for month in BUSINESS_MONTHS for _ in range(quota[month])]
    rng.shuffle(days)
    return [(month, _random_month_day(rng, month)) for month in days]


def _item_count(rng: random.Random) -> int:
    point = rng.random()
    return 1 if point < 0.52 else 2 if point < 0.85 else 3


def _allocate_discount(discount_cents: int, line_gross: list[int]) -> list[int]:
    """Split ``discount_cents`` across lines in integer cents, exactly.

    Uses largest-remainder apportionment on the gross line amounts so that the
    allocated cents always sum back to ``discount_cents`` and no line is
    discounted below zero.  Deterministic: ties break on line order.
    """
    if discount_cents < 0:
        raise DatasetError("Promotion discount cannot be negative")
    total = sum(line_gross)
    if not line_gross or total <= 0:
        if discount_cents:
            raise DatasetError("Cannot allocate a discount across zero-value lines")
        return [0] * len(line_gross)
    if discount_cents > total:
        raise DatasetError("Promotion discount exceeds the order gross amount")
    raw = [discount_cents * gross / total for gross in line_gross]
    floors = [int(value) for value in raw]
    remainder = discount_cents - sum(floors)
    order = sorted(range(len(line_gross)), key=lambda i: (-(raw[i] - floors[i]), i))
    for index in order[:remainder]:
        floors[index] += 1
    return floors


def _live_promotions(month: str, channel: str) -> tuple[tuple[str, int], ...]:
    """Every (promotion_id, discount_bps) genuinely live for a month/channel.

    The promotion must be inside its inclusive ``PROMOTION_WINDOWS`` span and
    must be scoped to the same channel; both conditions come from the published
    ``promotions`` table, never from a free-standing constant.
    """
    live = []
    for pid, _, code, bps in PROMOTION_DEFS:
        start, end = PROMOTION_WINDOWS[pid]
        if code == channel and start <= month <= end:
            live.append((pid, bps))
    return tuple(live)


def _promotion_discount(rng: random.Random, plan: tuple[str, int], month: str, channel: str,
                        order_gross_cents: int) -> tuple[str, int]:
    """Resolve whether a genuinely applied promotion covers this order.

    A promotion is *applied* only when its window contains the order's month and
    its channel matches; the discount is then a real integer-cent amount derived
    from its ``discount_bps`` and the order's *gross* line sum, floored at zero
    and capped so the order can never go negative.  Orders that are not covered
    by any live promotion, or that the (deterministic) lottery skips, return
    ``("", 0)`` and never appear in ``order_promotions``.
    """
    promotion_id, discount_bps = plan
    if not promotion_id:
        return "", 0
    # ``PROMOTION_WINDOWS`` is an *inclusive* (start, end) month span, so the
    # window test must be ``start <= month <= end``.  Testing membership in the
    # two-element tuple instead would reject every interior month and silently
    # collapse multi-month promotions to their two endpoints.
    window_start, window_end = _promotion_months(promotion_id)
    if not window_start <= month <= window_end:
        return "", 0
    if channel != _promotion_channel(promotion_id):
        return "", 0
    if rng.random() >= PROMOTION_TAKE_RATE:
        return "", 0
    # discount_bps is basis points of the gross order value, in integer cents.
    discount = order_gross_cents * discount_bps // 10_000
    if discount <= 0:
        return "", 0
    return promotion_id, min(discount, order_gross_cents)


def _build_business(rng: random.Random, seed: int, orders: int) -> dict:
    customers = _build_customers(rng, 2000)
    categories = _build_categories()
    products = _build_products(rng, 480)
    suppliers = _build_suppliers(rng)
    warehouses = _build_warehouses()
    addresses = _build_addresses(rng, customers)
    promotions = _build_promotions()
    promotion_plan = _build_promotion_plan()
    product_suppliers = _build_product_suppliers(rng, products, suppliers)
    channel_history = _build_channel_history()
    daily_spend = _build_channel_daily_spend(rng, seed)
    clicks = _build_marketing_clicks(rng)
    fx_rates = _build_daily_fx_rates()
    audit_log = _build_metric_audit_log()
    targets = _build_sales_targets(rng)

    order_rows: list[dict] = []
    item_rows: list[dict] = []
    payment_rows: list[dict] = []
    refund_rows: list[dict] = []
    shipment_rows: list[dict] = []
    order_promotion_rows: list[dict] = []
    inventory_rows: list[dict] = []
    historical_rows: list[dict] = []
    legacy_rows: list[dict] = []
    coupons = ("WELCOME10", "SAVE20", "FREESHIP", "MEMBER5", None, None, None, None)
    paid_orders: list[dict] = []
    month_starts = _month_starts()

    day_plan = _order_days(rng, orders)
    for number, (month, stamp) in enumerate(day_plan, start=1):
        progress = BUSINESS_MONTHS.index(month) / (len(BUSINESS_MONTHS) - 1)
        weights = {code: CHANNEL_BASE_WEIGHTS[0][i] * (1 - progress) + CHANNEL_BASE_WEIGHTS[1][i] * progress
                   for i, code in enumerate(CHANNEL_CODES)}
        channel = _weighted_key(rng, weights)
        status = "paid" if rng.random() < 0.955 else rng.choice(ORDER_STATUSES[1:])
        order_id = f"O{number:06d}"
        customer = customers[rng.randrange(len(customers))]
        created = stamp - timedelta(minutes=rng.randrange(5, 4320))
        # The order's own event stamp *is* the attempt instant, and it is the one
        # the business calendar apportions into a covered month.  Deriving the
        # attempt as ``created + 60..1800s`` could land days before that month
        # began, because ``created`` precedes ``stamp``: the row's declared month
        # would then disagree with the timestamp it is derived from, and the
        # attempt would fall outside the 12-month window.  ``attempted_at`` is
        # therefore the event stamp itself, and ``paid_at`` repeats the same
        # stamp for successful orders only (NULL for every non-paid status, which
        # keeps ``paid_cents = 0`` and the attempt explicit but unsettled).
        attempted = stamp
        order_month = _order_month(status, stamp, attempted)
        quantity_plan = [(_choose_product(rng, products), _item_count(rng))
                         for _ in range(_item_count(rng))]
        gross_lines = [product["unit_price_cents"] * quantity for product, quantity in quantity_plan]
        order_gross = sum(gross_lines)

        # Decide the genuinely applied promotion *before* building lines so its
        # discount can be allocated into the line rows themselves.  The lookup
        # uses the order's own derived local month, which is also what the
        # post-generation self check re-derives, so the two can never disagree.
        plan = promotion_plan.get(order_month, {}).get(channel, ("", 0))
        applied_promotion, promotion_discount = _promotion_discount(rng, plan, order_month, channel, order_gross)
        allocations = _allocate_discount(promotion_discount, gross_lines)
        net_lines = [gross - discount for gross, discount in zip(gross_lines, allocations)]
        order_total = sum(net_lines)

        order_rows.append({
            "order_id": order_id, "customer_id": customer["customer_id"],
            "paid_at": _iso(stamp) if status == "paid" else None,
            "attempted_at": _iso(attempted),
            "order_date": (stamp if status == "paid" else attempted).astimezone(ZoneInfo(ZONE)).date().isoformat(),
            "local_month": order_month,
            "channel_code": channel, "status": status,
            "paid_cents": order_total if status == "paid" else 0,
            "order_total_cents": order_total,
            "currency": CURRENCY_CODE,
            "amount_unit": AMOUNT_UNIT,
            "sales_agent_id": f"AG{1 + (number % 24):02d}", "coupon_code": rng.choice(coupons),
            "created_at": _iso(created),
        })
        for index, ((product, quantity), discount, line_total) in enumerate(
                zip(quantity_plan, allocations, net_lines), start=1):
            item_rows.append({
                "order_item_id": f"{order_id}-I{index}", "order_id": order_id,
                "product_id": product["product_id"], "quantity": quantity,
                "unit_price_cents": product["unit_price_cents"], "line_total_cents": line_total,
                "discount_cents": discount,
            })
        payment_rows.append({
            "payment_id": f"PAY{number:06d}", "order_id": order_id,
            "paid_at": _iso(stamp) if status == "paid" else None,
            "attempted_at": _iso(attempted),
            "created_at": _iso(attempted + timedelta(seconds=rng.randrange(30, 900))),
            "amount_cents": order_total if status == "paid" else 0, "status": status,
            "method": rng.choice(PAYMENT_METHODS), "provider": rng.choice(PAYMENT_PROVIDERS),
        })
        shipment = None
        if status == "paid":
            paid_orders.append(order_rows[-1])
            low, high = FREIGHT_BY_CHANNEL[channel]
            shipped = stamp + timedelta(days=rng.randrange(0, 3), hours=rng.randrange(6, 30))
            delivered = shipped + timedelta(days=rng.randrange(1, 9), hours=rng.randrange(4, 20))
            shipment = {
                "shipment_id": f"SH{len(shipment_rows) + 1:06d}", "order_id": order_id,
                "warehouse_id": warehouses[rng.randrange(len(warehouses))]["warehouse_id"],
                "carrier": rng.choice(CARRIERS), "shipped_at": _iso(shipped),
                "delivered_at": _iso(delivered), "freight_cents": rng.randrange(low, high + 1, 100),
                "status": "delivered",
            }
            shipment_rows.append(shipment)
            # One outflow per purchased item line, carrying the exact product and
            # the warehouse that actually shipped the order.  Only paid shipped
            # orders create sale_out movements.
            for product, quantity in quantity_plan:
                inventory_rows.append({
                    "movement_id": f"IM{len(inventory_rows) + 1:07d}",
                    "product_id": product["product_id"],
                    "warehouse_id": shipment["warehouse_id"],
                    "movement_type": "sale_out", "quantity": -quantity, "moved_at": _iso(shipped),
                    "reference_id": order_id,
                })
        if applied_promotion and promotion_discount > 0:
            order_promotion_rows.append({
                "order_promotion_id": f"OP{len(order_promotion_rows) + 1:06d}", "order_id": order_id,
                "promotion_id": applied_promotion, "discount_cents": promotion_discount,
                "applied_at": _iso(created + timedelta(seconds=rng.randrange(10, 600))),
            })
        if rng.random() < 0.42:
            recognized = month_starts[month] + timedelta(days=rng.randrange(6), hours=9)
            historical_rows.append({
                "historical_order_id": f"HO{len(historical_rows) + 1:06d}", "order_id": order_id,
                "recognized_at": _iso(recognized), "recognized_month": local_month(recognized),
                "revenue_cents": order_total, "recognition_policy": "order_cohort_recognized_v1",
                "status": "SUPERSEDED",
            })

    refund_rows, legacy_rows = _build_refunds(rng, paid_orders, month_starts)

    return {
        "customers": customers, "categories": categories, "products": products,
        "suppliers": suppliers, "warehouses": warehouses, "addresses": addresses,
        "promotions": promotions, "product_suppliers": product_suppliers,
        "channel_history": channel_history, "channel_daily_spend": daily_spend,
        "marketing_clicks": clicks, "daily_fx_rates": fx_rates,
        "metric_audit_log": audit_log, "sales_targets": targets,
        "orders": order_rows, "order_items": item_rows, "payments": payment_rows,
        "refunds": refund_rows, "shipments": shipment_rows,
        "order_promotions": order_promotion_rows, "inventory_movements": inventory_rows,
        "historical_orders": historical_rows, "legacy_refunds": legacy_rows,
    }, paid_orders


def _choose_product(rng: random.Random, products: list[dict]) -> dict:
    return products[rng.randrange(len(products))]


def _build_refunds(rng: random.Random, paid_orders: list[dict], month_starts: dict) -> tuple[list[dict], list[dict]]:
    """Create refund events; each order's successful total never exceeds paid_cents."""
    refunds: list[dict] = []
    legacy: list[dict] = []
    for order in paid_orders:
        month = order["local_month"]
        share = BASE_REFUND_SHARE * CHANNEL_REFUND_SHARE[order["channel_code"]]
        if month == "2026-09" and order["channel_code"] == "web":
            share = SEPTEMBER_WEB_REFUND_SHARE
        if rng.random() >= share:
            continue
        table = REVERSAL_TABLE_SEPTEMBER_WEB if (month == "2026-09" and order["channel_code"] == "web") else REVERSAL_TABLE
        point, reversals = rng.random(), table[-1][1]
        for threshold, count in table:
            if point < threshold:
                reversals = count
                break
        paid_at = datetime.fromisoformat(order["paid_at"].replace("Z", "+00:00"))
        remaining = order["paid_cents"]
        # A refund can only happen after the payment, and the event must also
        # have room to land before the month closes.  A month is therefore
        # eligible when ``month_end > paid_at + delay``; the *same* month as the
        # payment is eligible whenever enough of it is left after the payment
        # instant.  Filtering on ``month_starts[m] >= paid_at`` instead would
        # drop the payment month for every order that paid after the 1st (i.e.
        # almost all of them), and would drop the final business month entirely,
        # which is why 2026-09 web used to produce zero refunds despite its
        # raised share.
        eligible = []
        for candidate in BUSINESS_MONTHS:
            _, candidate_end = _month_bounds(candidate)
            if candidate_end > paid_at + REFUND_REQUIRED_DELAY:
                eligible.append(candidate)
        if not eligible:
            continue
        # most refunds land in the payment month itself; the tail drifts to the
        # remaining months of the window, which is what creates cross-month events.
        weights = {m: (0.55 if m == month else 1.0) for m in eligible}
        created: list[dict] = []
        any_success = False
        for _ in range(reversals):
            if remaining <= 0:
                break
            event_month = _weighted_key(rng, weights)
            _, month_end = _month_bounds(event_month)
            window_start = max(month_starts[event_month], paid_at + REFUND_REQUIRED_DELAY)
            if window_start >= month_end:
                continue
            span = max(1, int((month_end - window_start).total_seconds()))
            stamp = window_start + timedelta(seconds=rng.randrange(span))
            status = "succeeded"
            if rng.random() < 0.24:
                status = "failed" if rng.random() < 0.5 else "pending"
            if len(created) == reversals - 1:
                amount = remaining
            else:
                ceiling = remaining - 100 if remaining > 200 else 0
                amount = 0 if ceiling < 100 else max(100, min(int(remaining * rng.uniform(0.15, 0.55) // 100 * 100), ceiling))
            if amount <= 0:
                continue
            created.append({
                "refund_id": f"R{len(refunds) + 1:06d}", "order_id": order["order_id"],
                "refunded_at": _iso(stamp), "local_month": local_month(stamp),
                "refund_cents": amount, "status": status, "reason_code": rng.choice(REFUND_REASONS),
                "reversal_of": None,
                "settled_at": _iso(stamp + timedelta(days=3, hours=rng.randrange(0, 20))) if status == "succeeded" else None,
            })
            refunds.append(created[-1])
            if status == "succeeded":
                remaining -= amount
                any_success = True
        for index, row in enumerate(created):
            if index:
                row["reversal_of"] = created[index - 1]["refund_id"]
        if created and any_success and rng.random() < 0.18:
            recorded = month_starts[created[0]["local_month"]] + timedelta(days=rng.randrange(0, 20))
            legacy.append({
                "legacy_refund_id": f"LR{len(legacy) + 1:06d}", "order_id": order["order_id"],
                "recorded_at": _iso(recorded), "recorded_month": local_month(recorded),
                "amount_cents": created[0]["refund_cents"], "allocation_policy": "payment_month_allocation_v0",
                "status": "SUPERSEDED",
            })
    return refunds, legacy


def _build_tickets(rng: random.Random, ticket_count: int, links: list[dict]) -> list[dict]:
    tickets = []
    for index in range(1, ticket_count + 1):
        context, texts = TICKET_CONTEXTS[(index - 1) % len(TICKET_CONTEXTS)]
        linked = index % 2 == 1 and bool(links)
        link = links[((index - 1) // 2) % len(links)] if linked else None
        suffix = "" if not link else f"订单号 {link['order_id']}。"
        text = rng.choice(texts) + suffix
        if not link and rng.random() < 0.25:
            text += "该反馈没有可核实的交易编号。"
        stamp = _random_month_day(rng, BUSINESS_MONTHS[(index * 5) % len(BUSINESS_MONTHS)])
        tickets.append({
            "ticket_id": f"T{index:05d}", "text": text, "created_at": _iso(stamp),
            "order_id": link["order_id"] if link else None,
            "channel_code": link["channel_code"] if link else CHANNEL_CODES[(index * 3) % len(CHANNEL_CODES)],
            "context": context,
        })
    return tickets


def _build_ticket_links(rng: random.Random, paid_orders: list[dict], ticket_count: int) -> list[dict]:
    """One genuine ticket/order link per linked ticket; odd ticket ids are linked."""
    count = min(ticket_count // 2, len(paid_orders))
    reasons = ("refund_followup", "delivery_inquiry", "billing_question", "product_quality")
    links = []
    for index in range(count):
        order = paid_orders[(index * 7 + 3) % len(paid_orders)]
        stamp = datetime.fromisoformat(order["paid_at"].replace("Z", "+00:00")) + timedelta(days=rng.randrange(0, 30), hours=rng.randrange(1, 20))
        while local_month(stamp) not in BUSINESS_MONTHS:
            stamp -= timedelta(days=1)
        links.append({
            "link_id": f"L{index + 1:05d}", "ticket_id": f"T{index * 2 + 1:05d}", "order_id": order["order_id"],
            "local_month": local_month(stamp), "link_reason": reasons[index % len(reasons)],
            "confidence": round(rng.uniform(0.72, 0.99), 2), "linked_at": _iso(stamp),
            "channel_code": order["channel_code"],
        })
    return links


# --------------------------------------------------------------------------
# file writers
# --------------------------------------------------------------------------

#: Tables that may legitimately hold zero rows.  ``support_ticket_links`` is the
#: only one, and only when ``ticket_count = 0``: there is then no genuine
#: ticket/order pair to record, and inventing links would fabricate
#: relationships the source data does not contain.  Every other table must
#: always carry data, and with the default ``ticket_count = 3000`` this table is
#: nonempty too.
EMPTY_TOLERANT_TABLES = frozenset({"support_ticket_links"})


def _may_be_empty(table: str, ticket_count: int) -> bool:
    """Whether ``table`` is allowed to hold no rows for this configuration.

    Only ``support_ticket_links`` may be empty.  It holds one genuine link per
    *linked* ticket, and only odd-numbered ticket ids are linked, so the table is
    genuinely empty exactly when no ticket can be linked at all -- i.e.
    ``ticket_count < 2``.  For any larger ticket count the table must carry the
    real links it produced, and a silently-empty table is still a defect.
    """
    return table in EMPTY_TOLERANT_TABLES and ticket_count < 2


def _write_sqlite(path: Path, business: dict, ticket_count: int) -> None:
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA journal_mode=MEMORY")
        for table in TABLE_ORDER:
            columns = ", ".join(f'"{name}" {sql_type}' for name, sql_type in TABLE_SCHEMAS[table])
            db.execute(f'CREATE TABLE "{table}" ({columns})')
        for table in TABLE_ORDER:
            rows = business[table]
            # All 24 tables are always created.  A table may only be empty when
            # this run legitimately produced no rows for it.
            if not rows and not _may_be_empty(table, ticket_count):
                raise DatasetError(f"Table {table} would be created empty")
            if not rows:
                continue
            names = [name for name, _ in TABLE_SCHEMAS[table]]
            placeholders = ", ".join("?" for _ in names)
            quoted = ", ".join(f'"{name}"' for name in names)
            db.executemany(f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})',
                           [tuple(row[name] for name in names) for row in rows])
        for statement in INDEX_STATEMENTS:
            db.execute(statement)
        db.commit()
        db.execute("VACUUM")


def _write_channels(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel_code", "channel_name", "currency_code", "metric_timezone"])
        writer.writerows(CHANNELS)


def _write_tickets(path: Path, tickets: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for ticket in tickets:
            handle.write(json.dumps(ticket, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------

APPROVED_CASH_NET_DOC = """# CASH_NET_v1：批准的现金净额口径
状态：APPROVED；文档类型：metric_definition；版本：1

## 适用范围
本文件是唯一获批的业务指标口径。单位是人民币整数分（CNY_cents），业务时区是 Australia/Sydney。
数据库中的事件时间保存为带时区的 UTC 时间戳；汇总时按每个事件自身的时间戳转换到悉尼本地日历月。

## 计算规则
1. 支付事件取 orders 中 status = paid 的记录，金额字段 paid_cents，时间字段 paid_at。
2. 退款事件取 refunds 中 status = succeeded 的记录，金额字段 refund_cents，时间字段 refunded_at。
3. 某个本地月的现金净额 = 落入该月的成功支付金额合计 - 落入该月的成功退款金额合计。
4. 跨月的退款扣在退款成功所属的本地月，不回溯到支付月所属的月份。
5. 退款通过 order_id 关联原订单，渠道取原订单的 channel_code。
6. 退款与订单是一对多关系，先在退款粒度聚合再连接订单，避免订单金额被重复累计。

## 关联与数据源
注册来源是 business（SQLite）与 channels（CSV）。渠道映射必须唯一；缺失或重复映射时报错而不是静默丢弃。

## 边界
- 现金净额不等于会计确认收入，也不等于毛利。
- 变化率 =（本期 - 上期）/ 上期；上期为零时变化率不可定义，必须显式说明。
- 历史表（historical_orders、legacy_refunds）与派生列 local_month 只用于对账和查询下推，不能替代原始事件时间。
- 工单只能提供交易归属线索，不构成金额变化的因果证据。
"""

APPROVED_CASH_NET_TAIL = """## 变更记录
- 版本 1 在第二版数据环境首次发布，取代按支付月回溯扣减的旧口径。
- 本财年（2025-10 至 2026-09）内没有获批的替代版本。

## 常见误用
- 把退款按支付月回溯：会使已经关闭的月份数字发生变化，禁止。
- 先用订单表连接退款表再对订单金额求和：一对多 JOIN 会放大支付金额，禁止。
- 用本地日期字符串截断代替时区转换：在夏令时切换附近会产生错月，禁止。
"""


def _doc(table: str) -> str:
    columns = "\n".join(f"| {name} | {sql_type} | 见字段字典与业务含义说明 |" for name, sql_type in TABLE_SCHEMAS[table])
    return f"""# 字段字典：{table}
状态：REFERENCE；文档类型：table_dictionary

## 用途
本文件描述 {table} 表的字段、类型与常见误用。它只做参考，不定义任何批准指标。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
{columns}

## 使用约束
- 该表在 business（SQLite）来源中注册；本环境共 24 张表。
- 金额字段一律是 CNY_cents（人民币整数分），不要按元理解，也不要在聚合前四舍五入。
- orders.currency 是 ISO 4217 货币代码（CNY）；orders.amount_unit 是金额单位（CNY_cents）。
  两者不是同一概念，也不可互换：不要把金额单位当作货币写进 currency 列。
- order_items.line_total_cents = quantity * unit_price_cents - discount_cents；
  order_promotions.discount_cents 必须等于该订单各行 discount_cents 之和，不要重复扣减。
- 未成功订单的 paid_at 为 NULL、paid_cents 为 0，其 local_month 由 attempted_at 推导；
  attempted_at 只表示尝试时间，不是支付成功证据。
- attempted_at 是该订单自身的事件时间戳（已支付订单的 paid_at 与它相同），
  因此无论订单成功与否，其事件时间都落在覆盖的 12 个业务月之内。
- 时间字段是带时区的 UTC 时间戳；业务月需要显式转换到 IANA 时区 Australia/Sydney。
- 表内没有分析结论或预期答案，任何金额差异都要由下游查询独立计算。
"""


def _join_doc() -> str:
    return """# 连接粒度与基数（参考）
状态：REFERENCE；文档类型：join_rule

## 事实粒度
- orders：一行一笔订单。paid_at 仅对 status = paid 的订单有值；未成功订单为 NULL，paid_cents 为 0，
  其本地月由 attempted_at 推导（只描述尝试时间，不代表收款）。
- order_items：一行一个订单明细行，一个订单 1 至 3 行。
  line_total_cents = quantity * unit_price_cents - discount_cents，
  discount_cents 是该行分摊到的促销折扣整数分。
- payments：一行一笔支付记录；每笔订单恰好一条支付记录。
- refunds：一行一次退款事件；一个订单可以有零到多次退款事件。
- shipments：一行一次发货；仅成功支付的订单有记录。
- inventory_movements：sale_out 一行对应一个已支付订单的一条订单明细行，商品与数量一致、数量为负。

## 方向与基数
| 关系 | 基数 | 说明 |
| --- | --- | --- |
| order_items.order_id -> orders.order_id | 多对一 | 明细先行聚合再连接订单金额 |
| payments.order_id -> orders.order_id | 一对一 | 支付与订单金额一致，失败尝试 paid_at 为 NULL |
| refunds.order_id -> orders.order_id | 多对一 | 退款金额只在退款粒度求和 |
| orders.channel_code -> channels.channel_code | 多对一 | 渠道映射唯一，metric_timezone 是 IANA Australia/Sydney |
| order_promotions.order_id -> orders.order_id | 多对一 | 促销可能不覆盖全部订单；记录的是真实生效折扣 |
| inventory_movements.reference_id -> orders.order_id | 多对一 | 每个已发货订单的每条明细行一条 sale_out |

## 常见错误
- 把订单表与退款表直接连接后对 paid_cents 求和，会按退款次数重复累计。
- 认为每笔订单必有一次退款，或认为每笔退款必属于支付月。
- 把非支付订单的 attempted_at 当作支付时间，或据此把它计入现金净额。
- 用 order_items.discount_cents 与 order_promotions.discount_cents 重复扣减。
- 把 historical_orders、legacy_refunds 当作当前事实表。
"""


def _measure_doc() -> str:
    return """# 度量参考：支付、退款与净额（参考）
状态：REFERENCE；文档类型：metric_reference

本文件解释可用的度量构件，但不批准任何新指标。当前唯一获批定义见 CASH_NET_v1 的批准文档。

- paid_cents：订单实际收到并计入现金净额的整数分；只有 status = paid 的订单非零，其他状态一律为 0。
- order_total_cents：订单净额，等于该订单全部明细行 line_total_cents 之和（含促销折扣后的净额）。
- refund_cents：单次退款事件的整数分；状态为 succeeded 才计入。
- net_cents：成功支付减成功退款的整数分差额，必须声明所属本地月。
- 订单笔数：按 orders 行数计数；不要把 order_items 行数当作订单数。
- 金额单位是 CNY_cents（人民币整数分）；货币是 ISO 4217 代码 CNY，二者不是同一概念。
  orders.currency 存货币代码 CNY，orders.amount_unit 存金额单位 CNY_cents，列名与含义一一对应。

任何渠道或月份的对比都必须说明时间口径、状态过滤条件和关联粒度；否则数字不可比。
"""


def _refund_procedure() -> str:
    return f"""# 退款处理流程（参考）
状态：REFERENCE；文档类型：procedure

## 步骤
1. 客服根据工单描述定位订单；工单缺失订单号时先向用户索取，不做推测。
2. 在 refunds 中登记一条退款事件，包含 refund_id、order_id、refunded_at、refund_cents 与状态。
3. 状态取值：succeeded（已到账）、pending（处理中）、failed（失败）、cancelled（撤销）。
4. 成功退款累计不能超过该订单的 paid_cents；部分退款允许，多次退款允许。
5. 退款可能跨越支付月；事件时间只记录发生时的 UTC 时刻，不做月份回溯。

## 约束
- 退款时间不早于支付时间。
- 退款事件必须落在本数据集覆盖的 12 个业务月（{'、'.join(BUSINESS_MONTHS)}）内。
- 重复退款要保留 reversal_of 指向的上一条事件，便于识别重复申请。
- 失败的退款事件不能从历史中删除：它们同样是运营事实。

## 单证
退款必须以原订单渠道回退；渠道取自订单，不在退款表内单独记录。
"""


def _payment_procedure() -> str:
    return """# 支付状态处理流程（参考）
状态：REFERENCE；文档类型：procedure

## 状态机
- pending：已创建支付单，尚未得到收单结果，paid_at 为 NULL，金额为 0。
- paid：收单成功，写入 paid_at 与 paid_cents。
- failed：收单失败，paid_at 为 NULL，金额为 0，不进入现金净额。
- cancelled：用户或系统撤销，paid_at 为 NULL，金额为 0。

## 约束
- 每笔订单恰好一条 payments 记录，用于对账；金额与状态同订单一致。
- 只有 status = paid 的记录才写 paid_at；未成功尝试必须保持 paid_at 为 NULL，
  不得为了看起来完整而补上成功时间戳。
- 每次尝试的时间记录在 attempted_at（带时区 UTC），它表示尝试发生，不代表收款成功。
- 支付金额是非负整数分；不允许浮点元。
- 只有 paid 状态的订单才可能产生退款。
- 支付时间与创建时间使用带时区 UTC 时间戳，业务日按 Australia/Sydney 计算。
"""


def _channel_change() -> str:
    return """# 渠道变更记录（参考）
状态：REFERENCE；文档类型：change_log

## 2025-10 起的渠道结构
四个渠道：web（官网）、app（移动端）、marketplace（第三方平台）、partner（合作伙伴）。渠道映射唯一，缺失映射必须报错。

## 合同版本
channel_history 记录每个渠道的合同版本。第一版在 2025-09-30 结束，第二版自 2025-10-01 生效。
第二版只适用于生效日之后的事件；不得回溯第一版期间，也不得混合两个版本的佣金口径。

## 政策变化
运营在 2026-09 调高了官网退款申请的受理力度，并同步调整了客服排班。该变化只影响事件分布，
不改变支付与退款的状态定义，也不构成任何渠道业绩的因果结论。
"""


def _ticket_link_doc() -> str:
    return """# 工单与交易关联规则（参考）
状态：REFERENCE；文档类型：relationship_rule

## 关联来源
tickets（JSONL）保存 ticket_id、text、created_at、order_id 与 channel_code。
order_id 允许为空；为空表示该反馈没有可核实的交易归属，只能作为定性线索。

## 关联表
support_ticket_links 只保存真实存在的 ticket 与 order 组合，字段包含 link_reason、confidence 与 linked_at。
link_reason 说明关联依据（退款跟进、配送咨询、账务问题、商品质量），confidence 表示归属确认强度。

## 禁止的推断
- 工单数量不能当作退款笔数或退款金额。
- 工单文本不能证明金额变化的业务原因。
- 关联比例是数据生成设定，不代表客户满意度或业务责任归属。
"""


def _operations_policy() -> str:
    return """# 运营政策（参考）
状态：REFERENCE；文档类型：policy

## 履约
- 成功支付的订单在 0 至 3 天内发货，1 至 9 天内签收。
- 承运商与仓库按订单渠道与库存就近选择；运费按渠道区间取值。

## 库存
inventory_movements 记录库存变动，movement_type = sale_out 表示销售出库，数量为负。
每个已支付并已发货的订单，按其每一条订单明细行生成一条 sale_out：
product_id 是该明细行实际购买的商品，quantity 是该行数量的负数，
warehouse_id 是实际承担该订单发货的 shipments.warehouse_id。
未支付订单不产生 sale_out；库存变动通过 reference_id 追溯到订单，不得手工修改历史变动。

## 促销
促销窗口由 promotions 定义，order_promotions 记录订单**实际使用**的促销与折扣金额。
折扣必须真实生效：订单先确定适用促销，再把 discount_bps 折算成的整数分折扣
按明细行金额比例分摊到 order_items.discount_cents，
因此 line_total_cents = quantity * unit_price_cents - discount_cents，
order_promotions.discount_cents 等于该订单各行 discount_cents 之和，
order_total_cents 等于各行 line_total_cents 之和。
没有真实生效的促销不会出现在 order_promotions 中；折扣金额是非负整数分。

## 月度目标
sales_targets 按本地月与渠道给出目标净额与目标订单数；目标不是实际业绩，也不参与任何指标计算。
"""


def _timezone_doc() -> str:
    return """# 时区与业务月（参考）
状态：REFERENCE；文档类型：reference

## 事实
- 所有事件时间列保存为带时区 UTC 时间戳。
- 业务日历时区是 IANA 时区 Australia/Sydney（不是城市名 Sydney）。
- 货币是 ISO 4217 的 CNY（orders.currency / channels.csv 的 currency_code）；
  金额单位是 CNY_cents（orders.amount_unit）。二者必须分开表述，不能把单位当货币。
- orders.local_month 与 refunds.local_month 是按事件自身时间戳转换出的派生列，用于源端聚合降低行数。
- orders.order_date 是同一事件的悉尼本地业务日期，不参与金额计算。
- 已支付订单的归属月取自 paid_at；未成功订单没有 paid_at，其 local_month 改由 attempted_at 推导，
  这只描述尝试发生的时间，不构成任何支付成功证据。

## 使用规则
1. 跨月比较必须先确定月份来源：事件自身时间转换得到的本地月。
2. 不允许用 UTC 月份直接当作本地月。
3. 派生列只用于查询下推；独立复核时应重新由原始时间戳推导。
4. 不能把非支付订单的 attempted_at 当成支付时间使用。
"""


def _inventory_doc() -> str:
    return """# 库存与供应商参考
状态：REFERENCE；文档类型：reference

suppliers 保存供应商名称、国家、交期与账期；product_suppliers 保存商品的供货成本、最小起订量与主供应商标记。
warehouses 保存仓库城市、区域与容量。inventory_movements 记录出入库事件；
销售出库（sale_out）按订单明细行逐行生成，商品与数量同该明细行一致、数量为负，
仓库取该订单实际发货的 shipments.warehouse_id，reference_id 指向订单。

供货成本不等于售价，也不等于毛利；商品成本与售价都属于 CNY_cents（人民币整数分）。
供货关系可以是一对多：一个商品可以有多家供应商，但最多只有一家标记为 is_primary。
"""


def _historical_doc() -> str:
    return """# 历史口径：支付月回溯（STALE）
状态：STALE；文档类型：historical_metric

旧口径曾把后续退款回溯扣减到支付月，用于订单 cohort 的确认收入分析。
该口径已被 CASH_NET_v1 取代，不得用于当前现金净额。

historical_orders 与 legacy_refunds 保留了旧口径下的记录，字段中带有 recognition_policy 或 allocation_policy，
并统一标记 status = SUPERSEDED。这些表不能作为现行交易事实，也不能与 orders/refunds 混用求和。
"""


def _stale_channel_doc() -> str:
    return """# 历史渠道口径：单渠道世界（STALE）
状态：STALE；文档类型：historical_reference

第一版数据环境只有 web 与 app 两个渠道，且渠道佣金口径与现在不同。
本文件仅用于说明历史差异，任何基于两渠道的对比都不能套用到四渠道的当前环境。
"""


def _draft_metric_doc() -> str:
    return """# 草案指标：订单成功率（DRAFT）
状态：DRAFT；文档类型：draft_metric

草稿设想用成功支付订单数除以创建订单数描述转化，但分母定义（取消订单是否计入）尚未确认，
也没有经过任何审批，不得作为报告口径使用。
"""


def _draft_gross_doc() -> str:
    return """# 草案指标：毛收入（DRAFT）
状态：DRAFT；文档类型：draft_metric

草稿讨论把支付金额减去成本与运费得到毛收入，但成本口径（是否含仓储）未确认。
本文件不构成任何批准定义，也不提供任何金额参考。
"""


def _draft_semantic_doc() -> str:
    return """# 草案：工单语义标签（DRAFT）
状态：DRAFT；文档类型：draft_rule

草稿提议把工单文本归类为支付、配送、商品描述、退款时效与其他五类，用于定性观察量级变化。
标签规则尚未评审；即使评审通过，标签也只能解释现象，不能证明金额变化的因果关系。
"""


def _long_fx_doc() -> str:
    body = ["""# 结算与汇率长文（参考）
状态：REFERENCE；文档类型：reference_manual

本文件较长，用于检验长文档处理能力。前八十行主要是背景、术语与流程，
关键约束集中在文件后半部分。

## 背景
跨境结算按月归档，汇率取自 daily_fx_rates，基准货币 USD，报价货币 CNY。
汇率只用于折算展示，业务金额本身始终以整数分保存，不因汇率变化而改写历史金额。

## 术语
- 事件时间：业务事件发生的带时区时刻。
- 归档日：结算批次生成日期，只用于对账，不参与业务月份归属。
- 折算系数：报价货币 / 基准货币的比率，保留四位小数。

## 流程概览
1. 收集当月事件并校验状态过滤条件。
2. 按本地月归集，检查跨月事件是否落在覆盖区间内。
3. 生成对账批次，记录批次与来源行数。
4. 人工复核异常，不做自动修正。
"""]
    for index in range(1, 18):
        body.append(f"""## 附录 {index:02d}：批次核对清单
- 检查项 {index}.1：事件时间是否带时区，是否落在业务覆盖的 12 个月之内。
- 检查项 {index}.2：状态过滤是否一致，失败与待处理事件是否被排除或单独标注。
- 检查项 {index}.3：金额是否为整数分，是否出现浮点元导致的进位差异。
- 检查项 {index}.4：关联是否在正确粒度上完成，是否出现一对多连接放大金额。
- 检查项 {index}.5：渠道映射是否唯一，是否存在未登记渠道。
- 检查项 {index}.6：派生列与原始时间戳是否一致，派生列是否被误当作独立事实。
""")
    body.append("""## 关键约束（重要）
以下三条是本文件最重要的规则，位于文件后段：
1. 汇率折算不得改变业务事实：任何以折算后金额替代原始整数分金额的做法都不被接受。
2. 结算批次不得跨月混合：批次归属月份由事件时间决定，与批次生成时间无关。
3. 历史批次不得重算：一旦归档，后续更正必须新增批次并保留原批次，不允许覆盖。
""")
    return "\n".join(body)


def _long_definitions_doc() -> str:
    body = ["""# 指标定义长文（参考）
状态：REFERENCE；文档类型：reference_manual

本文件汇总度量定义的背景与边界，用于检验长文档的关键规则是否能被检索到。
开头部分只介绍背景，真正的约束在文件后半部分。

## 背景
指标定义描述从原始事件到可比较数字的转换规则。定义一旦批准，任何报告都必须使用同一版本。
不同版本之间的差异必须显式记录，不能靠记忆或口头约定。

## 通用结构
1. 事件集合：哪些记录参与计算。
2. 状态过滤：哪些状态是有效的。
3. 时间归属：事件落到哪个业务周期。
4. 关联方式：如何在正确的粒度上连接维度。
5. 单位与时区：金额单位、时区与舍入规则。
"""]
    for index in range(1, 16):
        body.append(f"""### 说明 {index:02d}
- 本节说明与批准口径无冲突，仅为背景材料：{index}。
- 示例描述不包含真实金额，也不构成任何分析结论。
- 与当前批准口径冲突的旧版本一律作废。
- 本节讨论的状态取值、时间字段与关联方向沿用现有表结构，不新增字段。
- 若下游需要更细的粒度，应先确认事实表粒度，再决定聚合顺序。
- 本节内容可以整体替换，不影响任何批准口径。
""")
    body.append("""## 关键规则（重要）
1. 时间归属只能由事件自身时间戳决定，不能由批次、上传时间或报告时间决定。
2. 金额比较只能在相同状态过滤条件下进行；混合 paid 与 pending 的数字没有意义。
3. 关联粒度错误会放大金额：一对多关系必须先聚合到事实粒度再连接。
4. 没有任何批准版本的指标不得进入报告；参考与草稿文件只提供背景。
5. 相关不等于因果：工单或事件分布的变化不能证明业务原因。
""")
    return "\n".join(body)


def _long_runbook_doc() -> str:
    body = ["""# 运营手册长文（参考）
状态：REFERENCE；文档类型：runbook

本手册描述日常对账与巡检流程。前半部分是排班与工具说明，关键约束在后半部分。

## 排班
- 工作日 09:00-18:00（Australia/Sydney）为常规处理时段。
- 月末最后两个工作日延长至 20:00，用于对账收尾。

## 工具
- 源数据只读访问，禁止直接修改业务表。
- 查询结果保留来源与过滤条件，便于复核。
"""]
    for index in range(1, 22):
        body.append(f"""### 巡检项 {index:02d}
- 观察对象：{index} 号巡检覆盖的渠道、仓库与工单队列。
- 记录方式：仅记录量级与异常描述，不记录推算结论。
- 处理方式：异常上报，不在源数据上做任何修补。
- 复核要求：核对原始时间戳与状态过滤条件，必要时重新推导派生列。
- 归档要求：保留巡检记录与来源，便于后续独立复核。
""")
    body.append("""## 关键约束（重要）
1. 源数据只读：不得修改、删除或回填任何业务记录。
2. 月度对比必须使用同一口径版本，跨版本比较需要显式声明并标注不可比。
3. 未支付与失败事件不进入现金净额，也不得为了"看起来完整"而补数。
4. 工单只作为定性线索，不得在报告中把工单数量与金额变化写成因果关系。
5. 派生列仅用于查询下推，复核必须回到原始时间戳。
""")
    return "\n".join(body)


def _generic_reference(name: str, status: str, kind: str, paragraphs: tuple[str, ...]) -> str:
    body = [f"# {name}", f"状态：{status}；文档类型：{kind}", ""]
    for index, paragraph in enumerate(paragraphs, start=1):
        body.append(f"## 说明 {index:02d}")
        body.append(paragraph)
        body.append("")
    body.append("## 边界")
    body.append("本文件不包含分析结论、示例查询语句或任何预先算出的金额，只描述业务背景与规则。")
    return "\n".join(body) + "\n"


def _document_specs() -> list[dict]:
    """Ordered document specs; only the first one carries the approved metric id.

    Order matters: a caller requesting the minimum of 40 documents still gets
    every table dictionary, the join rules, the metric definitions and the
    procedures, because truncation happens at the end of this list.
    """
    specs: list[dict] = [
        {"document_id": "cash_net_v1_approved", "status": "APPROVED", "metric_id": "CASH_NET_v1",
         "type": "metric_definition", "title": "CASH_NET_v1：批准的现金净额口径",
         "body": APPROVED_CASH_NET_DOC + APPROVED_CASH_NET_TAIL},
    ]
    specs += [{"document_id": f"table_dictionary_{table}", "status": "REFERENCE", "type": "table_dictionary",
               "title": f"字段字典：{table}", "body": _doc(table)} for table in TABLE_ORDER]
    specs += [
        {"document_id": "join_grain_and_cardinality", "status": "REFERENCE", "type": "join_rule",
         "title": "连接粒度与基数（参考）", "body": _join_doc()},
        {"document_id": "measure_reference", "status": "REFERENCE", "type": "metric_reference",
         "title": "度量参考：支付、退款与净额（参考）", "body": _measure_doc()},
        {"document_id": "refund_procedure", "status": "REFERENCE", "type": "procedure",
         "title": "退款处理流程（参考）", "body": _refund_procedure()},
        {"document_id": "payment_status_procedure", "status": "REFERENCE", "type": "procedure",
         "title": "支付状态处理流程（参考）", "body": _payment_procedure()},
        {"document_id": "channel_change_log", "status": "REFERENCE", "type": "change_log",
         "title": "渠道变更记录（参考）", "body": _channel_change()},
        {"document_id": "ticket_link_rules", "status": "REFERENCE", "type": "relationship_rule",
         "title": "工单与交易关联规则（参考）", "body": _ticket_link_doc()},
        {"document_id": "operations_policy", "status": "REFERENCE", "type": "policy",
         "title": "运营政策（参考）", "body": _operations_policy()},
        {"document_id": "timezone_and_business_month", "status": "REFERENCE", "type": "reference",
         "title": "时区与业务月（参考）", "body": _timezone_doc()},
        {"document_id": "inventory_and_supplier_reference", "status": "REFERENCE", "type": "reference",
         "title": "库存与供应商参考", "body": _inventory_doc()},
        {"document_id": "settlement_and_fx_manual", "status": "REFERENCE", "type": "reference_manual",
         "title": "结算与汇率长文（参考）", "body": _long_fx_doc()},
        {"document_id": "metric_definition_manual", "status": "REFERENCE", "type": "reference_manual",
         "title": "指标定义长文（参考）", "body": _long_definitions_doc()},
        {"document_id": "operations_runbook", "status": "REFERENCE", "type": "runbook",
         "title": "运营手册长文（参考）", "body": _long_runbook_doc()},
        {"document_id": "historical_payment_month_basis", "status": "STALE", "type": "historical_metric",
         "title": "历史口径：支付月回溯（STALE）", "body": _historical_doc()},
        {"document_id": "historical_two_channel_basis", "status": "STALE", "type": "historical_reference",
         "title": "历史渠道口径：单渠道世界（STALE）", "body": _stale_channel_doc()},
        {"document_id": "draft_order_success_rate", "status": "DRAFT", "type": "draft_metric",
         "title": "草案指标：订单成功率（DRAFT）", "body": _draft_metric_doc()},
        {"document_id": "draft_gross_revenue", "status": "DRAFT", "type": "draft_metric",
         "title": "草案指标：毛收入（DRAFT）", "body": _draft_gross_doc()},
        {"document_id": "draft_ticket_semantics", "status": "DRAFT", "type": "draft_rule",
         "title": "草案：工单语义标签（DRAFT）", "body": _draft_semantic_doc()},
    ]
    index = 0
    while len(specs) < 300:
        index += 1
        title, doc_type, body = _supplement_document(index)
        specs.append({"document_id": f"supplement_{index:03d}", "status": "REFERENCE",
                      "type": doc_type, "title": title, "body": body})
    return specs


def _supplement_document(index: int) -> tuple[str, str, str]:
    """Extra reference material so that larger document counts stay diverse."""
    variant = index % 6
    if variant == 0:
        table = TABLE_ORDER[index % len(TABLE_ORDER)]
        return f"字段应用说明：{table}（第 {index} 版）", "table_dictionary", _generic_reference(
            f"字段应用说明：{table}（第 {index} 版）", "REFERENCE", "table_dictionary",
            (f"{table} 的第 {index} 号应用说明，只描述字段用法，不引入新的口径。",
             "该表所有金额字段为整数分，时间字段为带时区 UTC 时间戳。",
             "跨表关联前先确认粒度，多对一关系需要先聚合事实侧。"))
    if variant == 1:
        code = CHANNEL_CODES[index % len(CHANNEL_CODES)]
        title = f"渠道运营参考：{code}（第 {index} 版）"
        return title, "channel_guide", _generic_reference(
            title, "REFERENCE", "channel_guide",
            (f"{code} 渠道的日常运营由渠道经理负责，跟进内容包含商品上架、促销排期与客服协同。",
             "渠道合同、佣金与结算周期以 channel_history 的当前版本为准，历史版本不得套用。",
             "渠道对比必须使用相同的时间口径与状态过滤；不同渠道的退款受理规则可能不同。"))
    if variant == 2:
        category = CATEGORY_DEFS[index % len(CATEGORY_DEFS)]
        title = f"商品类目指引：{category[1]}（第 {index} 版）"
        return title, "catalog_guide", _generic_reference(
            title, "REFERENCE", "catalog_guide",
            (f"类目 {category[0]}（{category[1]}）的商品有独立的定价区间与供货策略。",
             "类目结构存在父子层级；统计类目时必须明确层级，否则父子重复计数。",
             "商品上下架状态变化不追溯改写历史订单金额。"))
    if variant == 3:
        month = BUSINESS_MONTHS[index % len(BUSINESS_MONTHS)]
        title = f"月度结算备忘：{month}（第 {index} 版）"
        return title, "settlement_note", _generic_reference(
            title, "REFERENCE", "settlement_note",
            (f"{month} 的结算备忘只记录批次与来源行数，不记录任何汇总金额。",
             "批次归属月份由事件时间决定，与批次生成时间无关。",
             "发现异常时新增更正批次，不覆盖已归档批次。"))
    if variant == 4:
        context = TICKET_CONTEXTS[index % len(TICKET_CONTEXTS)][0]
        title = f"客服场景说明：{context}（第 {index} 版）"
        return title, "support_guide", _generic_reference(
            title, "REFERENCE", "support_guide",
            (f"{context} 场景的话术用于解释流程，不承诺具体到账时间，也不承诺金额认定结果。",
             "需要交易归属时必须向用户索取订单号，不允许从描述中推测订单。",
             "工单文本只作为定性线索，不能证明任何金额变化的因果关系。"))
    warehouse = WAREHOUSE_DEFS[index % len(WAREHOUSE_DEFS)]
    title = f"仓储履约参考：{warehouse[1]}（第 {index} 版）"
    return title, "operations_guide", _generic_reference(
        title, "REFERENCE", "operations_guide",
        (f"{warehouse[1]} 负责覆盖 {warehouse[3]} 区域的常规履约，容量为 {warehouse[4]} 件。",
         "出入库事件记录在 inventory_movements，通过 reference_id 追溯到订单。",
         "库存与运费属于成本信息，不得与支付金额或退款金额混合计算。"))


def _write_documents(folder: Path, document_count: int) -> list[dict]:
    docs_dir = folder / "documents"
    docs_dir.mkdir()
    documents = []
    for spec in _document_specs()[:document_count]:
        path = docs_dir / f"{spec['document_id']}.md"
        path.write_text(spec["body"], encoding="utf-8")
        entry = {"document_id": spec["document_id"], "path": f"documents/{spec['document_id']}.md",
                 "status": spec["status"], "title": spec["title"], "type": spec["type"],
                 "lines": len(spec["body"].splitlines())}
        if spec.get("metric_id"):
            entry["metric_id"] = spec["metric_id"]
        documents.append(entry)
    return documents


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def metric_contract() -> dict:
    """The approved CASH_NET_v1 contract, unchanged from the existing definition."""
    return {
        "metric_id": "CASH_NET_v1", "version": 1, "status": "APPROVED", "timezone": ZONE,
        "unit": "CNY_cents",
        "formula": "successful paid events minus successful refund events in each event's own local calendar month",
        "payment_status": "paid", "refund_status": "succeeded",
        "refund_channel": "original_order.channel_code",
        "event_time": {"orders": "paid_at", "refunds": "refunded_at"},
        "source_ids": ["business", "channels"], "missing_channel_policy": "reject",
        "approved_by": "synthetic-fixture-owner",
    }


def _build_registry(fixture, seed, counts, documents) -> dict:
    row_counts = {table: counts[table] for table in TABLE_ORDER}
    return {
        "fixture": fixture,
        "generator": GENERATOR_VERSION,
        "seed": seed,
        "config": {"order_count": counts["orders"], "ticket_count": counts["tickets"],
                   "document_count": counts["documents"], "business_months": list(BUSINESS_MONTHS),
                   "channels": list(CHANNEL_CODES), "timezone": ZONE,
                   "currency_code": CURRENCY_CODE, "amount_unit": AMOUNT_UNIT},
        "sources": [
            {"source_id": "business", "kind": "sqlite", "path": "business.sqlite",
             "tables": list(TABLE_ORDER), "row_counts": row_counts},
            {"source_id": "channels", "kind": "csv", "path": "channels.csv",
             "schema": [{"name": "channel_code", "type": "TEXT"}, {"name": "channel_name", "type": "TEXT"},
                        {"name": "currency_code", "type": "TEXT"}, {"name": "metric_timezone", "type": "TEXT"}],
             # currency_code is the ISO 4217 currency (CNY).  It is not the
             # amount unit: integer money columns are CNY_cents.
             "currency_code": CURRENCY_CODE, "amount_unit": AMOUNT_UNIT},
            {"source_id": "tickets", "kind": "jsonl", "path": "tickets.jsonl",
             "row_count": counts["tickets"],
             "schema": [{"name": "ticket_id", "type": "TEXT"}, {"name": "text", "type": "TEXT"},
                        {"name": "created_at", "type": "TEXT"}, {"name": "order_id", "type": "TEXT"},
                        {"name": "channel_code", "type": "TEXT"}, {"name": "context", "type": "TEXT"}]},
        ],
        "documents": documents,
        "metric_contracts": [metric_contract()],
        "manifest": {
            "generated_by": GENERATOR_VERSION,
            "deterministic": True,
            "wall_clock_used": False,
            "derived_columns": {
                "orders.attempted_at": (
                    "timezone-aware UTC instant at which the order was attempted; it is the order's own event "
                    "stamp and always lies inside the covered 12-month window. It records an attempt only and "
                    "is never evidence of successful payment."),
                "orders.local_month": (
                    "Australia/Sydney calendar month of the order's own event stamp: orders.paid_at for paid "
                    "orders, orders.attempted_at otherwise. Because a paid order's paid_at equals its "
                    "attempted_at, the chosen event is unambiguous for every row. For orders that never "
                    "succeeded there is no paid_at, so the month comes from orders.attempted_at instead. "
                    "The fallback is descriptive attribution for non-paid rows only and never substitutes "
                    "for a payment timestamp."),
                "orders.order_date": (
                    "Australia/Sydney business date of the same event timestamp used by orders.local_month "
                    "(paid_at for paid orders, attempted_at otherwise)"),
                "refunds.local_month": "local calendar month of refunds.refunded_at in Australia/Sydney",
            },
            "paid_at_semantics": (
                "paid_at is populated only for status='paid' orders and is NULL for every other order; "
                "paid_cents is zero for every non-paid order. Non-paid rows are never given a synthetic "
                "success timestamp."),
            "amount_unit": AMOUNT_UNIT,
            "currency_code": CURRENCY_CODE,
            "currency_columns": (
                "orders.currency holds the ISO 4217 currency code "
                f"({CURRENCY_CODE}); orders.amount_unit holds the unit the integer money columns are "
                f"counted in ({AMOUNT_UNIT}). They are different concepts and are published in "
                "separate columns; no unit is ever stored as a currency."),
            "row_counts": {**row_counts, "tickets": counts["tickets"], "documents": counts["documents"],
                           "channels": len(CHANNELS)},
            "notes": "Descriptive metadata only: row counts and parameters. No expected values or evaluator results.",
        },
    }


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def _temp_folder(source_root: Path, fixture: str) -> Path:
    """Private, freshly created temporary sister folder of the target.

    The name is derived only from the already-validated fixture identifier plus
    a fresh uuid4 fragment, so it can never collide with a fixture directory and
    never contains a path separator.
    """
    return source_root / f".{fixture}.tmp-{uuid.uuid4().hex[:12]}"


def _discard_temporary(temporary: Path, source_root: Path, fixture: str) -> None:
    """Delete one temporary folder, but only after proving what it is.

    Recursive deletion is only ever allowed for a path that the coordinator can
    still verify at cleanup time: it must resolve *directly* under the resolved
    ``source_root``, its resolved parent must equal that ``source_root``, and its
    name must match this run's private ``.<fixture>.tmp-<hex>`` shape.  Anything
    else is left untouched and reported instead of being deleted.
    """
    try:
        resolved_root = Path(source_root).resolve()
        resolved_temp = Path(temporary).resolve()
    except OSError:
        return
    expected_prefix = f".{fixture}.tmp-"
    if resolved_temp.parent != resolved_root:
        raise DatasetError(f"Refusing to remove {resolved_temp}: parent is not the resolved source_root")
    if not resolved_temp.name.startswith(expected_prefix):
        raise DatasetError(f"Refusing to remove {resolved_temp}: not this run's private temporary folder")
    if resolved_temp == resolved_root:
        raise DatasetError("Refusing to remove the source root itself")
    shutil.rmtree(resolved_temp, ignore_errors=True)


def generate_dataset(project_root: Path, fixture: str = DEFAULT_FIXTURE, seed: int = 73,
                     order_count: int = 50000, ticket_count: int = 3000,
                     document_count: int = 120) -> dict:
    """Generate one deterministic commerce fixture and return its registry dict.

    Reproducibility: the random stream is seeded from
    ``(seed, order_count, ticket_count, document_count)`` only.  The fixture
    identifier is a publication label, so the *same* size parameters under two
    different fixture names yield byte-identical logical business data.

    Publication: artifacts are written into a private sibling temporary folder
    and published with a single atomic ``os.replace``.  An existing target is
    never overwritten, and a failure removes only the temporary folder this call
    created after re-verifying that it still resolves directly under the
    resolved ``source_root``.
    """
    _validate_parameters(fixture, seed, order_count, ticket_count, document_count)
    source_root = Path(project_root).resolve() / "var" / "sources"
    source_root.mkdir(parents=True, exist_ok=True)
    folder = _fixture_folder(project_root, fixture)
    if folder.exists():
        raise DatasetError(f"Fixture already exists: {fixture}; choose another fixture identifier or remove it explicitly")
    temporary = _temp_folder(source_root, fixture)
    temporary.mkdir()
    try:
        counts = _generate_into(temporary, seed, order_count, ticket_count, document_count)
        registry = _build_registry(fixture, seed, counts, counts["documents_list"])
        (temporary / "registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, folder)
    except BaseException:
        _discard_temporary(temporary, source_root, fixture)
        raise
    return registry


def _generate_into(folder: Path, seed, order_count, ticket_count, document_count) -> dict:
    """Build every artifact inside ``folder``; no fixture name enters the data."""
    rng = random.Random(_rng_seed(seed, order_count, ticket_count, document_count))
    business, paid_orders = _build_business(rng, seed, order_count)
    links = _build_ticket_links(rng, paid_orders, ticket_count)
    tickets = _build_tickets(rng, ticket_count, links)
    business["support_ticket_links"] = [
        {"link_id": link["link_id"], "ticket_id": link["ticket_id"], "order_id": link["order_id"],
         "local_month": link["local_month"], "link_reason": link["link_reason"],
         "confidence": link["confidence"], "linked_at": link["linked_at"]} for link in links]
    _self_check(business, tickets, ticket_count)
    _write_sqlite(folder / "business.sqlite", business, ticket_count)
    _write_channels(folder / "channels.csv")
    _write_tickets(folder / "tickets.jsonl", tickets)
    documents = _write_documents(folder, document_count)
    counts = {table: len(business[table]) for table in TABLE_ORDER}
    counts |= {"orders": len(business["orders"]), "tickets": len(tickets), "documents": document_count,
               "documents_list": documents}
    return counts


def _self_check(business: dict, tickets: list[dict], ticket_count: int) -> None:
    """Host-side invariants; a violated contract must fail before publish, not after."""
    if set(business) != TABLE_NAMES:
        raise DatasetError("Generated table set does not match the 24 declared tables")
    for table in TABLE_ORDER:
        rows = business[table]
        # The same predicate guards the self check and the SQLite writer, so the
        # two can never disagree about which empty tables are acceptable.
        if not rows and not _may_be_empty(table, ticket_count):
            raise DatasetError(f"Table {table} would be created empty")
        if any(row.get(TABLE_SCHEMAS[table][0][0]) is None for row in rows):
            raise DatasetError(f"Table {table} contains a null primary identifier")
    orders = {row["order_id"]: row for row in business["orders"]}
    if len(orders) != len(business["orders"]):
        raise DatasetError("Duplicate order_id")
    # ---- order lines: net lines and per-line allocated promotion discount ----
    # Everything the later loops need about order lines is grouped here, in the
    # single mandatory pass over ``order_items``: per-order allocated discount,
    # per-order net total, per-order gross total, and the per-order line lists.
    # Building these once keeps the promotion and shipment loops O(1) per order
    # instead of rescanning every item row per recorded promotion/shipment.
    allocations: dict[str, int] = {}
    net_totals: dict[str, int] = {}
    gross_by_order: dict[str, int] = {}
    items_by_order: dict[str, list[dict]] = {}
    for item in business["order_items"]:
        if item["discount_cents"] < 0:
            raise DatasetError("Order item discount cannot be negative")
        gross = item["unit_price_cents"] * item["quantity"]
        if item["discount_cents"] > gross:
            raise DatasetError("Order item discount exceeds the line gross amount")
        if item["line_total_cents"] != gross - item["discount_cents"]:
            raise DatasetError("line_total_cents must equal quantity*unit_price_cents - discount_cents")
        if item["line_total_cents"] < 0:
            raise DatasetError("Order item line total cannot be negative")
        order_id = item["order_id"]
        items_by_order.setdefault(order_id, []).append(item)
        allocations[order_id] = allocations.get(order_id, 0) + item["discount_cents"]
        net_totals[order_id] = net_totals.get(order_id, 0) + item["line_total_cents"]
        gross_by_order[order_id] = gross_by_order.get(order_id, 0) + gross
    if set(net_totals) != set(orders):
        raise DatasetError("Every order must have at least one order item")
    # ---- orders: paid_at only for paid, totals consistent with net lines ----
    known_products = {row["product_id"] for row in business["products"]}
    for order_id, order in orders.items():
        if order["order_total_cents"] != net_totals[order_id]:
            raise DatasetError(f"order_total_cents must equal the net item lines for {order_id}")
        expected_paid = order["order_total_cents"] if order["status"] == "paid" else 0
        if order["paid_cents"] != expected_paid:
            raise DatasetError(f"paid_cents must be the order total for paid orders and zero otherwise ({order_id})")
        if order["status"] == "paid":
            if not order["paid_at"]:
                raise DatasetError(f"Paid order {order_id} must carry paid_at")
        elif order["paid_at"] is not None:
            raise DatasetError(f"Non-paid order {order_id} must keep paid_at NULL")
        if not order["attempted_at"] or _parse_stamp(order["attempted_at"]) is None:
            raise DatasetError(f"Order {order_id} must carry an aware attempted_at")
        attempted_stamp = _parse_stamp(order["attempted_at"])
        if attempted_stamp < _parse_stamp(order["created_at"]):
            raise DatasetError(f"Order {order_id} was attempted before it was created")
        # The attempt is the order's own event stamp, so it must itself be inside
        # the covered calendar; a month label alone never proves that.
        if local_month(attempted_stamp) not in BUSINESS_MONTHS:
            raise DatasetError(f"Order {order_id} was attempted outside the 12-month window")
        if order["local_month"] not in BUSINESS_MONTHS:
            raise DatasetError(f"Order {order_id} falls outside the 12-month window")
        if order["status"] == "paid" and attempted_stamp != _parse_stamp(order["paid_at"]):
            raise DatasetError(f"Paid order {order_id} must settle on its attempt stamp")
        if order["currency"] != CURRENCY_CODE:
            raise DatasetError(f"Order {order_id} must declare the ISO 4217 currency {CURRENCY_CODE}")
        if order["amount_unit"] != AMOUNT_UNIT:
            raise DatasetError(f"Order {order_id} must declare the amount unit {AMOUNT_UNIT}")
        if order["currency"] == order["amount_unit"]:
            raise DatasetError(f"Order {order_id} must not label its amount unit a currency")
        # local_month follows paid_at for paid orders and attempted_at otherwise.
        expected_month = _order_month(order["status"], _parse_stamp(order["paid_at"]),
                                      _parse_stamp(order["attempted_at"]))
        if order["local_month"] != expected_month:
            raise DatasetError(f"local_month does not match the order's own event timestamp ({order_id})")
    # ---- promotions: every recorded discount is genuinely allocated to lines ----
    promotion_ids = {row["promotion_id"] for row in business["promotions"]}
    recorded: dict[str, int] = {}
    for row in business["order_promotions"]:
        order = orders.get(row["order_id"])
        if order is None:
            raise DatasetError("order_promotions references an unknown order")
        if row["promotion_id"] not in promotion_ids:
            raise DatasetError("order_promotions references an unknown promotion")
        if row["discount_cents"] <= 0:
            raise DatasetError("A recorded promotion must carry a positive integer-cent discount")
        if row["order_id"] in recorded:
            raise DatasetError(f"An order may apply at most one promotion ({row['order_id']})")
        recorded[row["order_id"]] = row["discount_cents"]
    for order_id, discount in recorded.items():
        if allocations.get(order_id, 0) != discount:
            raise DatasetError(
                f"order_promotions.discount_cents must equal the sum of allocated item discounts ({order_id})")
    if len(business["order_promotions"]) != len(recorded):
        raise DatasetError("order_promotions must contain at most one row per order")
    # Which promotions are live is a property of the (month, channel) cell, so it
    # is resolved once per cell instead of once per order row.
    live_by_cell = {(month, channel): _live_promotions(month, channel)
                    for month in BUSINESS_MONTHS for channel in CHANNEL_CODES}
    eligible_orders = [
        order for order in business["orders"]
        if live_by_cell.get((order["local_month"], order["channel_code"]))
    ]
    if eligible_orders and not business["order_promotions"]:
        # The fixture must actually exercise the promotion path whenever any
        # order is covered by a live promotion, otherwise the "promotions affect
        # totals" contract would be untested by the data itself.
        raise DatasetError("Eligible orders exist but no promotion was genuinely applied")
    for row in business["order_promotions"]:
        order = orders[row["order_id"]]
        allowed = {promotion_id for promotion_id, _ in
                   live_by_cell[(order["local_month"], order["channel_code"])]}
        if row["promotion_id"] not in allowed:
            raise DatasetError(
                f"{row['promotion_id']} was applied outside its window or channel for {row['order_id']}")
        # The recorded discount must be exactly what discount_bps implies on the
        # order's gross line sum, so no free-floating discount can slip in.  The
        # gross sum comes from the single items pass above, never a rescan.
        gross = gross_by_order.get(row["order_id"], 0)
        bps = next(value for pid, _, _, value in PROMOTION_DEFS if pid == row["promotion_id"])
        if row["discount_cents"] != min(gross * bps // 10_000, gross):
            raise DatasetError(f"Recorded promotion discount does not match discount_bps for {row['order_id']}")
    # ---- payments: one per order, amounts and timestamps mirror the order ----
    payments = business["payments"]
    if len(payments) != len(orders) or {p["order_id"] for p in payments} != set(orders):
        raise DatasetError("payments must contain exactly one record per order")
    for payment in payments:
        order = orders[payment["order_id"]]
        if payment["amount_cents"] != order["paid_cents"] or payment["status"] != order["status"]:
            raise DatasetError(f"payment does not match its order for {payment['order_id']}")
        if payment["paid_at"] != order["paid_at"]:
            raise DatasetError(f"payment.paid_at must mirror its order for {payment['order_id']}")
        if not payment["paid_at"] and payment["amount_cents"] != 0:
            raise DatasetError(f"An unsuccessful payment attempt must settle zero ({payment['payment_id']})")
    # ---- inventory: one outflow per purchased line, at the shipment's warehouse ----
    shipment_warehouse: dict[str, str] = {}
    for shipment in business["shipments"]:
        if shipment["order_id"] not in orders:
            raise DatasetError("Shipment references an unknown order")
        if orders[shipment["order_id"]]["status"] != "paid":
            raise DatasetError("Only paid orders may be shipped")
        shipment_warehouse[shipment["order_id"]] = shipment["warehouse_id"]
    movements_by_order: dict[str, list[dict]] = {}
    for movement in business["inventory_movements"]:
        reference = movement["reference_id"]
        if reference not in orders:
            raise DatasetError("inventory_movements must be traceable to an order via reference_id")
        movements_by_order.setdefault(reference, []).append(movement)
    for order_id in shipment_warehouse:
        lines = items_by_order.get(order_id, [])
        movements = movements_by_order.get(order_id, [])
        if len(movements) != len(lines):
            raise DatasetError(f"Each purchased line needs exactly one outflow movement ({order_id})")
        expected = sorted((item["product_id"], -item["quantity"]) for item in lines)
        actual = sorted((movement["product_id"], movement["quantity"]) for movement in movements)
        if expected != actual:
            raise DatasetError(f"Inventory outflows must match the purchased lines exactly ({order_id})")
        warehouse = shipment_warehouse[order_id]
        if any(movement["warehouse_id"] != warehouse for movement in movements):
            raise DatasetError(f"Inventory outflows must use the shipment's warehouse ({order_id})")
        if any(movement["movement_type"] != "sale_out" or movement["quantity"] >= 0 for movement in movements):
            raise DatasetError(f"sale_out movements must be negative outflows ({order_id})")
    for order_id, movements in movements_by_order.items():
        if order_id not in shipment_warehouse:
            raise DatasetError("Only paid shipped orders may create sale_out movements")
    # Every paid order must be shipped; non-paid orders must never be.
    unshipped_paid = [oid for oid, order in orders.items()
                      if order["status"] == "paid" and oid not in shipment_warehouse]
    if unshipped_paid:
        raise DatasetError(f"Every paid order must have a shipment ({unshipped_paid[:5]})")
    if len(shipment_warehouse) != sum(1 for order in orders.values() if order["status"] == "paid"):
        raise DatasetError("Shipments must correspond one-to-one with paid orders")
    for movement in business["inventory_movements"]:
        if movement["product_id"] not in known_products:
            raise DatasetError("inventory_movements references an unknown product")
    # ---- refunds ----
    succeeded: dict[str, int] = {}
    for refund in business["refunds"]:
        order = orders.get(refund["order_id"])
        if order is None or order["status"] != "paid":
            raise DatasetError("Refunds may only reference paid orders")
        if refund["local_month"] not in BUSINESS_MONTHS:
            raise DatasetError("Refund event falls outside the 12-month window")
        if _parse_stamp(refund["refunded_at"]) < _parse_stamp(order["paid_at"]):
            raise DatasetError("Refund event precedes the payment event")
        if refund["local_month"] != local_month(_parse_stamp(refund["refunded_at"])):
            raise DatasetError("Refund local_month does not match its own timestamp")
        if refund["status"] == "succeeded":
            succeeded[refund["order_id"]] = succeeded.get(refund["order_id"], 0) + refund["refund_cents"]
    for order_id, total in succeeded.items():
        if total > orders[order_id]["paid_cents"]:
            raise DatasetError(f"Successful refunds exceed the paid amount for {order_id}")
    known = set(orders)
    known_tickets = {row["ticket_id"]: row for row in tickets}
    for link in business["support_ticket_links"]:
        if link["order_id"] not in known or link["ticket_id"] not in known_tickets:
            raise DatasetError("support_ticket_links contains an unknown ticket or order")
        if known_tickets[link["ticket_id"]]["order_id"] != link["order_id"]:
            raise DatasetError("support_ticket_links must only contain genuinely linked pairs")
    for row in tickets:
        if row["order_id"] is not None and row["order_id"] not in known:
            raise DatasetError("Ticket references an unknown order")


def _parse_stamp(value: str | None) -> datetime | None:
    """Parse an aware UTC timestamp; ``None`` passes through as ``None``."""
    if value is None:
        return None
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise DatasetError(f"Timestamp is not timezone-aware: {value}")
    return stamp


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m local_data_agent.dataset_v2",
                                     description="Generate a reproducible commerce source fixture.")
    parser.add_argument("--project-root", required=True, help="project root containing var/sources")
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE, help="fixture identifier (safe characters only)")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--orders", type=int, default=50000)
    parser.add_argument("--tickets", type=int, default=3000)
    parser.add_argument("--documents", type=int, default=120)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = generate_dataset(Path(args.project_root), args.fixture, args.seed,
                                    args.orders, args.tickets, args.documents)
    except DatasetError as exc:
        print(f"dataset-v2: {exc}", file=sys.stderr)
        return 2
    summary = {"fixture": registry["fixture"], "seed": registry["seed"],
               "target": str(_fixture_folder(Path(args.project_root), args.fixture)),
               "amount_unit": registry["config"]["amount_unit"],
               "currency_code": registry["config"]["currency_code"],
               "row_counts": registry["manifest"]["row_counts"], "documents": len(registry["documents"])}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
