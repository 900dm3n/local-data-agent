"""Synthetic sources; the independent evaluator is deliberately not in registry."""
from __future__ import annotations

import csv
from contextlib import closing
import json
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ZONE = "Australia/Sydney"


def _utc(day: str) -> str:
    return datetime.fromisoformat(day + "T12:00:00").replace(tzinfo=ZoneInfo(ZONE)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def bootstrap(project_root: Path, fixture="cash-net-v1", seed=0, variant=None) -> dict:
    if not fixture or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in fixture):
        raise ValueError("Invalid fixture identifier")
    folder = Path(project_root).resolve() / "var" / "sources" / fixture
    if folder.exists():
        raise ValueError(f"Fixture already exists: {fixture}; choose another fixture identifier")
    allowed = {None, "held-out", "duplicate-channels", "orphan-refund", "no-metric", "timezone-boundary", "zero-denominator", "new-refund", "null-amount", "negative-amount", "duplicate-order", "failed-events", "unknown-channel"}
    if variant not in allowed:
        raise ValueError("Unknown fixture variant")
    folder.mkdir(parents=True)
    orders = [[f"O{i + 1}", _utc(day), channel, cents, "paid"] for i, (day, channel, cents) in enumerate([
        ("2026-08-05", "web", 10000), ("2026-08-20", "app", 20000),
        ("2026-08-28", "web", 10000), ("2026-09-05", "web", 12000),
        ("2026-09-15", "app", 18000), ("2026-09-28", "web", 10000)])]
    refunds = [[rid, oid, _utc(day), cents, "succeeded"] for rid, oid, day, cents in [
        ("R1", "O2", "2026-08-25", 2000), ("R2", "O4", "2026-09-10", 3000),
        ("R3", "O4", "2026-09-12", 1000), ("R4", "O3", "2026-09-06", 5000)]]
    if variant == "held-out" or seed:
        rng = random.Random(seed)
        for row in orders:
            row[3] = rng.randint(50, 250) * 100
        for row in refunds:
            row[3] = rng.randint(1, 20) * 100
    if variant == "new-refund":
        refunds.append(["R5", "O6", _utc("2026-09-29"), 1000, "succeeded"])
    if variant == "orphan-refund":
        refunds[3][1] = "O-missing"
    if variant == "timezone-boundary":
        orders.extend([["O7", "2026-08-31T14:00:00Z", "web", 700, "paid"],
                       ["O8", "2026-09-30T14:00:00Z", "app", 900, "paid"],
                       ["O9", "2026-10-31T13:00:00Z", "app", 1100, "paid"]])
    if variant == "zero-denominator":
        orders = [r for r in orders if not r[1].startswith("2026-08")]
        refunds = [r for r in refunds if r[1] in {o[0] for o in orders}]
    if variant == "null-amount":
        orders[0][3] = None
    if variant == "negative-amount":
        orders[0][3] = -100
    if variant == "duplicate-order":
        orders.append(list(orders[0]))
    if variant == "failed-events":
        orders.append(["O10", _utc("2026-09-15"), "web", 90000, "failed"])
        refunds.append(["R10", "O4", _utc("2026-09-16"), 70000, "failed"])
    if variant == "unknown-channel":
        orders[0][2] = "unmapped"
    business = folder / "business.sqlite"
    with closing(sqlite3.connect(business)) as db:
        db.executescript("CREATE TABLE orders(order_id TEXT,paid_at TEXT,channel_code TEXT,paid_cents INTEGER,status TEXT); CREATE TABLE refunds(refund_id TEXT,order_id TEXT,refunded_at TEXT,refund_cents INTEGER,status TEXT); CREATE TABLE marketing_clicks(campaign TEXT,clicked_at TEXT,click_count INTEGER); CREATE TABLE historical_orders(order_id TEXT,recognized_at TEXT,revenue_cents INTEGER);")
        db.executemany("INSERT INTO orders VALUES(?,?,?,?,?)", orders)
        db.executemany("INSERT INTO refunds VALUES(?,?,?,?,?)", refunds)
        db.execute("INSERT INTO marketing_clicks VALUES('summer','2026-09-01',123)")
        db.commit()
    channels = folder / "channels.csv"
    with channels.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["channel_code", "channel_name"])
        writer.writerows([("web", "官网"), ("app", "移动端")])
        if variant == "duplicate-channels":
            writer.writerow(["web", "重复官网"])
    tickets = folder / "tickets.jsonl"
    texts = ["用户反馈页面支付按钮偶尔无响应，未提供订单编号。", "客服收到退款到账时间咨询，未提供退款金额。", "用户表示更喜欢移动端下单，没有关联订单。", "官网用户反馈商品描述不够清楚，没有可核实的交易关联。", "客服说明到账延迟可能涉及银行处理，需要进一步核查。"]
    tickets.write_text("".join(json.dumps({"ticket_id": f"T{i+1}", "text": t}, ensure_ascii=False) + "\n" for i, t in enumerate(texts)), encoding="utf-8")
    contract = {"metric_id": "CASH_NET_v1", "version": 1, "status": "APPROVED", "timezone": ZONE,
                "unit": "CNY_cents", "formula": "successful paid events minus successful refund events in each event's own local calendar month", "payment_status": "paid", "refund_status": "succeeded", "refund_channel": "original_order.channel_code", "event_time": {"orders": "paid_at", "refunds": "refunded_at"}, "source_ids": ["business", "channels"], "missing_channel_policy": "reject", "approved_by": "synthetic-fixture-owner"}
    approved = folder / "cash_net.md"
    approved.write_text("# CASH_NET_v1：批准的现金净额口径\n状态：APPROVED；版本：1\n金额单位：CNY 整数分；展示可以除以100为元。\n时区：Australia/Sydney；数据库时间戳为UTC，按本地日历月汇总。\n支付只统计 status=paid 的 paid_cents，使用 paid_at。\n退款只统计 status=succeeded 的 refund_cents，使用 refunded_at。\n跨月退款扣在退款成功月，不回溯到支付月。\n退款通过 order_id 关联原订单取 channel_code；不能将订单支付金额在退款一对多 JOIN 后重复累计。\n渠道映射唯一，缺失或重复映射必须报错。\n现金净额不等同会计确认收入。\n变化率=(本期-上期)/上期；上期为零时不可定义。\n工单没有订单关联时仅为定性线索，不能证明金额变化的业务原因。\n", encoding="utf-8")
    historical = folder / "historical_metric.md"
    historical.write_text("# 历史收入定义：订单 cohort\n状态：STALE；不适用于 CASH_NET_v1。\n曾按支付月回溯扣除后续退款。这个旧定义不能替代批准的按事件月现金净额定义。\n", encoding="utf-8")
    registry = {"fixture": fixture, "seed": seed, "variant": variant,
                "sources": [{"source_id": "business", "kind": "sqlite", "path": str(business), "tables": ["orders", "refunds", "marketing_clicks", "historical_orders"]}, {"source_id": "channels", "kind": "csv", "path": str(channels)}, {"source_id": "tickets", "kind": "jsonl", "path": str(tickets)}],
                "documents": [{"document_id": "historical_metric", "path": str(historical), "status": "STALE"}], "metric_contracts": []}
    if variant != "no-metric":
        registry["documents"].insert(0, {"document_id": "cash_net", "path": str(approved), "status": "APPROVED", "metric_id": "CASH_NET_v1"})
        registry["metric_contracts"] = [contract]
    else:
        approved.unlink()
    (folder / "registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return registry


def compute_gold(registry: dict) -> dict:
    """Evaluator-only independent event ledger; never include this in a tool result."""
    source = next(s for s in registry["sources"] if s["source_id"] == "business")
    with closing(sqlite3.connect(Path(source["path"]).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        orders = [dict(r) for r in db.execute("SELECT * FROM orders")]
        refunds = [dict(r) for r in db.execute("SELECT * FROM refunds")]
    ids = [r["order_id"] for r in orders]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate order IDs invalidate gold")
    lookup = {r["order_id"]: r for r in orders}
    totals, channels, payments, refunded = {}, {}, {}, {}
    def month(stamp):
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(ZoneInfo(ZONE)).strftime("%Y-%m")
    for rows, sign, amount, timestamp, status in [(orders, 1, "paid_cents", "paid_at", "paid"), (refunds, -1, "refund_cents", "refunded_at", "succeeded")]:
        for row in rows:
            if row["status"] != status:
                continue
            cents = row[amount]
            if isinstance(cents, bool) or not isinstance(cents, int) or cents < 0:
                raise ValueError("Invalid money in gold input")
            original = row if sign == 1 else lookup.get(row["order_id"])
            if original is None:
                raise ValueError("Orphan refund invalidates gold")
            key, channel = month(row[timestamp]), original["channel_code"]
            totals[key] = totals.get(key, 0) + sign * cents
            channels[(key, channel)] = channels.get((key, channel), 0) + sign * cents
            series = payments if sign == 1 else refunded
            series[key] = series.get(key, 0) + cents
    aug, sep = totals.get("2026-08", 0), totals.get("2026-09", 0)
    return {"monthly_cents": totals, "monthly_channel_cents": [{"month": m, "channel_code": c, "net_cents": n} for (m, c), n in sorted(channels.items())], "payment_cents": payments, "refund_cents": refunded, "delta_cents": sep-aug, "change_rate": (sep-aug)/aug if aug else None, "metric_id": "CASH_NET_v1"}
