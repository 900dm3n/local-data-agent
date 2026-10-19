"""Coordinator's independent source audit; evaluator output stays outside registry."""
import argparse
from collections import defaultdict
from contextlib import closing
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo


def audit(registry_path):
    registry_path = Path(registry_path).resolve()
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    folder = registry_path.parent
    def source_path(relative):
        p = (folder / relative).resolve()
        if not p.is_relative_to(folder):
            raise ValueError('Dataset source outside registered folder')
        return p
    sources = {s['source_id']: s for s in registry['sources']}
    checks = []
    def check(name, passed, details=None):
        checks.append({'name': name, 'passed': bool(passed), 'details': details})
    with closing(sqlite3.connect(source_path(sources['business']['path']).as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        names = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        counts = {name: db.execute('SELECT COUNT(*) FROM "' + name.replace('"', '""') + '"').fetchone()[0] for name in names}
        check('24_nonempty_tables', len(names) == 24 and all(counts.values()), counts)
        orders = {r['order_id']: dict(r) for r in db.execute('SELECT * FROM orders')}
        refunds = [dict(r) for r in db.execute('SELECT * FROM refunds')]
        check('unique_order_ids', len(orders) == counts['orders'])
        check('order_item_sums', db.execute('''SELECT COUNT(*) FROM orders o LEFT JOIN
            (SELECT order_id,SUM(line_total_cents) AS cents FROM order_items GROUP BY order_id) i
            ON o.order_id=i.order_id WHERE i.cents IS NULL OR i.cents<>o.order_total_cents
            OR (o.status='paid' AND o.paid_cents<>o.order_total_cents)
            OR (o.status<>'paid' AND o.paid_cents<>0)''').fetchone()[0] == 0)
        check('item_quantity_and_price', db.execute('''SELECT COUNT(*) FROM order_items
            WHERE quantity<1 OR unit_price_cents<0 OR discount_cents<0
            OR line_total_cents<>quantity*unit_price_cents-discount_cents OR line_total_cents<0''').fetchone()[0] == 0)
        for name, sql in {
            'orders_customer_fk': 'SELECT COUNT(*) FROM orders o LEFT JOIN customers c USING(customer_id) WHERE c.customer_id IS NULL',
            'items_order_fk': 'SELECT COUNT(*) FROM order_items i LEFT JOIN orders o USING(order_id) WHERE o.order_id IS NULL',
            'items_product_fk': 'SELECT COUNT(*) FROM order_items i LEFT JOIN products p USING(product_id) WHERE p.product_id IS NULL',
            'products_category_fk': 'SELECT COUNT(*) FROM products p LEFT JOIN categories c USING(category_id) WHERE c.category_id IS NULL',
            'shipments_order_fk': 'SELECT COUNT(*) FROM shipments s LEFT JOIN orders o USING(order_id) WHERE o.order_id IS NULL',
            'shipment_warehouse_fk': 'SELECT COUNT(*) FROM shipments s LEFT JOIN warehouses w USING(warehouse_id) WHERE w.warehouse_id IS NULL',
            'promotion_order_fk': 'SELECT COUNT(*) FROM order_promotions p LEFT JOIN orders o USING(order_id) WHERE o.order_id IS NULL',
            'promotion_fk': 'SELECT COUNT(*) FROM order_promotions op LEFT JOIN promotions p USING(promotion_id) WHERE p.promotion_id IS NULL',
            'promotion_allocated_discount': 'SELECT COUNT(*) FROM order_promotions op LEFT JOIN (SELECT order_id,SUM(discount_cents) AS cents FROM order_items GROUP BY order_id) i USING(order_id) WHERE op.discount_cents<>i.cents',
            'inventory_item_match': "SELECT COUNT(*) FROM (SELECT order_id,product_id,SUM(quantity) AS qty FROM order_items GROUP BY order_id,product_id) i JOIN orders o USING(order_id) LEFT JOIN (SELECT reference_id AS order_id,product_id,-SUM(quantity) AS qty FROM inventory_movements WHERE movement_type='sale_out' GROUP BY reference_id,product_id) m ON i.order_id=m.order_id AND i.product_id=m.product_id WHERE o.status='paid' AND (m.qty IS NULL OR m.qty<>i.qty)",
            'payment_fk_and_amount': 'SELECT COUNT(*) FROM payments p LEFT JOIN orders o USING(order_id) WHERE o.order_id IS NULL OR p.amount_cents<>o.paid_cents OR p.status<>o.status',
        }.items():
            count = db.execute(sql).fetchone()[0]
            check(name, count == 0, {'violations': count})
        check('one_payment_per_order', counts['payments'] == counts['orders'] and db.execute('SELECT COUNT(*) FROM (SELECT order_id FROM payments GROUP BY order_id HAVING COUNT(*)<>1)').fetchone()[0] == 0)
        zone = ZoneInfo('Australia/Sydney')
        def stamp(value):
            result = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if result.tzinfo is None:
                raise ValueError('Naive event timestamp')
            return result
        def month(value):
            return stamp(value).astimezone(zone).strftime('%Y-%m')
        months = {month(o['paid_at'] if o['status'] == 'paid' else o['attempted_at']) for o in orders.values()}
        check('12_business_months', months == {'2025-10','2025-11','2025-12','2026-01','2026-02','2026-03','2026-04','2026-05','2026-06','2026-07','2026-08','2026-09'}, sorted(months))
        check('derived_order_months', all(o['local_month'] == month(o['paid_at'] if o['status'] == 'paid' else o['attempted_at']) for o in orders.values()))
        check('nonpaid_events_not_fabricated', all(o['paid_at'] is None and o['paid_cents'] == 0 for o in orders.values() if o['status'] != 'paid'))
        check('derived_refund_months', all(r['local_month'] == month(r['refunded_at']) for r in refunds))
        check('refund_temporal_and_paid_fk', all(r['order_id'] in orders and orders[r['order_id']]['status'] == 'paid' and stamp(r['refunded_at']) >= stamp(orders[r['order_id']]['paid_at']) for r in refunds))
        refund_sums = defaultdict(int)
        ledger = defaultdict(lambda: {'paid_cents': 0, 'refund_cents': 0, 'net_cents': 0})
        for o in orders.values():
            if o['status'] == 'paid':
                ledger[(month(o['paid_at']), o['channel_code'])]['paid_cents'] += o['paid_cents']
        for r in refunds:
            if r['status'] == 'succeeded':
                refund_sums[r['order_id']] += r['refund_cents']
                ledger[(month(r['refunded_at']), orders[r['order_id']]['channel_code'])]['refund_cents'] += r['refund_cents']
        check('refund_amount_cap', all(v <= orders[k]['paid_cents'] for k,v in refund_sums.items()))
        check('cross_month_refunds_exist', any(month(r['refunded_at']) != month(orders[r['order_id']]['paid_at']) for r in refunds))
        check('repeat_refunds_exist', db.execute('SELECT COUNT(*) FROM (SELECT order_id FROM refunds GROUP BY order_id HAVING COUNT(*)>1)').fetchone()[0] > 0)
        links = {r['ticket_id']: r['order_id'] for r in db.execute('SELECT ticket_id,order_id FROM support_ticket_links')}
        tickets = [json.loads(line) for line in source_path(sources['tickets']['path']).read_text(encoding='utf-8').splitlines() if line.strip()]
        linked = [t for t in tickets if t.get('order_id')]
        check('ticket_links_match', len(links) == len(linked) and all(links.get(t['ticket_id']) == t['order_id'] and t['order_id'] in orders for t in linked))
        check('linked_and_unlinked_tickets', not tickets or 0 < len(linked) < len(tickets), {'tickets':len(tickets),'linked':len(linked)})
    docs = []
    for d in registry['documents']:
        content = source_path(d['path']).read_text(encoding='utf-8')
        docs.append({'document_id':d['document_id'], 'sha256':hashlib.sha256(content.encode()).hexdigest(), 'lines':len(content.splitlines()), 'status':d.get('status')})
    check('distinct_documents', len({d['sha256'] for d in docs}) == len(docs), {'count':len(docs)})
    check('long_documents_exist', any(d['lines'] > 120 for d in docs), [d['document_id'] for d in docs if d['lines']>120])
    check('only_current_cash_metric_approved', len([d for d in registry['documents'] if d.get('metric_id') and d.get('status') == 'APPROVED']) == 1 and len(registry['metric_contracts']) == 1)
    check('gold_not_registered', not any(word in registry_path.read_text(encoding='utf-8').lower() for word in ['gold', 'reference_sql', 'monthly_cents']))
    for amounts in ledger.values():
        amounts['net_cents'] = amounts['paid_cents'] - amounts['refund_cents']
    return {'scope':'Coordinator independent source audit, not autonomous model acceptance. Gold stays evaluator-only.',
        'passed':all(c['passed'] for c in checks), 'checks':checks, 'table_counts':counts,
        'ledger': [dict(month=m,channel_code=c,**v) for (m,c),v in sorted(ledger.items())], 'documents':docs}


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.registry)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':result['passed'],'checks':len(result['checks']),'output':str(args.output)},ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)
