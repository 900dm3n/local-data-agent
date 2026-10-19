"""Pure deterministic compiler for host-built financial table narratives.

The model never writes financial direction prose. This module reads already
verified table rows (plain dicts, e.g. ``pq.read_table(path).to_pylist()``),
accumulates monthly and comparison facts **across every selected artifact** and
derives a bounded qualitative summary plus explicit comparison records.

Contract enforced here, with no silent repair:

* Money is exact integer cents; booleans and floats are rejected as money.
* Periods are real ``YYYY-MM`` strings.
* Rows accumulate by ``(channel_code, month)`` and
  ``(channel_code, base_month, current_month)`` -- never by artifact id -- so the
  same fact appearing in two tables merges, and a genuine disagreement rejects.
* A monthly row requires ``net_cents``; ``paid_cents``/``refund_cents`` are
  optional but all-or-nothing, nonnegative, and must satisfy
  ``net_cents == paid_cents - refund_cents``.
* Comparison rows must satisfy ``delta_cents == current_net_cents - base_net_cents``
  and, when present, ``change_pct`` must be a finite number consistent with the
  exact ratio (``None`` whenever the base is zero). Direction always comes from
  ``delta_cents``, never from the sign of a percentage, because a negative base
  inverts the percentage sign.
* A derived comparison is built only from a monthly group with exactly two
  distinct months; three or more months never invent one pair.
* The final ``groups`` list contains comparisons only.
* No whole-company total is inferred from a partial set of channel rows.

The result is intentionally free of dynamic numbers, dates and channel names, so
the publication layer can reject any unbound numeric text.
"""

from __future__ import annotations

import math

MONTH_PATTERN = r"[0-9]{4}-(0[1-9]|1[0-2])"

# ``change_pct`` is stored as percent points; the tolerance matches the
# publication-time source reconciliation tolerance.
_PCT_TOLERANCE = 0.005001
_MAX_ROWS = 20
_MAX_TABLES = 16

_LIMITATION = "本说明仅描述已核对的所选金额表，业务原因尚未证明。"

_MONTHLY_FIELDS = {"month", "net_cents", "paid_cents", "refund_cents", "channel_code",
                   "channel_name", "payment_count", "refund_count"}
_COMPARISON_FIELDS = {
    "base_month", "current_month", "base_net_cents", "current_net_cents", "delta_cents", "change_pct",
    "base_paid_cents", "current_paid_cents", "base_refund_cents", "current_refund_cents", "channel_code",
}
# Presentational columns the host ignores instead of rejecting the row.
_BENIGN = {"channel_name", "payment_count", "refund_count"}


class _Reject(ValueError):
    """Raised for any shape or value the compiler refuses to guess about."""


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _month(value, field):
    if not isinstance(value, str) or len(value) != 7 or value[4] != "-":
        raise _Reject(f"{field} must be a real YYYY-MM calendar month")
    year, month = value[:4], value[5:]
    if not (year.isascii() and year.isdigit() and month.isascii() and month.isdigit()):
        raise _Reject(f"{field} must be a real YYYY-MM calendar month")
    if int(year) == 0 or not 1 <= int(month) <= 12:
        raise _Reject(f"{field} must be a real YYYY-MM calendar month")
    return value


def _channel(value):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _Reject("channel_code must be a nonempty string or absent")
    return value


def _provided(row, field):
    """A field counts as provided only when explicitly present and not None."""
    return field in row and row[field] is not None


def _monthly(row, artifact_id):
    unknown = set(row) - _MONTHLY_FIELDS
    if unknown:
        raise _Reject(f"Unsupported monthly row fields: {sorted(unknown - _BENIGN)}")
    net = row["net_cents"]
    if not _is_int(net):
        raise _Reject("net_cents must be an exact integer number of cents")
    parts = _parts(row, "paid_cents", "refund_cents")
    if parts is not None and parts[0] - parts[1] != net:
        raise _Reject("net_cents must equal paid_cents minus refund_cents")
    return {"month": _month(row["month"], "month"), "channel_code": _channel(row.get("channel_code")),
            "net_cents": net, "parts": parts, "source_artifact_ids": [artifact_id]}


def _parts(row, paid_field, refund_field):
    paid, refund = paid_field in row, refund_field in row
    if paid != refund:
        raise _Reject(f"{paid_field} and {refund_field} must both be present or both absent")
    if not paid:
        return None
    values = (row[paid_field], row[refund_field])
    if not all(_is_int(value) for value in values) or any(value < 0 for value in values):
        raise _Reject(f"{paid_field} and {refund_field} must be nonnegative integer cents")
    return values


def _comparison(row, artifact_id):
    unknown = set(row) - _COMPARISON_FIELDS
    if unknown:
        raise _Reject(f"Unsupported comparison row fields: {sorted(unknown)}")
    base, current = _month(row["base_month"], "base_month"), _month(row["current_month"], "current_month")
    if base == current:
        raise _Reject("Comparison rows require two distinct months")
    base_net, current_net, delta = row["base_net_cents"], row["current_net_cents"], row["delta_cents"]
    if not all(_is_int(value) for value in (base_net, current_net, delta)):
        raise _Reject("Comparison amounts must be exact integer cents")
    if delta != current_net - base_net:
        raise _Reject("delta_cents must equal current_net_cents minus base_net_cents")
    expected = _expected_pct(base_net, current_net)
    if "change_pct" in row:
        value = row["change_pct"]
        if value is None:
            if expected is not None:
                raise _Reject("change_pct cannot be null for a nonzero base")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise _Reject("change_pct must be a finite number or null")
        elif expected is None:
            raise _Reject("change_pct must be absent or null when the base is zero")
        elif abs(float(value) - expected) > _PCT_TOLERANCE:
            raise _Reject("change_pct disagrees with the exact base/current ratio")
    base_parts = _parts(row, "base_paid_cents", "base_refund_cents")
    current_parts = _parts(row, "current_paid_cents", "current_refund_cents")
    if (base_parts is None) != (current_parts is None):
        raise _Reject("Comparison components require both periods")
    if base_parts is not None and (base_parts[0] - base_parts[1] != base_net
                                  or current_parts[0] - current_parts[1] != current_net):
        raise _Reject("Comparison components disagree with period net amounts")
    return {"base_month": base, "current_month": current, "channel_code": _channel(row.get("channel_code")),
            "base_net_cents": base_net, "current_net_cents": current_net, "delta_cents": delta,
            "change_pct": expected, "parts": None if base_parts is None else (base_parts, current_parts),
            "source_artifact_ids": [artifact_id]}


def _expected_pct(base, current):
    return None if base == 0 else (current - base) / base * 100


def _merge(store, key, item, kind):
    """Merge compatible duplicates across artifacts; conflicting shared values reject."""
    existing = store.get(key)
    if existing is None:
        item["source_artifact_ids"] = sorted(set(item["source_artifact_ids"]))
        store[key] = item
        return
    for field in ("net_cents", "base_net_cents", "current_net_cents", "delta_cents", "change_pct"):
        if field in item and existing.get(field) != item[field]:
            raise _Reject(f"Conflicting {kind} facts for {key}: {field} differs across selected tables")
    # Missing monthly parts may be supplemented by another row for the same fact.
    if item["parts"] is not None:
        if existing["parts"] is None:
            existing["parts"] = item["parts"]
        elif existing["parts"] != item["parts"]:
            raise _Reject(f"Conflicting {kind} facts for {key}: payment/refund parts differ")
    existing["source_artifact_ids"] = sorted(set(existing["source_artifact_ids"]) | set(item["source_artifact_ids"]))


def compile_table_narrative(tables):
    """Derive ``{summary, groups, source_artifact_ids}`` from verified table rows.

    Returns document keys ordered ``summary``, ``groups``, ``source_artifact_ids``;
    each group key is ``channel_code, base_month, current_month, before, after,
    delta, change_pct, net_direction, source_artifact_ids`` plus the optional
    ``monthly_paid_cents``/``monthly_refund_cents`` parts.
    """
    if not isinstance(tables, list) or not tables:
        raise _Reject("At least one verified table is required")
    if len(tables) > _MAX_TABLES:
        raise _Reject("Too many tables for narrative compilation")

    monthly, comparisons, artifacts = {}, {}, []
    for table in tables:
        if not isinstance(table, dict) or set(table) != {"artifact_id", "rows"}:
            raise _Reject("Each table must provide artifact_id and rows")
        artifact_id = table["artifact_id"]
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise _Reject("artifact_id must be a nonempty string")
        if artifact_id in artifacts:
            raise _Reject("Duplicate artifact selection")
        rows = table["rows"]
        if not isinstance(rows, list) or not 1 <= len(rows) <= _MAX_ROWS:
            raise _Reject("Each table must carry between one and twenty verified rows")
        artifacts.append(artifact_id)
        for row in rows:
            if not isinstance(row, dict):
                raise _Reject("Each row must be an object")
            monthly_shape = {"month", "net_cents"}.issubset(row)
            comparison_shape = {"base_month", "current_month", "base_net_cents",
                                "current_net_cents", "delta_cents"}.issubset(row)
            if monthly_shape and comparison_shape:
                raise _Reject("A row cannot mix monthly and comparison shapes")
            if monthly_shape:
                item = _monthly(row, artifact_id)
                _merge(monthly, (item["channel_code"], item["month"]), item, "monthly")
            elif comparison_shape:
                item = _comparison(row, artifact_id)
                _merge(comparisons, (item["channel_code"], item["base_month"], item["current_month"]),
                       item, "comparison")
            else:
                raise _Reject("Row is neither a recognized monthly nor comparison row")

    # A monthly group with exactly two distinct months yields one derived comparison.
    for channel, months in _by_channel(monthly).items():
        if len(months) != 2:
            continue
        base, current = sorted(months)
        older, newer = months[base], months[current]
        key = (channel, base, current)
        derived = {"base_month": base, "current_month": current, "channel_code": channel,
                   "base_net_cents": older["net_cents"], "current_net_cents": newer["net_cents"],
                   "delta_cents": newer["net_cents"] - older["net_cents"],
                   "change_pct": _expected_pct(older["net_cents"], newer["net_cents"]),
                   "parts": (older["parts"], newer["parts"]) if older["parts"] is not None and newer["parts"] is not None else None,
                   "source_artifact_ids": sorted(set(older["source_artifact_ids"]) | set(newer["source_artifact_ids"]))}
        _merge(comparisons, key, derived, "derived comparison")

    # Explicit comparisons must agree with any overlapping monthly points,
    # including scopes with more than two months (which infer no new pair).
    for (channel, base, current), point in comparisons.items():
        for month, field in ((base, "base_net_cents"), (current, "current_net_cents")):
            monthly_point = monthly.get((channel, month))
            if monthly_point is not None:
                if monthly_point["net_cents"] != point[field]:
                    raise _Reject("Comparison disagrees with an overlapping monthly fact")
                point["source_artifact_ids"] = sorted(set(point["source_artifact_ids"]) | set(monthly_point["source_artifact_ids"]))

    groups = []
    for key in sorted(comparisons, key=lambda item: (item[0] is not None, str(item[0]), item[1], item[2])):
        point = comparisons[key]
        group = {"channel_code": point["channel_code"], "base_month": point["base_month"],
                 "current_month": point["current_month"], "base_net_cents": point["base_net_cents"],
                 "current_net_cents": point["current_net_cents"], "delta_cents": point["delta_cents"],
                 "change_pct": point["change_pct"], "net_direction": _direction(point["delta_cents"]),
                 "source_artifact_ids": list(point["source_artifact_ids"])}
        if point["parts"] is not None:
            base_parts, current_parts = point["parts"]
            group.update(base_paid_cents=base_parts[0], base_refund_cents=base_parts[1],
                         current_paid_cents=current_parts[0], current_refund_cents=current_parts[1])
        groups.append(group)

    return {"summary": _summary(monthly, comparisons, bool(groups)),
            "groups": groups, "source_artifact_ids": list(artifacts)}


def _by_channel(store):
    scopes = {}
    for (channel, month), point in store.items():
        scopes.setdefault(channel, {})[month] = point
    return scopes


def _direction(delta):
    return "up" if delta > 0 else "down" if delta < 0 else "flat"


def _phrase(direction):
    return "上升" if direction == "up" else "下降" if direction == "down" else "不变"


def _summary(monthly, comparisons, has_groups):
    """Qualitative wording only: no digits, no month names, no channel names."""
    parts = []
    overall = [point for (channel, _base, _current), point in comparisons.items() if channel is None]
    selected = [point for (channel, _base, _current), point in comparisons.items() if channel is not None]
    if overall:
        directions = {_direction(point["delta_cents"]) for point in overall}
        parts.append("所选全体汇总的净额" + _phrase(next(iter(directions))) if len(directions) == 1
                     else "所选全体汇总在各比较期间的净额方向不一致")
        if len(overall) == 1 and overall[0]["parts"] is not None:
            base_parts, current_parts = overall[0]["parts"]
            parts.append("所选全体支付金额" + _phrase(_direction(current_parts[0] - base_parts[0])))
            parts.append("所选全体退款金额" + _phrase(_direction(current_parts[1] - base_parts[1])))
    # Selected channels are aggregated collectively; partial channel facts never
    # stand in for a whole-company total.
    if selected:
        deltas = [point["delta_cents"] for point in selected]
        if all(delta > 0 for delta in deltas):
            parts.append("所选渠道的净额上升" if len(selected) == 1 else "所选渠道净额全部上升")
        elif all(delta < 0 for delta in deltas):
            parts.append("所选渠道的净额下降" if len(selected) == 1 else "所选渠道净额全部下降")
        elif all(delta == 0 for delta in deltas):
            parts.append("所选渠道的净额不变" if len(selected) == 1 else "所选渠道净额全部不变")
        else:
            parts.append("所选渠道净额方向不一致")
    if not has_groups and monthly:
        parts.append("已核对所选金额表的月度净额按月列示，未推断具体两期比较")
    if any(point["base_net_cents"] == 0 for point in comparisons.values()):
        parts.append("基期净额为零的比较，其变化率不可定义")
    if not has_groups and not monthly:
        parts.append("已核对所选金额表的净额")
    return "；".join(parts) + "。" + _LIMITATION
