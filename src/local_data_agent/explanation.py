"""Deterministic host-generated explanation of validated money evidence.

Two public functions only:

* ``build_explanation`` re-validates caller-supplied rows and recomputes every
  number itself, then emits statements plus a bounded, verbatim ticket section.
* ``render_explanation`` renders that payload as escaped Markdown.

The module never calls a model, database, or network, never runs SQL, and never
accepts free prose: every number in the output is derived from validated rows
and copy-verified citations. Algebraic contribution ("what moved") is kept
strictly separate from business causation ("why"), and a record association is
never presented as a causal link. Missing input fails loudly as ``ValueError``;
nothing is silently defaulted to zero and no fact is clipped to a prefix.
"""
from __future__ import annotations

import copy
import html
import re
from decimal import Decimal

__all__ = ["build_explanation", "render_explanation"]

_MONTH_RE = re.compile(r"(\d{4})-(0[1-9]|1[0-2])")
_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-\.]{0,127}")
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_EVIDENCE_ID_RE = re.compile(r"ticket:[A-Za-z0-9_\-\.]{1,64}:[A-Za-z0-9_\-\.]{1,64}:\d+-\d+")
_MONEY_COLUMNS = ("paid_cents", "refund_cents", "net_cents")
_MONTH_COLUMNS = frozenset(("month",) + _MONEY_COLUMNS)
_CHANNEL_COLUMNS = _MONTH_COLUMNS | {"channel_code"}
_STATUSES = ("verified", "unlinked", "unverified", "conflicting")
_MAX_EVIDENCE = 8
_MAX_TEXT_CHARS = 1200
_MAX_TOTAL_UTF8 = 8000

_LIMITATIONS = (
    "本说明仅复述已校验的月度与渠道款项合计，不解释业务原因。",
    "金额变化按 净额变化 = 付款变化 - 退款变化 的代数分解给出，属于算术贡献，不同于因果业务原因。",
    "本说明不推断渠道、事件或工单之间的因果方向；关联不等于因果。",
    "渠道合计仅覆盖已提供的渠道，未列出的渠道不在本次范围内。",
    "本地月份以记账月标识为准，时区差异不构成本地月份异常的结论。",
)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _money(cents) -> str:
    """Exact integer cents -> yuan text via Decimal. Never float, never defaulted."""
    if not _is_int(cents):
        raise ValueError(f"金额必须为整数分且不能是布尔值：{cents!r}")
    return f"{Decimal(cents) / Decimal(100):,.2f} 元"


def _month_key(value) -> int:
    if not isinstance(value, str):
        raise ValueError("month 必须为 YYYY-MM 字符串")
    match = _MONTH_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"非法记账月份：{value!r}")
    return int(match[1]) * 12 + int(match[2])


def _validate_rows(rows, base_month, current_month, channel: bool) -> dict:
    """Return ``{month: {key: row}}`` for exactly the two requested months.

    A row's fields must match the agreed set exactly, its money must be
    non-negative integer cents with ``net = paid - refund``, months must be
    distinct and ordered base < current, and no month may repeat a key.
    """
    expected = _CHANNEL_COLUMNS if channel else _MONTH_COLUMNS
    if not isinstance(rows, list):
        raise ValueError("rows 必须为列表")
    buckets: dict = {base_month: {}, current_month: {}}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("行必须为映射")
        extra = set(row) - expected
        missing = expected - set(row)
        if extra or missing:
            raise ValueError(f"行字段不符合约定：多余={sorted(extra)} 缺失={sorted(missing)}")
        if row["month"] not in buckets:
            raise ValueError(f"行包含未请求的记账月份：{row['month']!r}")
        for column in _MONEY_COLUMNS:
            if not _is_int(row[column]):
                raise ValueError(f"{column} 必须为整数分且不能是布尔值：{row[column]!r}")
        if row["paid_cents"] < 0 or row["refund_cents"] < 0:
            raise ValueError("paid_cents 与 refund_cents 不能为负数")
        if row["net_cents"] != row["paid_cents"] - row["refund_cents"]:
            raise ValueError("net_cents 必须等于 paid_cents - refund_cents")
        if channel:
            code = row["channel_code"]
            if not isinstance(code, str) or not _ID_RE.fullmatch(code):
                raise ValueError(f"channel_code 必须为非空合法标识：{code!r}")
        else:
            code = row["month"]
        bucket = buckets[row["month"]]
        if code in bucket:
            raise ValueError(f"重复的行：month={row['month']} key={code}")
        bucket[code] = dict(row)
    for label, month in (("基期", base_month), ("当期", current_month)):
        if not buckets[month]:
            raise ValueError(f"{label}行缺失：没有任何行匹配 {month}（不会按缺失处理）")
    if channel and set(buckets[base_month]) != set(buckets[current_month]):
        raise ValueError("两月渠道集合必须完全一致且非空："
                         f"{sorted(set(buckets[base_month]) ^ set(buckets[current_month]))}")
    return buckets


def _normalize_refs(artifact_refs) -> tuple[list, list]:
    """Accept only real artifact ids shaped ``{'month': [...], 'channel': [...]}``."""
    month_refs: list = []
    channel_refs: list = []
    if artifact_refs is None:
        return month_refs, channel_refs
    if not isinstance(artifact_refs, dict):
        raise ValueError("artifact_refs 必须为映射")
    for key, value in artifact_refs.items():
        if key not in ("month", "channel"):
            raise ValueError(f"未知的产物引用类别：{key!r}")
        if not isinstance(value, list):
            raise ValueError(f"artifact_refs[{key!r}] 必须为标识列表")
        for ref in value:
            if not isinstance(ref, str) or not _ID_RE.fullmatch(ref):
                raise ValueError(f"产物引用标识非法：{ref!r}")
        if key == "month":
            month_refs = list(value)
        else:
            channel_refs = list(value)
    return month_refs, channel_refs


def _statement(kind, text, values, refs) -> dict:
    if not isinstance(text, str) or not text:
        raise ValueError("statement 文本必须为非空字符串")
    return {"kind": kind, "text": text, "source_refs": [ref for ref in refs if ref], "values": values}


def _source_validation(source_validation, month_refs) -> tuple[dict, list, str]:
    """Summarize the *actual* checks list. Missing result means unverified."""
    scope = {"checks": [], "passed": None, "derived_event_months": "unverified",
             "note": "仅引用已提供的校验项；缺失或 None 的校验视为未验证，绝不默认通过。"}
    if source_validation is None:
        return scope, [], "unverified"
    if not isinstance(source_validation, dict) or not isinstance(source_validation.get("checks"), list):
        raise ValueError("source_validation 必须为 {'checks': [...]} 映射")
    seen: dict = {}
    for check in source_validation["checks"]:
        if not isinstance(check, dict) or not isinstance(check.get("name"), str):
            raise ValueError("校验项必须为带 name 的映射")
        name = check["name"]
        passed = check.get("passed")
        if passed is not None and not isinstance(passed, bool):
            raise ValueError(f"校验项 {name} 的 passed 必须为布尔值或缺失")
        if name in seen:
            raise ValueError(f"校验项重复：{name}")
        seen[name] = passed
        detail = None if passed is not False else str(check.get("detail") or check.get("reason") or "未提供失败明细")[:_MAX_TEXT_CHARS]
        scope["checks"].append({"name": name, "passed": passed, "detail": detail})
    declared = source_validation.get("passed")
    if declared is not None and not isinstance(declared, bool):
        raise ValueError("source_validation passed 必须为布尔值或缺失")
    scope["passed"] = declared

    derived = seen.get("derived_event_months")
    if derived is None:
        state = "unverified"
        body = "派生事件月份校验未提供结果或为 None，视为未验证（不默认通过）。"
    elif derived:
        state = "passed"
        body = "派生事件月份校验通过（check: source_validation:derived_event_months）。"
    else:
        state = "failed"
        detail = next(item["detail"] for item in scope["checks"] if item["name"] == "derived_event_months")
        body = f"派生事件月份校验失败（check: source_validation:derived_event_months）：{detail}"
    scope["derived_event_months"] = state
    scope["check_ref"] = "source_validation:derived_event_months"
    return scope, [_statement("source_validation", body,
                              {"check_ref": "source_validation:derived_event_months", "state": state},
                              ["source_validation:derived_event_months"])], state


def _channel_decomposition(ch_base, ch_current) -> list:
    items = []
    for code in sorted(ch_base):
        base, current = ch_base[code], ch_current[code]
        paid = current["paid_cents"] - base["paid_cents"]
        refund = current["refund_cents"] - base["refund_cents"]
        net = current["net_cents"] - base["net_cents"]
        if net != paid - refund:
            raise ValueError(f"渠道 {code} 的净额变化与付款/退款变化不一致")
        items.append({"channel_code": code, "paid_delta_cents": paid,
                      "refund_delta_cents": refund, "net_delta_cents": net})
    return sorted(items, key=lambda item: (item["net_delta_cents"], item["channel_code"]))


def _association(association, duplicate_warning: bool) -> dict:
    if not isinstance(association, dict):
        raise ValueError("association 必须为映射")
    status = association.get("status")
    if status not in _STATUSES:
        raise ValueError(f"association.status 非法：{status!r}")
    order_id = association.get("order_id")
    order_channel = association.get("order_channel")
    for label, value in (("order_id", order_id), ("order_channel", order_channel)):
        if value is not None and (not isinstance(value, str) or not _ID_RE.fullmatch(value)):
            raise ValueError(f"association.{label} 必须为空或合法标识")
    link_origin = association.get("link_origin")
    if link_origin is not None and (not isinstance(link_origin, str) or not link_origin):
        raise ValueError("association.link_origin 必须为空或非空字符串")
    assoc_source_id = association.get("association_source_id")
    if assoc_source_id is not None and (not isinstance(assoc_source_id, str) or not _ID_RE.fullmatch(assoc_source_id)):
        raise ValueError("association.association_source_id 必须为空或合法标识")
    assoc_source_hash = association.get("association_source_hash")
    if assoc_source_hash is not None and (not isinstance(assoc_source_hash, str) or not _HASH_RE.fullmatch(assoc_source_hash)):
        raise ValueError("association.association_source_hash 必须为空或 64 位十六进制")
    if status == "verified" and (order_id is None or assoc_source_id is None or assoc_source_hash is None):
        raise ValueError("已验证关联必须具有订单和来源身份")
    if not isinstance(duplicate_warning, bool):
        raise ValueError("association.duplicate_warning 必须为布尔值")
    detail = association.get("detail")
    if detail is not None and not isinstance(detail, str):
        raise ValueError("association.detail 必须为字符串或缺失")
    return copy.deepcopy(association)


def _tickets(ticket_result) -> tuple[list, dict, list]:
    """Copy citations verbatim after checking the real counters and bounds."""
    empty_scope = {"status": "not_searched", "population_ticket_count": None,
                   "matched_ticket_count": None, "returned_ticket_count": None,
                   "association_counts": {status: 0 for status in _STATUSES},
                   "filters": None, "query": None, "source_id": None, "association_only": True}
    if ticket_result is None:
        return [], empty_scope, []
    if not isinstance(ticket_result, dict):
        raise ValueError("ticket_result 必须为映射")
    if ticket_result.get("type") == "ticket" or any(key in ticket_result for key in ("citations", "physical_row", "text_line", "char_offset")):
        raise ValueError("ticket_result 使用非法字段（顶层 type/citations/physical_row/text_line/char_offset 不受支持）")
    for key in _STATUSES:
        if key in ticket_result:
            raise ValueError(f"ticket_result 顶层不能出现无范围的计数键：{key!r}")

    population, matched, returned = (ticket_result.get("population_ticket_count"),
                                     ticket_result.get("matched_ticket_count"),
                                     ticket_result.get("returned_ticket_count"))
    for label, value in (("population_ticket_count", population), ("matched_ticket_count", matched),
                         ("returned_ticket_count", returned)):
        if not _is_int(value) or value < 0:
            raise ValueError(f"{label} 必须为非负整数（布尔值无效）")
    evidence_raw = ticket_result.get("evidence")
    if not isinstance(evidence_raw, list):
        raise ValueError("ticket_result 缺少 evidence 列表")
    if not returned == len(evidence_raw) <= matched <= population:
        raise ValueError("计数必须满足 0 <= returned == len(evidence) <= matched <= population")

    source_id = ticket_result.get("source_id")
    if not isinstance(source_id, str) or not _ID_RE.fullmatch(source_id):
        raise ValueError("ticket_result.source_id 必须为合法标识")

    full = {} if ticket_result.get("association_counts") is None else ticket_result["association_counts"]
    if not isinstance(full, dict):
        raise ValueError("association_counts 必须为映射")
    association_counts = {}
    for status in _STATUSES:
        value = full.get(status, 0)
        if not _is_int(value) or value < 0:
            raise ValueError(f"association_counts.{status} 必须为非负整数")
        association_counts[status] = value
    if set(full) != set(_STATUSES):
        raise ValueError(f"association_counts 必须恰好包含 {list(_STATUSES)}（全量筛选后计数）")
    if sum(association_counts.values()) != population:
        raise ValueError("association_counts 合计必须等于 population_ticket_count")

    evidence: list = []
    seen: set = set()
    seen_tickets: set = set()
    total_bytes = 0
    for item in evidence_raw:
        if not isinstance(item, dict):
            raise ValueError("evidence 记录必须为映射")
        if item.get("type") != "ticket":
            raise ValueError("evidence 记录必须为 type='ticket'")
        evidence_id = item.get("evidence_id")
        ticket_id = item.get("ticket_id")
        if not isinstance(evidence_id, str) or not _EVIDENCE_ID_RE.fullmatch(evidence_id):
            raise ValueError("非法工单引用标识")
        for label, value in (("ticket_id", ticket_id),
                             ("source_id", item.get("source_id"))):
            if not isinstance(value, str) or not _ID_RE.fullmatch(value):
                raise ValueError(f"evidence {label} 必须为合法标识")
        expected_id = f"ticket:{item['source_id']}:{ticket_id}:{item.get('start_char')}-{item.get('end_char')}"
        if evidence_id in seen:
            raise ValueError(f"evidence_id 重复：{evidence_id}")
        if evidence_id != expected_id or not _EVIDENCE_ID_RE.fullmatch(evidence_id):
            raise ValueError(f"evidence_id 必须形如 ticket:source:id:start-end：{evidence_id!r}")
        seen.add(evidence_id)
        if ticket_id in seen_tickets:
            raise ValueError("重复的工单引用")
        seen_tickets.add(ticket_id)
        for label in ("source_hash", "record_hash"):
            value = item.get(label)
            if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
                raise ValueError(f"evidence {label} 必须为 64 位小写十六进制")
        values = {}
        for label in ("source_line", "start_line", "end_line", "line_count",
                      "start_char", "end_char"):
            value = item.get(label)
            if not _is_int(value):
                raise ValueError(f"evidence {label} 必须为整数（布尔值无效）")
            values[label] = value
        text = item.get("text")
        if not isinstance(text, str):
            raise ValueError("evidence.text 必须为字符串")
        if values["source_line"] < 1:
            raise ValueError("source_line 为 1 基，必须 >= 1")
        if not 1 <= values["start_line"] <= values["end_line"] <= values["line_count"]:
            raise ValueError("行号必须满足 1 <= start_line <= end_line <= line_count")
        if not 0 <= values["start_char"] < values["end_char"]:
            raise ValueError("字符偏移必须满足 0 <= start_char < end_char（全文 0 基）")
        if len(text) != values["end_char"] - values["start_char"]:
            raise ValueError("text 长度必须等于 end_char - start_char（不得裁剪）")
        if len(text) > _MAX_TEXT_CHARS:
            raise ValueError(f"单条引用文本不得超过 {_MAX_TEXT_CHARS} 字符")
        total_bytes += len(text.encode("utf-8"))
        if total_bytes > _MAX_TOTAL_UTF8:
            raise ValueError(f"引用文本合计不得超过 {_MAX_TOTAL_UTF8} 字节")
        if not isinstance(item.get("association"), dict):
            raise ValueError("association 必须为映射")
        duplicate_warning = item["association"].get("duplicate_warning", False)
        if not isinstance(duplicate_warning, bool):
            raise ValueError("duplicate_warning 必须为布尔值")
        _association(item.get("association"), duplicate_warning)
        evidence.append(copy.deepcopy(item))
    if len(evidence) > _MAX_EVIDENCE:
        raise ValueError(f"引用不得超过 {_MAX_EVIDENCE} 条")

    for label in ("filters", "query"):
        value = ticket_result.get(label)
        if value is not None and not isinstance(value, (dict, list, str)):
            raise ValueError(f"ticket_result.{label} 必须为映射、列表或字符串")
    scope = {"status": "searched", "population_ticket_count": population,
             "matched_ticket_count": matched, "returned_ticket_count": returned,
             "association_counts": association_counts,
             "filters": copy.deepcopy(ticket_result.get("filters")),
             "query": ticket_result.get("query"), "source_id": source_id,
             "association_only": True,
             "sample_note": "引用为返回样本，不能据此推断总体数量。"}
    return evidence, scope, [
        _statement("ticket_scope",
                   f"工单计数：总体 {population}、匹配 {matched}、返回 {returned}；本次逐字引用 {len(evidence)} 条。"
                   f" 关联计数（全量筛选后）：已验证 {association_counts['verified']}、未关联 "
                   f"{association_counts['unlinked']}、未验证 {association_counts['unverified']}、"
                   f"冲突 {association_counts['conflicting']}。计数分别呈现，样本不能推断总体。",
                   {"population_ticket_count": population, "matched_ticket_count": matched,
                    "returned_ticket_count": returned, "association_counts": dict(association_counts),
                    "returned_evidence_count": len(evidence)},
                   [item["evidence_id"] for item in evidence]),
        _statement("ticket_warning",
                   "工单引用仅表示记录之间存在关联（关联不等于因果）：即使关联状态为 verified，"
                   "也不代表因果关系，更不代表退款是变化的原因。",
                   {"association_only": True, "causal": False, "statuses": dict(association_counts)},
                   [item["evidence_id"] for item in evidence]),
    ]


def build_explanation(month_rows, channel_rows, base_month, current_month,
                      ticket_result=None, source_validation=None, artifact_refs=None) -> dict:
    """Explain validated monthly/channel money movement and ticket evidence.

    All numbers are recomputed from the supplied rows. Malformed, duplicate,
    non-integer, out-of-order, or non-reconciling inputs raise ``ValueError``.
    """
    base_key, current_key = _month_key(base_month), _month_key(current_month)
    if base_key == current_key:
        raise ValueError("基期与当期必须为两个不同的月份")
    if base_key > current_key:
        raise ValueError("基期必须早于当期；本模块不会静默交换两个期间")

    month_refs, channel_refs = _normalize_refs(artifact_refs)
    months = _validate_rows(month_rows, base_month, current_month, channel=False)
    ch_base = _validate_rows(channel_rows, base_month, current_month, channel=True)

    base_row, current_row = months[base_month][base_month], months[current_month][current_month]
    for column in _MONEY_COLUMNS:
        for month in (base_month, current_month):
            channels = ch_base[month]
            total = sum(row[column] for row in channels.values())
            if total != months[month][month][column]:
                raise ValueError(f"{month} 渠道 {column} 合计 {total} 与月度 {column} 不一致")
    if not ch_base[base_month]:
        raise ValueError("渠道集合必须非空")

    paid_delta = current_row["paid_cents"] - base_row["paid_cents"]
    refund_delta = current_row["refund_cents"] - base_row["refund_cents"]
    net_delta = current_row["net_cents"] - base_row["net_cents"]
    if net_delta != paid_delta - refund_delta:
        raise ValueError("月度净额变化与付款/退款变化不一致")

    channels = _channel_decomposition(ch_base[base_month], ch_base[current_month])
    if sum(item["net_delta_cents"] for item in channels) != net_delta:
        raise ValueError("渠道净额变化合计与月度净额变化不一致")

    def direction(delta: int) -> str:
        return "上升" if delta > 0 else ("下降" if delta < 0 else "持平")

    statements = [_statement("month_delta",
                             f"从 {base_month} 到 {current_month}：付款变化 {_money(paid_delta)}，"
                             f"退款变化 {_money(refund_delta)}，净额变化 = 付款变化 - 退款变化 = "
                             f"{_money(net_delta)}；净额方向为{direction(net_delta)}（{base_month} 至 {current_month}）。",
                             {"base_month": base_month, "current_month": current_month,
                              "paid_delta_cents": paid_delta, "refund_delta_cents": refund_delta,
                              "net_delta_cents": net_delta},
                             month_refs)]

    ranked = "、".join(f"{item['channel_code']}（净额变化 {_money(item['net_delta_cents'])}）" for item in channels)
    statements.append(_statement("channel_rank",
                                 f"按净额变化升序的渠道分解：{ranked}。各渠道净额变化合计等于月度净额变化 "
                                 f"{_money(net_delta)}；该排序是代数贡献排序，不是原因排序。",
                                 {"net_delta_cents": net_delta, "channels": copy.deepcopy(channels)},
                                 channel_refs))

    negative = [item for item in channels if item["net_delta_cents"] < 0]
    if net_delta < 0:
        if not negative:
            raise ValueError("总净额下降但没有任何渠道为负向贡献")
        all_negative = len(negative) == len(channels)
        worst = negative[0]["net_delta_cents"]
        top = [item["channel_code"] for item in negative if item["net_delta_cents"] == worst]
        tie = "（并列最多）" if len(top) > 1 else ""
        if all_negative:
            lead = "本次范围内所有渠道净额均为负向贡献"
        else:
            lead = "本次范围内仅部分渠道净额下降，另有渠道为正或持平，因此不构成“所有渠道”的结论"
        statements.append(_statement("channel_decline",
                                     f"{lead}：总净额变化 {_money(net_delta)}。负向渠道为 "
                                     + "、".join(f"{item['channel_code']}（{_money(item['net_delta_cents'])}）"
                                                 for item in negative)
                                     + f"；下降最多的是 {'、'.join(top)}{tie}，贡献 {_money(worst)}。",
                                     {"net_delta_cents": net_delta,
                                      "all_channels_negative": all_negative,
                                      "negative_contributions": [
                                          {"channel_code": item["channel_code"],
                                           "net_delta_cents": item["net_delta_cents"]} for item in negative],
                                      "top_negative_channels": top,
                                      "top_negative_delta_cents": worst},
                                     channel_refs))
    else:
        statements.append(_statement("channel_decline",
                                     f"总净额变化 {_money(net_delta)}，整体净额未下降；"
                                     "个别渠道仍可能有负向贡献，具体以渠道分解为准。",
                                     {"net_delta_cents": net_delta, "all_channels_negative": False,
                                      "negative_contributions": copy.deepcopy(negative),
                                      "top_negative_channels": [item["channel_code"] for item in negative
                                                                if item["net_delta_cents"] == negative[0]["net_delta_cents"]]},
                                     channel_refs))

    statements.append(_statement("decomposition_note",
                                 "以上方向与金额均为基期到当期的算术贡献分解（付款变动的效果与退款变动的效果分别列出，"
                                 "两者共同构成净额变化）；该分解只是代数恒等式的呈现，不构成任何因果业务原因。",
                                 {"net_delta_cents": net_delta, "paid_delta_cents": paid_delta,
                                  "refund_delta_cents": refund_delta,
                                  "paid_direction": direction(paid_delta),
                                  "refund_direction": direction(refund_delta)},
                                 month_refs))

    verification, validation_statements, validation_state = _source_validation(source_validation, month_refs)
    statements.extend(validation_statements)

    limitations = list(_LIMITATIONS)
    if paid_delta < 0 and refund_delta > 0:
        limitations.append("付款与退款同时变动，不能把净额变化表述为“退款是唯一原因”。")
    if validation_state != "passed":
        limitations.append("来源校验未确认通过，因此本说明不对本地月份异常作任何推断。")
    ticket_evidence, ticket_scope, ticket_statements = _tickets(ticket_result)
    if ticket_scope["status"] == "not_searched":
        limitations.append("未提供工单检索结果，因此没有工单计数，也没有任何关联引用。")
    statements.extend(ticket_statements)

    return {"statements": statements, "ticket_evidence": ticket_evidence, "ticket_scope": ticket_scope,
            "limitations": list(dict.fromkeys(limitations))[:12], "verification_scope": verification}


def _escape(value) -> str:
    """Escape pipes, brackets, backticks, newlines and HTML so quotes stay inert."""
    text = html.escape(str(value), quote=False).replace("\\", "\\\\").replace("`", "\\`").replace("*", "\\*")
    text = text.replace("[", "\\[").replace("]", "\\]").replace("!", "\\!")
    text = text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    return text.replace("<", "\\<").replace("&", "\\&")


def _quote_row(item: dict) -> str:
    bounds = f"行 {item['start_line']}-{item['end_line']}（共 {item['line_count']} 行）"
    offsets = f"字符 {item['start_char']}-{item['end_char']}"
    association = item["association"]
    detail = f"，说明 {_escape(association['detail'])}" if association["detail"] else ""
    link = (f"状态 {association['status']}；order_id {_escape(association['order_id'] or '无')}；"
            f"order_channel {_escape(association['order_channel'] or '无')}；来源 {_escape(association['link_origin'])}；"
            f"关联来源 {_escape(association['association_source_id'])}/{_escape(association['association_source_hash'])}；"
            f"重复警示 {association['duplicate_warning']}{detail}")
    return (f"| {_escape(item['evidence_id'])} | {_escape(item['text'])} | "
            f"{item['source_line']} | {bounds} | {offsets} | {_escape(item['source_hash'])} | "
            f"{_escape(item['record_hash'])} | {_escape(link)} |")


def _count(value) -> str:
    if value is None:
        return "未提供"
    if _is_int(value):
        return str(value)
    raise ValueError("计数必须为整数或未提供")


def render_explanation(payload) -> str:
    """Render the host-generated payload as escaped Markdown (quotes unchanged)."""
    if not isinstance(payload, dict):
        raise ValueError("payload 必须为映射")
    for field in ("statements", "ticket_evidence", "ticket_scope", "limitations", "verification_scope"):
        if field not in payload:
            raise ValueError(f"payload 缺少字段：{field}")
    if not isinstance(payload["statements"], list) or not isinstance(payload["ticket_evidence"], list):
        raise ValueError("payload 结构非法")

    lines = ["## 解释（由本地程序生成，数字均为计算所得）", "",
             "本节不接受模型自定义文本：所有数值都由本地程序从已校验行重新计算。", "",
             "| 说明 | 类别 | 来源引用 |", "|---|---|---|"]
    for statement in payload["statements"]:
        if not isinstance(statement, dict) or not isinstance(statement.get("text"), str):
            raise ValueError("statement 结构非法")
        refs = statement.get("source_refs") or []
        if not isinstance(refs, list):
            raise ValueError("source_refs 必须为列表")
        lines.append(f"| {_escape(statement['text'])} | {_escape(statement.get('kind', 'note'))} "
                     f"| {_escape('、'.join(str(ref) for ref in refs) or '未提供产物引用')} |")

    evidence = payload["ticket_evidence"]
    lines.extend(["", "### 工单引用（逐字复制，仅供参考，不构成因果）", "",
                  "| 引用 | 原文（逐字，未裁剪） | 物理行 | 文本行范围 | 全局字符偏移 | 来源哈希 | 记录哈希 | 关联 |",
                  "|---|---|---|---|---|---|---|---|"])
    if evidence:
        for item in evidence:
            lines.append(_quote_row(item))
    else:
        lines.append("| 无 | 本次没有逐字引用 | - | - | - | - | - | - |")
    lines.extend(["", "原文中的链接、图片与命令不会被渲染为可执行或可点击内容；原文数字不作为结论。"])
    lines.extend(["", "引用是返回的样本，总体与匹配数量为全量筛选后的计数，二者按范围分开列示，样本不能推断总体。"])

    scope = payload["ticket_scope"] if isinstance(payload["ticket_scope"], dict) else {}
    lines.extend(["", "### 工单计数范围", "",
                  f"- 总体数量（全量筛选后）：{_count(scope.get('population_ticket_count'))}",
                  f"- 匹配数量（全量筛选后）：{_count(scope.get('matched_ticket_count'))}",
                  f"- 返回数量（返回样本）：{_count(scope.get('returned_ticket_count'))}",
                  f"- 逐字引用条数（返回样本）：{len(evidence)}"])
    counts = scope.get("association_counts") or {}
    lines.append("- 关联计数（全量筛选后）：" + "、".join(
        f"{status} {_count(counts.get(status))}" for status in _STATUSES))
    if scope.get("query") is not None:
        lines.append(f"- 查询：{_escape(scope['query'])}")
    if scope.get("filters") is not None:
        lines.append(f"- 筛选：{_escape(scope['filters'])}")
    lines.append(f"- 计数状态：{_escape(scope.get('status', 'not_searched'))}；"
                 "所有计数为查询层提供的全量筛选后计数，引用条数是返回样本，不从样本反推总体。")

    verification = payload["verification_scope"] if isinstance(payload["verification_scope"], dict) else {}
    lines.extend(["", "### 校验范围", "",
                  f"- 来源校验：{_escape(verification.get('derived_event_months', 'unverified'))}"
                  f"（check: {_escape(verification.get('check_ref', 'source_validation:derived_event_months'))}）"])
    checks = verification.get("checks") or []
    if checks:
        lines.append("- 已见校验项：" + "、".join(
            f"{_escape(item['name'])}={'通过' if item['passed'] is True else ('失败' if item['passed'] is False else '未提供')}"
            for item in checks if isinstance(item, dict)) + "；未提供的校验不视为通过。")
    else:
        lines.append("- 已见校验项：无；未提供的校验不视为通过。")
    lines.append("- 本说明不依据时区差推断本地月份异常。")

    lines.extend(["", "### 范围与限制", ""])
    for limitation in payload["limitations"]:
        lines.append(f"- {_escape(limitation)}")
    lines.extend(["", "关联不等于因果：引用与计数不构成业务原因。", ""])
    return "\n".join(lines)
