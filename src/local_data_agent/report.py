"""Render host-calculated claims; model prose cannot supply numeric answers."""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq


def _inside(run_dir: Path, path: str | Path) -> Path:
    candidate = Path(path)
    candidate = (candidate if candidate.is_absolute() else run_dir / candidate).resolve()
    if not candidate.is_relative_to(run_dir.resolve()):
        raise ValueError("Artifact path must be inside the current run workspace")
    if not candidate.is_file():
        raise ValueError("Artifact file does not exist")
    return candidate


def _artifact_for(binding: str, artifacts: dict) -> dict:
    matches = []
    for name, artifact in artifacts.items():
        if binding in {name, artifact.get("artifact_id"), artifact.get("output_name")}:
            if all(artifact is not previous for previous in matches):
                matches.append(artifact)
    if len(matches) != 1:
        raise ValueError("Claim artifact binding is missing or ambiguous")
    return matches[0]


def _typed_equal(value, expected) -> bool:
    # Do not silently match a string "1", integer 1, or boolean True.
    if value is None or expected is None:
        return value is expected
    if isinstance(value, bool) or isinstance(expected, bool):
        return type(value) is type(expected) and value == expected
    if type(value) is not type(expected):
        # A JSON integer can represent an Arrow integer of any bit width.
        return False
    return value == expected


def _number(value) -> int | float | Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError("Numeric claims must bind a numeric column")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Numeric claims cannot contain NaN or infinity")
    if isinstance(value, Decimal) and not value.is_finite():
        raise ValueError("Numeric claims cannot contain NaN or infinity")
    return value


def _check_filter_type(field, value) -> None:
    if value is None:
        return
    field_type = field.type
    if pa.types.is_dictionary(field_type):
        field_type = field_type.value_type
    allowed = ((pa.types.is_string(field_type) or pa.types.is_large_string(field_type)) and type(value) is str
               or pa.types.is_boolean(field_type) and type(value) is bool
               or pa.types.is_integer(field_type) and type(value) is int
               or pa.types.is_floating(field_type) and type(value) is float)
    if not allowed or isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Claim filter type must match the actual artifact column")


def _check_label_numbers(label: str, filters: dict) -> None:
    """Allow date identifiers from bound month filters, never free numeric facts."""
    months = []
    for name, value in filters.items():
        if (name == "month" or name.endswith("_month")) and isinstance(value, str):
            match = re.fullmatch(r"(\d{4})-(\d{2})", value)
            if match and 1 <= int(match[2]) <= 12:
                months.append((value, int(match[1]), int(match[2])))
    date_spans = [match.span() for month, _, _ in months for match in re.finditer(re.escape(month), label)]
    for token in re.finditer(r"\d+", label):
        if any(start <= token.start() and token.end() <= end for start, end in date_spans):
            continue
        suffix = label[token.end():]
        value = int(token.group())
        if suffix.startswith("年") and any(value == year for _, year, _ in months):
            continue
        if suffix.startswith("月") and any(value == month for _, _, month in months):
            continue
        raise ValueError("Claim label contains an unbound numeric token; only bound month dates are allowed")


def _display(value, unit: str) -> str:
    if unit == "CNY_yuan":
        value = Decimal(str(_number(value))) / Decimal(100)
        return f"{value:,.2f} 元"
    if unit != "raw":
        raise ValueError("Unknown display unit")
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        value = _number(value)
        return str(value)
    if value is None:
        return "不可定义"
    if isinstance(value, (bool, str)):
        return str(value)
    raise ValueError("Unsupported scalar claim value")


def _markdown(value) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def render_report(run_dir: Path, claims: list[dict], artifacts: dict, summary: str,
                  evidence: list[dict] | None = None) -> dict:
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError("Run workspace does not exist")
    if not isinstance(summary, str) or len(summary) > 8000:
        raise ValueError("Summary must be bounded text")
    numeric_view = summary
    # An approved identifier's version digit is not a freehand monetary fact.
    metric_ids = {artifact.get("metric_contract", {}).get("metric_id") for artifact in artifacts.values()
                  if artifact.get("metric_contract", {}).get("status") == "APPROVED"}
    for identifier in metric_ids:
        if isinstance(identifier, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", identifier):
            numeric_view = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(identifier) + r"(?![A-Za-z0-9_])", "metric", numeric_view)
    if re.search(r"\d", numeric_view):
        raise ValueError("Summary contains unbound numeric text; put values in bound claims")
    if not isinstance(claims, list) or not 1 <= len(claims) <= 40:
        raise ValueError("Report requires between one and forty bound claims")
    seen = set()
    validated = []
    tables = {}
    allowed = {"claim_id", "artifact_id", "column", "filter", "aggregation", "display_unit", "label"}
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) - allowed:
            raise ValueError("Unknown claim fields")
        claim_id = claim.get("claim_id")
        if not isinstance(claim_id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_\-]{0,63}", claim_id) or claim_id in seen:
            raise ValueError("Claim identifiers must be valid and unique")
        seen.add(claim_id)
        label = claim.get("label", claim_id)
        if not isinstance(label, str) or not 1 <= len(label) <= 200:
            raise ValueError("Invalid claim label")
        artifact = _artifact_for(claim.get("artifact_id"), artifacts)
        if artifact.get("kind") != "table":
            raise ValueError("Claim must bind a table artifact")
        path = _inside(run_dir, artifact["path"])
        if path not in tables:
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            table = pq.read_table(path)
            tables[path] = (table, actual_hash)
        table, actual_hash = tables[path]
        if not artifact.get("sha256") or actual_hash != artifact["sha256"]:
            raise ValueError("Claim artifact hash failed verification")
        if table.num_rows != artifact.get("row_count"):
            raise ValueError("Claim artifact row count failed verification")
        column = claim.get("column")
        if column not in table.column_names:
            raise ValueError("Claim column does not exist")
        filters = claim.get("filter", {})
        if not isinstance(filters, dict) or len(filters) > 8:
            raise ValueError("Claim filters must be a bounded mapping")
        if any(name not in table.column_names for name in filters):
            raise ValueError("Claim filter column does not exist")
        if any(not isinstance(value, (str, int, float, bool, type(None))) for value in filters.values()):
            raise ValueError("Claim filters only support scalar typed equality")
        for name, expected in filters.items():
            _check_filter_type(table.schema.field(name), expected)
        _check_label_numbers(label, filters)
        rows = [row for row in table.to_pylist()
                if all(_typed_equal(row[name], value) for name, value in filters.items())]
        values = [row[column] for row in rows]
        aggregation = claim.get("aggregation")
        if aggregation == "single_row":
            if len(values) != 1:
                raise ValueError("single_row claim must select exactly one row")
            value = values[0]
        elif aggregation == "sum":
            if not values or any(value is None for value in values):
                raise ValueError("sum claim requires nonempty numeric values without nulls")
            value = sum(_number(value) for value in values)
        elif aggregation == "count":
            value = sum(value is not None for value in values)
        else:
            raise ValueError("Unsupported claim aggregation")
        display_unit = claim.get("display_unit", "raw")
        if display_unit == "CNY_yuan" and aggregation == "count":
            raise ValueError("Counts cannot be displayed as currency")
        if display_unit == "CNY_yuan":
            field = table.schema.field(column)
            unit = (field.metadata or {}).get(b"unit")
            if not pa.types.is_integer(field.type) or not (column.endswith("_cents") or unit == b"CNY_cents"):
                raise ValueError("Currency claims must bind integer cents columns with explicit unit")
        formatted = _display(value, display_unit)
        validated.append({**claim, "label": label, "filter": filters,
                          "artifact_id": artifact["artifact_id"], "value": value,
                          "formatted": formatted, "matched_rows": len(rows), "verified_sha256": artifact["sha256"]})
    if evidence is None:
        evidence = []
    if not isinstance(evidence, list) or len(evidence) > 100 or any(not isinstance(item, dict) for item in evidence):
        raise ValueError("Evidence must be a bounded list of records")
    lines = ["# 数据分析结果", "", "以下数值由本地程序读取完整产物并计算；模型未填写数值。", "",
             "| 事实 | 结果 | 绑定 |", "|---|---|---|"]
    for item in validated:
        binding = f"{item['claim_id']} → {item['artifact_id']} / {item['column']} / {item['aggregation']}"
        lines.append(f"| {_markdown(item['label'])} | {_markdown(item['formatted'])} | {_markdown(binding)} |")
    lines.extend(["", "## 分析说明与假设边界", "", "数值事实以上方绑定为准。业务原因与文本线索属于假设，尚未进行因果验证。", "", summary.strip()])
    if evidence:
        lines.extend(["", "## 引用证据", "", "完整出处保存在 evidence.json；引用代表来源记录，不自动证明因果关系。"])
        for item in evidence:
            identifier = item.get("evidence_id", item.get("document_id", "source evidence"))
            lines.append(f"- {_markdown(identifier)}")
    answer = "\n".join(lines).strip() + "\n"
    _atomic_json(run_dir / "claims.json", validated)
    _atomic_json(run_dir / "evidence.json", evidence)
    report_path = run_dir / "report.md"
    temporary = run_dir / "report.md.tmp"
    temporary.write_text(answer, encoding="utf-8")
    temporary.replace(report_path)
    return {"report_path": str(report_path), "answer": answer, "claims": validated}
