"""Regression checks from actual API-run failures; no network or known answers sent to models."""

import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from local_data_agent.report import render_report


def artifact(root):
    path = root / "result.parquet"
    pq.write_table(pa.table({"month": ["2026-09"], "net_cents": [31000]}), path)
    return {"artifact_id": "artifact_test", "output_name": "net", "kind": "table", "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "row_count": 1,
            "metric_contract": {"metric_id": "CASH_NET_v1", "status": "APPROVED"}}


def claim():
    return {"claim_id": "current", "artifact_id": "net", "column": "net_cents", "filter": {"month": "2026-09"},
            "aggregation": "single_row", "display_unit": "CNY_yuan", "label": "本期现金净额"}


def test_approved_metric_version_identifier_is_not_a_numeric_fact(tmp_path):
    report = render_report(tmp_path, [claim()], {"net": artifact(tmp_path)},
                           "按已批准的 CASH_NET_v1 分析，退款增加；业务原因尚未核实。")
    assert "310.00 元" in report["answer"]
    assert "CASH_NET_v1" in report["answer"]


@pytest.mark.parametrize("text", ["按 CASH_NET_v1 计算，净额999元。", "按未批准 CASH_NET_v2 计算。", "净额310元。"])
def test_metric_identifier_exemption_cannot_hide_unbound_values(tmp_path, text):
    with pytest.raises(ValueError, match="unbound numeric"):
        render_report(tmp_path, [claim()], {"net": artifact(tmp_path)}, text)


def test_filters_are_typed_objects_not_sql_fragments(tmp_path):
    bad = claim()
    bad["filter"] = "month=2026-09"
    with pytest.raises(ValueError, match="bounded mapping"):
        render_report(tmp_path, [bad], {"net": artifact(tmp_path)}, "退款增加；原因未核实。")
