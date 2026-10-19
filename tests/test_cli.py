import hashlib
import json
from pathlib import Path
import types
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from local_data_agent import cli
from local_data_agent.doctor import diagnose
from local_data_agent.report import render_report


def test_project_root_argument_before_or_after_command(tmp_path):
    before = cli.parser().parse_args(["--project-root", str(tmp_path), "doctor"])
    after = cli.parser().parse_args(["doctor", "--project-root", str(tmp_path)])
    assert before.project_root == after.project_root == tmp_path


def test_cli_waiting_input_is_success_and_unicode_is_preserved(monkeypatch, capsys):
    monkeypatch.setattr(cli, "dispatch", lambda _: {"status": "waiting_input", "question": "请说明业务口径"})
    assert cli.main(["run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["question"] == "请说明业务口径"


@pytest.mark.parametrize("status,expected_exit", [("SUCCEEDED", 0), ("WAITING_INPUT", 0), ("FAILED", 1), ("CANCELLED", 1), ("UNVERIFIED", 1)])
def test_uppercase_runner_status_exit_code(monkeypatch, capsys, status, expected_exit):
    monkeypatch.setattr(cli, "dispatch", lambda _: {"status": status})
    assert cli.main(["run"]) == expected_exit
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_actual_runner_budget_failure_sets_cli_failure_exit(tmp_path, capsys):
    from local_data_agent.fixtures import bootstrap
    bootstrap(tmp_path)
    assert cli.main(["run", "--project-root", str(tmp_path), "--mode", "replay", "--case", "budget"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAILED"


def test_cli_unverified_failure_and_secret_redaction(monkeypatch, capsys):
    monkeypatch.setattr(cli, "dispatch", lambda _: {"status": "unverified"})
    assert cli.main(["run", "--mode", "live"]) == 1
    monkeypatch.setenv("DATA_AGENT_API_KEY", "private-test-secret")

    def failure(_):
        raise RuntimeError("Provider rejected private-test-secret")

    monkeypatch.setattr(cli, "dispatch", failure)
    assert cli.main(["run", "--mode", "live"]) == 1
    output = capsys.readouterr().out
    assert "private-test-secret" not in output
    assert "[REDACTED]" in output


def test_missing_api_configuration_is_explicitly_unverified(monkeypatch, capsys):
    configuration_error = type("ConfigurationError", (ValueError,), {"__module__": "local_data_agent.model_client"})

    def failure(_):
        raise configuration_error("API credential missing")

    monkeypatch.setattr(cli, "dispatch", failure)
    assert cli.main(["run", "--mode", "live"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "unverified"


def test_doctor_does_not_call_model_without_probe(monkeypatch, tmp_path):
    def unexpected_network():
        raise AssertionError("Unexpected model creation")
    monkeypatch.setitem(sys.modules, "local_data_agent.model_client",
                        types.SimpleNamespace(create_model_client=unexpected_network))
    monkeypatch.setenv("DATA_AGENT_MODEL", "example")
    monkeypatch.setenv("DATA_AGENT_API_KEY", "secret")
    result = diagnose(tmp_path)
    assert result["model"]["configured"] is True
    assert result["model"]["network_called"] is False
    assert "secret" not in json.dumps(result)


def test_doctor_recognizes_local_config_without_reading_credential_contents(monkeypatch, tmp_path):
    monkeypatch.delenv("DATA_AGENT_MODEL", raising=False)
    monkeypatch.delenv("DATA_AGENT_CONFIG_FILE", raising=False)
    key = tmp_path / "credential.txt"
    key.write_text("secret-content-must-not-be-returned", encoding="utf-8")
    (tmp_path / "config.local.json").write_text(json.dumps({"model": "test-model", "key_file": str(key), "api_mode": "chat"}), encoding="utf-8")
    result = diagnose(tmp_path)
    assert result["model"]["configured"] is True
    assert result["model"]["api_mode"] == "chat"
    assert "secret-content" not in json.dumps(result)


def test_evaluation_case_selection_is_passed_to_evaluator(monkeypatch, tmp_path):
    received = {}

    def evaluate(root, **kwargs):
        received.update(kwargs)
        return {"status": "succeeded"}

    monkeypatch.setitem(sys.modules, "local_data_agent.evaluation", types.SimpleNamespace(evaluate=evaluate))
    args = cli.parser().parse_args(["evaluate", "--project-root", str(tmp_path), "--suite", "agent-live", "--cases", "L01,L02"])
    cli.dispatch(args)
    assert received == {"suite": "agent-live", "repeats": 3, "cases": ["L01", "L02"]}


def test_metric_approval_cli_dispatches_explicit_actor(monkeypatch, tmp_path):
    received = {}

    class Runner:
        def __init__(self, root):
            received["root"] = root

        def approve_metric(self, run_id, metric_id, approved_by):
            received.update(run_id=run_id, metric_id=metric_id, approved_by=approved_by)
            return {"status": "WAITING_INPUT"}

    monkeypatch.setitem(sys.modules, "local_data_agent.runner", types.SimpleNamespace(AgentRunner=Runner))
    args = cli.parser().parse_args(["approve-metric", "run_actual", "--project-root", str(tmp_path),
                                    "--metric", "CASH_NET_v1", "--approved-by", "human-owner"])
    assert cli.dispatch(args)["status"] == "WAITING_INPUT"
    assert received == {"root": tmp_path.resolve(), "run_id": "run_actual", "metric_id": "CASH_NET_v1", "approved_by": "human-owner"}


def test_real_paused_metric_approval_preserves_data_snapshot(tmp_path, capsys):
    from local_data_agent.fixtures import bootstrap
    from local_data_agent.replay import ReplayModelClient
    from local_data_agent.runner import AgentRunner

    class AskOnlyClient(ReplayModelClient):
        def __init__(self):
            self.case = "explicit-metric-approval-test"
            self.actions = [("ask_user", {"question": "缺少批准定义，请确认现金事件口径", "reason": "missing_context", "choices": ["现金净额", "其他口径"]})]

    bootstrap(tmp_path, variant="no-metric")
    result = AgentRunner(tmp_path, model=AskOnlyClient()).run("收入是多少")
    assert result["status"] == "WAITING_INPUT"
    checkpoint_path = Path(result["run_dir"]) / "checkpoint.json"
    before = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert cli.main(["approve-metric", result["run_id"], "--project-root", str(tmp_path),
                     "--metric", "CASH_NET_v1", "--approved-by", "human-owner"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "WAITING_INPUT"
    assert output["metric_contract"]["approved_by"] == "human-owner"
    assert output["metric_contract"]["event_time"] == {"orders": "paid_at", "refunds": "refunded_at"}
    after = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert before["manifest"]["sources"] == after["manifest"]["sources"]
    assert after["manifest"]["metric_contracts"][0]["status"] == "APPROVED"
    assert cli.main(["approve-metric", result["run_id"], "--project-root", str(tmp_path),
                     "--metric", "CASH_NET_v1", "--approved-by", "model"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


@pytest.fixture
def table_artifact(tmp_path):
    path = tmp_path / "result.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"month": "aug", "channel": "web", "net_cents": 20000},
        {"month": "aug", "channel": "app", "net_cents": 18000},
        {"month": "sep", "channel": "web", "net_cents": 13000},
        {"month": "sep", "channel": "app", "net_cents": 18000},
    ]), path)
    return {"artifact_id": "a-result", "output_name": "result", "kind": "table", "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "row_count": 4,
            "preview": [{"month": "sep", "net_cents": 99999999}]}


def _claim(**overrides):
    return {"claim_id": "september_net", "artifact_id": "result", "column": "net_cents",
            "filter": {"month": "sep"}, "aggregation": "sum", "display_unit": "CNY_yuan",
            "label": "九月现金净额", **overrides}


def test_report_ignores_preview_and_computes_full_filtered_artifact(tmp_path, table_artifact):
    result = render_report(tmp_path, [_claim()], {"result": table_artifact}, "退款增加是可见现象；原因仍需核查。")
    assert result["claims"][0]["value"] == 31000
    assert result["claims"][0]["formatted"] == "310.00 元"
    assert "99999999" not in result["answer"]
    assert json.loads((tmp_path / "claims.json").read_text(encoding="utf-8"))[0]["matched_rows"] == 2


def test_report_rejects_unbound_numeric_summary(tmp_path, table_artifact):
    with pytest.raises(ValueError, match="unbound"):
        render_report(tmp_path, [_claim()], {"result": table_artifact}, "现金净额是999元")


def test_report_rejects_unbound_numeric_label(tmp_path, table_artifact):
    with pytest.raises(ValueError, match="label.*unbound numeric"):
        render_report(tmp_path, [_claim(label="净额999元")], {"result": table_artifact}, "继续核查原因。")


@pytest.mark.parametrize("label", ["2026-09现金净额", "2026年9月现金净额", "9月现金净额"])
def test_report_allows_label_dates_only_from_bound_month_filter(tmp_path, table_artifact, label):
    path = Path(table_artifact["path"])
    table = pq.read_table(path).to_pylist()
    for row in table:
        row["month"] = {"aug": "2026-08", "sep": "2026-09"}[row["month"]]
    pq.write_table(pa.Table.from_pylist(table), path)
    table_artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    claim = _claim(filter={"month": "2026-09"}, label=label)
    result = render_report(tmp_path, [claim], {"result": table_artifact}, "继续核查原因。")
    assert result["claims"][0]["value"] == 31000


def test_report_date_filter_does_not_allow_currency_numbers_in_label(tmp_path, table_artifact):
    with pytest.raises(ValueError, match="unbound numeric"):
        render_report(tmp_path, [_claim(filter={"month": "2026-09"}, label="9元净额")], {"result": table_artifact}, "继续核查原因。")


@pytest.mark.parametrize("change", [
    {"column": "missing"}, {"filter": {"missing": "sep"}},
    {"aggregation": "single_row"}, {"filter": {"net_cents": "13000"}, "aggregation": "single_row"},
    {"display_unit": "USD"}, {"aggregation": "count", "display_unit": "CNY_yuan"},
])
def test_report_rejects_bad_bindings(tmp_path, table_artifact, change):
    with pytest.raises(ValueError):
        render_report(tmp_path, [_claim(**change)], {"result": table_artifact}, "继续核查原因。")


def test_report_rejects_path_outside_current_run(tmp_path, table_artifact):
    run_dir = tmp_path / "nested_run"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="inside"):
        render_report(run_dir, [_claim()], {"result": table_artifact}, "继续核查原因。")


def test_report_rejects_mutated_artifact(tmp_path, table_artifact):
    table_artifact["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        render_report(tmp_path, [_claim()], {"result": table_artifact}, "继续核查原因。")
