"""Execution diagnostics: innermost step attribution and protocol output."""

import json
from copy import deepcopy

import pytest

from local_data_agent.agent_loop import SYSTEM_PROMPT, execute_calls
from local_data_agent.execution_diagnostics import StepExecutionError, attribute_step_failure

ROW_LIMIT_MESSAGE = "LOAD exceeded output row limit: 20000 rows above 10000 row limit"

PLAN = [
    {"step_id": "source_load", "type": "LOAD", "output_name": "src_tbl", "depends_on": [],
     "source_id": "src_orders", "table": "orders"},
    {"step_id": "other_load", "type": "LOAD", "output_name": "other_tbl", "depends_on": [],
     "source_id": "src_other", "table": "other"},
    {"step_id": "middle", "type": "SELECT", "output_name": "mid", "depends_on": ["source_load"],
     "sql": "SELECT * FROM src_tbl"},
    {"step_id": "final", "type": "SELECT", "output_name": "fin", "depends_on": ["middle"],
     "sql": "SELECT count(*) FROM mid"},
]


class FakeMachine:
    """Recursive execute-shaped subject that fails at the source LOAD."""

    def __init__(self, plan):
        self.state = {"plan": deepcopy(plan)}

    @attribute_step_failure
    def execute(self, step_id, created):
        steps = {step["step_id"]: step for step in self.state["plan"]}
        if step_id not in steps:
            raise ValueError("Unknown step_id")
        step = steps[step_id]
        for dependency in step["depends_on"]:
            self.execute(dependency, created)
        if step_id == "source_load":
            raise ValueError(ROW_LIMIT_MESSAGE)
        created.append(step["output_name"])
        return {"step_id": step_id, "output_name": step["output_name"]}


def test_innermost_source_attribution_and_original_cause():
    machine = FakeMachine(PLAN)
    created = []
    with pytest.raises(StepExecutionError) as caught:
        machine.execute("final", created)
    error = caught.value
    assert error.failed_step == {"step_id": "source_load", "type": "LOAD", "output_name": "src_tbl",
                                 "source_id": "src_orders", "table": "orders"}
    assert error.cause_type == "ValueError"
    assert str(error) == ROW_LIMIT_MESSAGE
    assert ROW_LIMIT_MESSAGE in str(error)


def test_not_attributed_to_parent_or_other_steps():
    machine = FakeMachine(PLAN)
    with pytest.raises(StepExecutionError) as caught:
        machine.execute("final", [])
    assert caught.value.failed_step["step_id"] not in {"final", "middle", "other_load"}


def test_created_prefix_copied_without_metadata_leaks():
    machine = FakeMachine(PLAN)
    created = ["pre_existing"]
    with pytest.raises(StepExecutionError) as caught:
        machine.execute("final", created)
    error = caught.value
    assert error.executed_outputs == ["pre_existing"]
    assert created == ["pre_existing"]
    error.executed_outputs.append("mutated")
    assert created == ["pre_existing"]
    serialized = json.dumps(error.failed_step) + json.dumps(error.executed_outputs) + str(error)
    for leak in ("SELECT", "params", ".parquet", "sql", "depends_on"):
        assert leak not in json.dumps(error.failed_step), leak
    assert set(error.failed_step) == {"step_id", "type", "output_name", "source_id", "table"}
    assert "SELECT" not in serialized


def test_executed_outputs_bounded():
    machine = FakeMachine(PLAN)
    created = ["o%d" % index for index in range(100)]
    with pytest.raises(StepExecutionError) as caught:
        machine.execute("final", created)
    assert len(caught.value.executed_outputs) == 64
    assert caught.value.executed_outputs == created[:64]


def test_parent_reraises_same_object():
    machine = FakeMachine(PLAN)
    errors = []
    for step_id in ("source_load", "middle", "final"):
        with pytest.raises(StepExecutionError) as caught:
            machine.execute(step_id, [])
        errors.append(caught.value)
    inner = [error.__dict__.copy() for error in errors]
    assert [item["failed_step"] for item in inner] == [errors[0].failed_step] * 3
    assert [item["cause_type"] for item in inner] == ["ValueError"] * 3
    assert errors[1].failed_step["step_id"] == errors[2].failed_step["step_id"] == "source_load"


def test_unknown_step_id_reraised_unchanged():
    machine = FakeMachine(PLAN)
    with pytest.raises(ValueError) as caught:
        machine.execute("missing_step", [])
    assert not isinstance(caught.value, StepExecutionError)
    assert str(caught.value) == "Unknown step_id"


def test_base_exception_not_caught():
    class Boom(BaseException):
        pass

    class Raising(FakeMachine):
        @attribute_step_failure
        def execute(self, step_id, created):
            raise Boom("interrupt")

    with pytest.raises(Boom):
        Raising(PLAN).execute("final", [])


class FakeStore:
    """start_call returns None once, then the stored output; finish_call deep-copies."""

    def __init__(self):
        self.calls = {}
        self.finished = []
        self.saved = []

    def start_call(self, run_id, call_id, name, arguments):
        stored = self.calls.get(call_id)
        return deepcopy(stored) if stored is not None else None

    def finish_call(self, run_id, call_id, output, state):
        self.calls[call_id] = deepcopy(output)
        self.finished.append(call_id)

    def save(self, state, event, payload):
        self.saved.append((event, deepcopy(payload)))


class FakeContext:
    def __init__(self, plan, error):
        self.state = {
            "run_id": "run_1",
            "status": "RUNNING",
            "history": [],
            "repair_counts": {},
            "plan": deepcopy(plan),
        }
        self.store = FakeStore()
        self.error = error
        self.dispatched = []

    def dispatch(self, name, arguments):
        self.dispatched.append((name, arguments))
        raise self.error


def _execute_call(step_id="final"):
    return {"call_id": "x", "name": "execute_step",
            "arguments": {"plan_version": 1, "step_id": step_id}}


def _diagnostics():
    source_error = ValueError(ROW_LIMIT_MESSAGE)
    step = PLAN[0]
    return StepExecutionError(step, ["mid_done"], source_error)


def test_execute_calls_preserves_diagnostics_and_error_code():
    error = _diagnostics()
    context = FakeContext(PLAN, error)
    execute_calls(context, [_execute_call()])

    history = context.state["history"]
    assert len(history) == 1
    output = json.loads(history[0]["output"])
    assert output["error"] == "ValueError"
    assert output["retryable"] is True
    assert "row limit" in output["message"]
    assert output["failed_step"] == {"step_id": "source_load", "type": "LOAD", "output_name": "src_tbl",
                                     "source_id": "src_orders", "table": "orders"}
    assert output["executed_outputs"] == ["mid_done"]
    assert context.state["repair_counts"] == {"final": 1}


def test_cached_replay_does_not_execute_again():
    error = _diagnostics()
    context = FakeContext(PLAN, error)
    call = _execute_call()
    execute_calls(context, [call])
    # The store deep-copied the initial exception metadata; tampering with the
    # live error object must not change the durable reply that is replayed.
    error.failed_step["step_id"] = "tampered"
    error.failed_step["type"] = "RUN"
    error.executed_outputs.append("tampered_output")
    context.state["history"] = []
    execute_calls(context, [call])
    replay = context.state["history"][0]["output"]
    assert json.loads(replay)["failed_step"]["step_id"] == "source_load"
    assert json.loads(replay)["failed_step"]["type"] == "LOAD"
    assert json.loads(replay)["executed_outputs"] == ["mid_done"]
    assert len(context.dispatched) == 1
    assert context.state["repair_counts"] == {"final": 1}


def test_system_prompt_mentions_failed_step_and_executed_outputs():
    assert "failed_step" in SYSTEM_PROMPT
    assert "executed_outputs" in SYSTEM_PROMPT
    assert "patch_plan" in SYSTEM_PROMPT
