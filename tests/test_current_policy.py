"""Resumed live runs must send the current policy, not an obsolete stored prompt."""

import copy
import types

import pytest

from local_data_agent import agent_loop
from local_data_agent.agent_loop import SYSTEM_PROMPT, current_policy_history, run_loop
from local_data_agent.context_budget import check_history


def _full_history():
    return [
        {"role": "system", "content": "OLD"},
        {"role": "user", "content": "cash net"},
        {"type": "function_call", "call_id": "c1", "name": "search_context", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "{}"},
        {"type": "function_call", "call_id": "c2", "name": "inspect_schema", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c2", "output": "{}"},
        {"role": "user", "content": "why did it change"},
    ]


def test_provider_input_uses_current_policy_without_touching_original():
    original = _full_history()
    snapshot = copy.deepcopy(original)

    observed = current_policy_history(original)

    # Fresh provider input carries the current policy...
    assert observed[0]["role"] == "system"
    assert observed[0]["content"] == SYSTEM_PROMPT
    # ...while the persisted original stays exactly as archived.
    assert original == snapshot
    assert original[0] == {"role": "system", "content": "OLD"}
    assert observed is not original
    assert observed[0] is not original[0]

    # Every user message and every function_call/function_call_output pair is
    # unchanged and still in order, matched by call_id.
    observed_users = [item for item in observed if item.get("role") == "user"]
    original_users = [item for item in original if item.get("role") == "user"]
    assert observed_users == original_users
    assert [item.get("content") for item in observed_users] == ["cash net", "why did it change"]

    observed_pairs = [
        (item["type"], item["call_id"])
        for item in observed
        if item.get("type") in {"function_call", "function_call_output"}
    ]
    original_pairs = [
        (item["type"], item["call_id"])
        for item in original
        if item.get("type") in {"function_call", "function_call_output"}
    ]
    assert observed_pairs == original_pairs
    assert observed_pairs == [
        ("function_call", "c1"),
        ("function_call_output", "c1"),
        ("function_call", "c2"),
        ("function_call_output", "c2"),
    ]

    # Metadata on the replaced system item is retained alongside new content.
    stored = [{"role": "system", "content": "OLD", "policy_version": 3}, {"role": "user", "content": "hi"}]
    replaced = current_policy_history(stored)
    assert replaced[0] == {"role": "system", "content": SYSTEM_PROMPT, "policy_version": 3}
    assert stored[0] == {"role": "system", "content": "OLD", "policy_version": 3}


@pytest.mark.parametrize(
    "history",
    [
        [],
        [{"role": "user", "content": "cash net"}],
        [{"type": "function_call", "call_id": "c1", "name": "search_context", "arguments": "{}"}],
    ],
)
def test_helper_inserts_system_message_when_absent(history):
    snapshot = copy.deepcopy(history)

    observed = current_policy_history(history)

    assert observed[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert observed[1:] == history
    assert len(observed) == len(history) + 1
    assert history == snapshot


class _FakeStore:
    def __init__(self):
        self.saved = []

    def save(self, state, event, payload=None):
        self.saved.append((event, payload))

    def start_call(self, *args, **kwargs):
        return None

    def finish_call(self, *args, **kwargs):
        return None


class _FakeModel:
    def __init__(self):
        self.requests = []
        self.response = {
            "status": "completed",
            "tool_calls": [],
            "protocol_items": [],
            "usage": {},
        }

    def respond(self, history, tools):
        self.requests.append(history)
        return dict(self.response)


def _make_state():
    return {
        "history": [
            {"role": "system", "content": "OLD"},
            {"role": "user", "content": "cash net"},
        ],
        "model_requests": 0,
        "usage": [],
        "status": "RUNNING",
        "run_id": "test",
    }


def test_run_loop_sends_current_policy_and_measures_updated_history():
    state = _make_state()
    store = _FakeStore()
    model = _FakeModel()
    context = types.SimpleNamespace(state=state, store=store, model=model)

    run_loop(context, max_requests=1)

    assert len(model.requests) == 1
    provider_history = model.requests[0]
    assert provider_history[0] == {"role": "system", "content": SYSTEM_PROMPT}

    # The persisted archived history still holds the obsolete prompt.
    assert state["history"][0] == {"role": "system", "content": "OLD"}
    assert state["history"][1] == {"role": "user", "content": "cash net"}

    # Budget is measured on the actually prepared, policy-updated history.
    assert state["context_budget"]["estimate"] == check_history(provider_history)["estimate"]

    assert state["model_requests"] == 1
