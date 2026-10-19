"""Attribute execute_step failures to the step that actually failed.

The plan executor resolves missing upstream steps recursively, so an error
raised while building an upstream LOAD/SELECT/RUN can surface at the requested
downstream step. This module annotates the failing step nearest to the fault
without exposing SQL, parameters, or file paths in the metadata.
"""

import functools

CAUGHT_EXECUTION_ERRORS = (ValueError, RuntimeError, KeyError, TypeError, OSError)

MESSAGE_LIMIT = 2000

_FAILED_STEP_KEYS = ("step_id", "type", "output_name")

_OUTPUT_KEY_LIMIT = 64


class StepExecutionError(ValueError):
    """Execution failure annotated with the step that actually failed."""

    def __init__(self, step, created, error):
        super().__init__(str(error)[:MESSAGE_LIMIT])
        self.cause_type = type(error).__name__
        failed_step = {key: step.get(key) for key in _FAILED_STEP_KEYS}
        if step.get("type") == "LOAD":
            failed_step["source_id"] = step.get("source_id")
            failed_step["table"] = step.get("table")
        elif step.get("type") == "RUN" and "operator" in step:
            failed_step["operator"] = step.get("operator")
        self.failed_step = failed_step
        self.executed_outputs = list(created)[:_OUTPUT_KEY_LIMIT]


def attribute_step_failure(method):
    """Wrap an execute(step_id, created) method with innermost attribution."""

    @functools.wraps(method)
    def wrapper(self, step_id, created):
        try:
            return method(self, step_id, created)
        except StepExecutionError:
            # The innermost attribution must survive every parent frame.
            raise
        except CAUGHT_EXECUTION_ERRORS as error:
            steps = {step["step_id"]: step for step in self.state["plan"]}
            if step_id not in steps:
                raise
            raise StepExecutionError(steps[step_id], created, error) from None

    return wrapper
