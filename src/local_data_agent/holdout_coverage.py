"""Holdout fact-coverage checking (evaluator-only).

This module is never invoked by model-facing tools. It is used by the
evaluation harness to decide whether a published ``claims.json``-shaped list of
claims covers a list of required facts exactly.

Design rules (see accompanying tests):

* Required facts have EXACT keys: ``column``, ``filter``, ``value``,
  ``tolerance``.
* Scopes are compared by exact whole-filter equality with TYPE-STRICT scalar
  comparison (``True != 1``, ``1 != 1.0``).  Filters are canonicalized with
  ``json.dumps`` of sorted ``[key, tag]`` pairs, so arbitrary string values
  (including control characters) cannot collide.
* Money columns (``*_cents``) and counts (``*_count``) are integers with
  tolerance 0, and their required ``value`` must be a real integer (``None``
  and ``bool`` are rejected).  ``change_pct`` and other numerics are finite
  numbers with ``tolerance <= 0.005001``.
* A claim with no ``value`` KEY is ``missing_value``; an explicit ``None``
  value is a valid payload that only matches a required ``None``.
* A claim whose filter is malformed and whose column is required is reported
  as wrong (``invalid_filter``) and can never satisfy the requirement.
* A scope matched by more than one claim is ambiguous and wrong.
* Wrong values can never be masked by a correct duplicate claim.
* Diagnostics are JSON-safe: non-finite actuals are reported as strings, so
  ``json.dumps(result, allow_nan=False)`` always succeeds.
"""

from __future__ import annotations

import json
import math
from typing import Any

__all__ = ["check_fact_coverage"]

_REQUIRED_KEYS = frozenset({"column", "filter", "value", "tolerance"})
_MAX_TOLERANCE = 0.005001
_AGGREGATIONS = frozenset({"single_row", "sum"})


# --------------------------------------------------------------------------
# scalar helpers
# --------------------------------------------------------------------------
def _is_bool(x: Any) -> bool:
    return isinstance(x, bool)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_finite_number(x: Any) -> bool:
    if not _is_number(x):
        return False
    if isinstance(x, float):
        return math.isfinite(x)
    return True


def _type_strict_equal(a: Any, b: Any) -> bool:
    """Equality that distinguishes bool/int/float and int vs float."""
    if _is_bool(a) or _is_bool(b):
        return _is_bool(a) and _is_bool(b) and a == b
    if _is_int(a) and _is_int(b):
        return a == b
    if isinstance(a, float) and isinstance(b, float):
        return a == b
    if isinstance(a, int) and isinstance(b, float):
        return False
    if isinstance(a, float) and isinstance(b, int):
        return False
    if a is None or b is None:
        return a is None and b is None
    return type(a) is type(b) and a == b


def _canonical_filter(filt: Any, where: str) -> str:
    """Serialize a filter dict canonically; reject non-dict/nested/nonfinite.

    The result is a JSON array of ``[key, scalar_tag]`` pairs, so arbitrary
    string values (including control characters) cannot make two distinct
    filters collide.
    """
    if not isinstance(filt, dict):
        raise ValueError(f"{where}: filter must be a dict, got {type(filt).__name__}")
    parts: list[list[str]] = []
    for key in sorted(filt.keys(), key=lambda k: (str(type(k)), str(k))):
        if not isinstance(key, str):
            raise ValueError(f"{where}: filter keys must be strings")
        val = filt[key]
        if isinstance(val, (dict, list, tuple, set)):
            raise ValueError(f"{where}: filter value for {key!r} must be a scalar")
        if isinstance(val, float) and not math.isfinite(val):
            raise ValueError(f"{where}: filter value for {key!r} must be finite")
        if val is not None and not isinstance(val, (str, int, float, bool)):
            raise ValueError(f"{where}: filter value for {key!r} must be a scalar")
        parts.append([key, _scalar_tag(val)])
    return json.dumps(parts, ensure_ascii=True)


def _scalar_tag(val: Any) -> str:
    """Type-strict textual tag for a scalar filter value."""
    if val is None:
        return "N"
    if _is_bool(val):
        return f"B:{'1' if val else '0'}"
    if _is_int(val):
        return f"I:{val}"
    if isinstance(val, float):
        return f"F:{val.hex()}"
    return f"S:{val}"


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def _kind_of(column: str) -> str:
    if column.endswith("_cents"):
        return "money"
    if column.endswith("_count"):
        return "count"
    return "numeric"


def _validate_required(required: Any) -> list[dict]:
    if not isinstance(required, list):
        raise ValueError("required must be a list of required-fact dicts")
    if not required:
        raise ValueError("required must be non-empty")

    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for idx, item in enumerate(required):
        where = f"required[{idx}]"
        if not isinstance(item, dict):
            raise ValueError(f"{where}: must be a dict")
        keys = set(item.keys())
        if keys != set(_REQUIRED_KEYS):
            missing = sorted(_REQUIRED_KEYS - keys)
            extra = sorted(k for k in keys if k not in _REQUIRED_KEYS)
            raise ValueError(
                f"{where}: exact keys required; missing={missing} extra={extra}"
            )

        column = item["column"]
        if not isinstance(column, str) or not column.strip():
            raise ValueError(f"{where}: column must be a non-empty string")

        canonical = _canonical_filter(item["filter"], where)

        key = (column, canonical)
        if key in seen:
            raise ValueError(f"{where}: duplicate required (column, filter) key")
        seen.add(key)

        kind = _kind_of(column)
        value = item["value"]
        tolerance = item["tolerance"]

        if not _is_number(tolerance):
            raise ValueError(f"{where}: tolerance must be a number")
        if isinstance(tolerance, float) and not math.isfinite(tolerance):
            raise ValueError(f"{where}: tolerance must be finite")
        if tolerance < 0:
            raise ValueError(f"{where}: tolerance must be >= 0")

        if kind in ("money", "count"):
            if tolerance != 0:
                raise ValueError(f"{where}: {kind} tolerance must be 0")
            # Money/count required values are mandatory integers; None is not
            # an acceptable expected value for these kinds.
            if not _is_int(value):
                raise ValueError(
                    f"{where}: {kind} value must be an integer"
                )
        else:
            if tolerance > _MAX_TOLERANCE:
                raise ValueError(
                    f"{where}: numeric tolerance must be <= {_MAX_TOLERANCE}"
                )
            if value is not None and not _is_finite_number(value):
                raise ValueError(
                    f"{where}: numeric value must be a finite number or None"
                )

        out.append(
            {
                "index": idx,
                "column": column,
                "filter": item["filter"],
                "value": value,
                "tolerance": tolerance,
                "kind": kind,
                "key": key,
            }
        )
    return out


def _validate_claims(claims: Any) -> list[dict]:
    if not isinstance(claims, list):
        raise ValueError("claims must be a list of claim dicts")
    out = []
    for idx, claim in enumerate(claims):
        if not isinstance(claim, dict):
            raise ValueError(f"claims[{idx}]: must be a dict")
        out.append(claim)
    return out


# --------------------------------------------------------------------------
# value comparison
# --------------------------------------------------------------------------
def _expected_unit(column: str) -> str:
    return "CNY_yuan" if column.endswith("_cents") else "raw"


def _canonical_scope_key(column: str, filt: Any, where: str) -> str:
    """JSON-friendly diagnostic key for a (column, filter) scope."""
    return json.dumps([column, _canonical_filter(filt, where)], ensure_ascii=True)


def _json_safe(value: Any) -> Any:
    """Return a value safe for ``json.dumps(..., allow_nan=False)``.

    Non-finite floats (NaN/Infinity) are not valid JSON, so they are reported
    as their ``repr`` string instead of being emitted as a bare non-finite
    number.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def _claim_value_problem(claim: dict, column: str) -> str | None:
    """Return a reason string if the claim's own shape/type is malformed.

    A missing ``value`` KEY is distinguished from an explicitly present
    ``None``: the former is ``missing_value``, while an explicit ``None`` is a
    syntactically valid claim payload that is later compared to the required
    value by :func:`_values_match` (``None`` matches only ``None``).
    """
    agg = claim.get("aggregation")
    if not isinstance(agg, str) or agg not in _AGGREGATIONS:
        return f"invalid_aggregation:{agg!r}"

    if not isinstance(claim.get("column"), str) or claim.get("column") != column:
        return "column_mismatch"

    if _expected_unit(column) != claim.get("display_unit"):
        return f"wrong_display_unit:{claim.get('display_unit')!r}"

    if not isinstance(claim.get("artifact_id"), str) or not claim["artifact_id"].strip():
        return "missing_or_blank_artifact_id"

    if "value" not in claim:
        return "missing_value"

    value = claim["value"]
    if value is None:
        # Explicit None is allowed for numeric kinds only; money/count must be
        # integers and can never be None.
        if _kind_of(column) in ("money", "count"):
            return "malformed_value:NoneType"
        return None

    kind = _kind_of(column)
    if kind in ("money", "count"):
        if not _is_int(value):
            return f"malformed_value:{type(value).__name__}"
    else:
        if not _is_finite_number(value):
            return f"malformed_value:{_json_safe(value)!r}"
    return None


def _values_match(expected: Any, actual: Any, tolerance: Any, kind: str) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    if kind in ("money", "count"):
        return expected == actual
    if not _is_finite_number(actual):
        return False
    return abs(actual - expected) <= tolerance


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def check_fact_coverage(claims: list, required: list) -> dict:
    """Check whether ``claims`` exactly cover ``required`` facts.

    Returns ``{passed, expected_facts, matched_facts, matched, missing, wrong}``.
    ``passed`` is True only when every required fact is matched exactly once by
    a well-formed claim and nothing is missing or wrong.
    """
    reqs = _validate_required(required)
    claim_list = _validate_claims(claims)

    # Bucket claims by internal (column, canonical_filter) scope key.  A claim
    # whose filter is malformed cannot be scoped by equality; if its column is
    # required it is recorded as an ``invalid_filter`` diagnostic, and such a
    # claim can never be used to satisfy the requirement.
    buckets: dict[tuple[str, str], list[tuple[int, dict]]] = {}
    malformed_filter_claims: dict[str, list[tuple[int, dict]]] = {}
    for cidx, claim in enumerate(claim_list):
        column = claim.get("column")
        if not isinstance(column, str):
            continue
        try:
            canonical = _canonical_filter(claim.get("filter"), f"claims[{cidx}]")
        except ValueError:
            malformed_filter_claims.setdefault(column, []).append((cidx, claim))
            continue
        buckets.setdefault((column, canonical), []).append((cidx, claim))

    matched: list[dict] = []
    missing: list[dict] = []
    wrong: list[dict] = []
    matched_facts = 0

    for req in reqs:
        diagnostics = {
            "required_index": req["index"],
            "column": req["column"],
            "filter": req["filter"],
            "value": req["value"],
        }
        candidates = buckets.get(req["key"], [])
        bad_filters = malformed_filter_claims.get(req["column"], [])

        usable: list[tuple[int, dict]] = []
        malformed: list[tuple[int, dict, str]] = []
        for cidx, claim in candidates:
            claim_id = claim.get("claim_id")
            has_id = isinstance(claim_id, str) and claim_id.strip() != ""
            problem = _claim_value_problem(claim, req["column"])
            if problem is not None or not has_id:
                if not has_id:
                    problem = problem or "missing_claim_id"
                malformed.append((cidx, claim, problem or "malformed_claim"))
            else:
                usable.append((cidx, claim))

        # Malformed-filter claims on a required column are always wrong; they
        # can never produce a pass even when another good claim exists.
        for cidx, claim in bad_filters:
            wrong.append(
                dict(
                    diagnostics,
                    scope_key=_canonical_scope_key(
                        req["column"], req["filter"], "required"
                    ),
                    claim_id=claim.get("claim_id"),
                    claim_index=cidx,
                    actual=_json_safe(claim.get("value")),
                    reason="invalid_filter",
                )
            )

        # Ambiguity: more than one claim binds the same required scope.
        if len(candidates) > 1:
            matched_facts += 0
            wrong.append(
                dict(
                    diagnostics,
                    reason="ambiguous_duplicate_scope",
                    claim_ids=[
                        c.get("claim_id") for _, c in candidates
                    ],
                    candidate_count=len(candidates),
                )
            )
            continue

        for cidx, claim, problem in malformed:
            wrong.append(
                dict(
                    diagnostics,
                    claim_id=claim.get("claim_id"),
                    claim_index=cidx,
                    actual=_json_safe(claim.get("value")),
                    reason=problem,
                )
            )

        if not usable:
            if not candidates and not bad_filters:
                missing.append(
                    dict(
                        diagnostics,
                        reason="no_matching_claim",
                        scope_key=_canonical_scope_key(
                            req["column"], req["filter"], "required"
                        ),
                    )
                )
            # If candidates (with a well-formed filter, or a malformed filter on
            # this required column) existed they were recorded as wrong; a
            # malformed claim must not be reported as merely missing.
            continue

        cidx, claim = usable[0]
        actual = claim.get("value")
        ok = _values_match(req["value"], actual, req["tolerance"], req["kind"])
        if ok:
            matched_facts += 1
            matched.append(
                dict(
                    diagnostics,
                    claim_id=claim.get("claim_id"),
                    claim_index=cidx,
                    actual=_json_safe(actual),
                    tolerance=req["tolerance"],
                    reason="exact_match",
                )
            )
        else:
            wrong.append(
                dict(
                    diagnostics,
                    claim_id=claim.get("claim_id"),
                    claim_index=cidx,
                    actual=_json_safe(actual),
                    tolerance=req["tolerance"],
                    reason="value_mismatch",
                )
            )

    passed = (
        matched_facts == len(reqs)
        and not missing
        and not wrong
        and len(matched) == len(reqs)
    )

    return {
        "passed": passed,
        "expected_facts": len(reqs),
        "matched_facts": matched_facts,
        "matched": matched,
        "missing": missing,
        "wrong": wrong,
    }
