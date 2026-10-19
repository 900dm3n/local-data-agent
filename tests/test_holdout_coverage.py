"""Tests for local_data_agent.holdout_coverage (evaluator-only)."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_data_agent.holdout_coverage import check_fact_coverage  # noqa: E402


# --------------------------------------------------------------------------
# helpers -- all numbers are fictional and tiny
# --------------------------------------------------------------------------
def req(column, filt, value, tolerance=0.0):
    return {"column": column, "filter": filt, "value": value, "tolerance": tolerance}


def claim(claim_id, column, filt, value, *, agg="sum", unit=None, artifact_id="art-1"):
    return {
        "claim_id": claim_id,
        "artifact_id": artifact_id,
        "column": column,
        "filter": filt,
        "aggregation": agg,
        "value": value,
        "display_unit": unit if unit is not None else (
            "CNY_yuan" if column.endswith("_cents") else "raw"
        ),
        "label": f"fictional {column}",
    }


def reasons(diags):
    return [d["reason"] for d in diags]


# --------------------------------------------------------------------------
# happy paths
# --------------------------------------------------------------------------
def test_correct_shuffled_publication_with_unrelated_extras():
    required = [
        req("net_revenue_cents", {"channel": "all"}, 111),
        req("order_count", {"channel": "web"}, 7),
        req("refund_rate", {"channel": "web", "period": "current"}, 0.25, 0.0005),
        req("discount_rate", {"channel": "app", "region": "north"}, 1.5, 0.005001),
    ]
    claims = [
        # shuffled order, unrelated extra columns/filters included
        claim("c4", "basket_size", {"channel": "app"}, 3),
        claim("c3", "discount_rate", {"region": "north", "channel": "app"}, 1.5009),
        claim("c1", "net_revenue_cents", {"channel": "all"}, 111),
        claim("c9", "order_count", {"channel": "app"}, 4),
        claim("c2", "order_count", {"channel": "web"}, 7),
        claim("c8", "refund_rate", {"period": "prior", "channel": "web"}, 0.25),
        claim("c5", "refund_rate", {"channel": "web", "period": "current"}, 0.2504),
    ]
    snap_claims, snap_required = copy.deepcopy(claims), copy.deepcopy(required)
    result = check_fact_coverage(claims, required)
    assert result["passed"] is True
    assert result["expected_facts"] == 4
    assert result["matched_facts"] == 4
    assert result["missing"] == []
    assert result["wrong"] == []
    assert [m["required_index"] for m in result["matched"]] == [0, 1, 2, 3]
    assert claims == snap_claims and required == snap_required


def test_all_channel_total_never_satisfies_channel_filtered_requirement():
    required = [req("order_count", {"channel": "web"}, 5)]
    claims = [claim("c1", "order_count", {"channel": "all"}, 5)]
    result = check_fact_coverage(claims, required)
    assert result["passed"] is False
    assert result["missing"][0]["reason"] == "no_matching_claim"
    assert result["wrong"] == []


def test_period_filter_mismatch_is_not_a_match():
    required = [req("refund_rate", {"channel": "web", "period": "current"}, 0.4, 0.001)]
    claims = [claim("c1", "refund_rate", {"channel": "web", "period": "prior"}, 0.4)]
    result = check_fact_coverage(claims, required)
    assert result["passed"] is False
    assert result["matched_facts"] == 0
    assert len(result["missing"]) == 1


# --------------------------------------------------------------------------
# duplicates / ambiguity / masking
# --------------------------------------------------------------------------
def test_wrong_duplicate_correct_claim_cannot_be_masked():
    required = [req("order_count", {"channel": "web"}, 5)]
    claims = [
        claim("wrong", "order_count", {"channel": "web"}, 4),
        claim("right", "order_count", {"channel": "web"}, 5),
    ]
    result = check_fact_coverage(claims, required)
    assert result["passed"] is False
    assert result["matched_facts"] == 0
    assert reasons(result["wrong"]) == ["ambiguous_duplicate_scope"]


def test_identical_duplicates_are_ambiguous_and_wrong():
    required = [req("order_count", {"channel": "web"}, 5)]
    claims = [
        claim("a", "order_count", {"channel": "web"}, 5),
        claim("b", "order_count", {"channel": "web"}, 5),
    ]
    result = check_fact_coverage(claims, required)
    assert result["passed"] is False
    assert result["wrong"][0]["reason"] == "ambiguous_duplicate_scope"


def test_same_claim_id_reused_across_unrelated_scopes_is_allowed():
    required = [
        req("order_count", {"channel": "web"}, 5),
        req("order_count", {"channel": "app"}, 6),
    ]
    claims = [
        claim("dup-id", "order_count", {"channel": "web"}, 5),
        claim("dup-id", "order_count", {"channel": "app"}, 6),
    ]
    result = check_fact_coverage(claims, required)
    assert result["passed"] is True


# --------------------------------------------------------------------------
# type strictness
# --------------------------------------------------------------------------
def test_bool_is_not_an_integer_money_value():
    required = [req("net_revenue_cents", {"channel": "all"}, 12)]
    claims = [claim("c1", "net_revenue_cents", {"channel": "all"}, True)]
    result = check_fact_coverage(claims, required)
    assert result["passed"] is False
    assert reasons(result["wrong"]) == ["malformed_value:bool"]


def test_float_money_value_is_rejected():
    required = [req("net_revenue_cents", {"channel": "all"}, 12)]
    claims = [claim("c1", "net_revenue_cents", {"channel": "all"}, 12.0)]
    result = check_fact_coverage(claims, required)
    assert result["passed"] is False
    assert result["wrong"][0]["reason"] == "malformed_value:float"


def test_filter_type_strictness_int_vs_float_and_bool_vs_int():
    required = [req("order_count", {"store": 1}, 3)]
    claims = [claim("c1", "order_count", {"store": 1.0}, 3)]
    result = check_fact_coverage(claims, required)
    assert result["passed"] is False
    assert len(result["missing"]) == 1

    required_b = [req("order_count", {"flagged": True}, 3)]
    claims_b = [claim("c1", "order_count", {"flagged": 1}, 3)]
    result_b = check_fact_coverage(claims_b, required_b)
    assert result_b["passed"] is False
    assert len(result_b["missing"]) == 1


def test_huge_integer_exactness_beyond_float_precision():
    big = 2**60 + 1
    required = [req("gross_revenue_cents", {"channel": "all"}, big)]
    good = [claim("c1", "gross_revenue_cents", {"channel": "all"}, big)]
    bad = [claim("c1", "gross_revenue_cents", {"channel": "all"}, float(big))]

    assert float(big) == float(2**60)
    assert check_fact_coverage(good, required)["passed"] is True
    result = check_fact_coverage(bad, required)
    assert result["passed"] is False
    assert result["wrong"][0]["reason"] == "malformed_value:float"


# --------------------------------------------------------------------------
# tolerance and rate rounding
# --------------------------------------------------------------------------
def test_rate_tolerance_boundary_inclusive():
    required = [req("refund_rate", {"channel": "web"}, 0.25, 0.005001)]
    ok = [claim("c1", "refund_rate", {"channel": "web"}, 0.255001)]
    out = [claim("c1", "refund_rate", {"channel": "web"}, 0.255002)]
    assert check_fact_coverage(ok, required)["passed"] is True
    assert check_fact_coverage(out, required)["passed"] is False


def test_required_tolerance_above_cap_rejected():
    with pytest.raises(ValueError):
        check_fact_coverage([], [req("refund_rate", {"channel": "web"}, 0.25, 0.005002)])


# --------------------------------------------------------------------------
# None semantics
# --------------------------------------------------------------------------
def test_none_expected_does_not_match_zero():
    required = [req("refund_rate", {"channel": "web"}, None, 0.0)]
    claims = [claim("c1", "refund_rate", {"channel": "web"}, 0)]
    result = check_fact_coverage(claims, required)
    assert result["passed"] is False
    assert result["wrong"][0]["reason"] == "value_mismatch"


def test_none_expected_matches_none_actual():
    required = [req("refund_rate", {"channel": "web"}, None, 0.0)]
    claims = [claim("c1", "refund_rate", {"channel": "web"}, None)]
    result = check_fact_coverage(claims, required)
    assert result["passed"] is True


def test_zero_expected_does_not_match_none_actual():
    required = [req("refund_rate", {"channel": "web"}, 0.0, 0.0)]
    claims = [claim("c1", "refund_rate", {"channel": "web"}, None)]
    result = check_fact_coverage(claims, required)
    assert result["passed"] is False
    assert result["wrong"][0]["reason"] == "value_mismatch"


# --------------------------------------------------------------------------
# non-finite and malformed claim payloads
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_claimed_values_rejected(bad):
    required = [req("refund_rate", {"channel": "web"}, 0.25, 0.001)]
    claims = [claim("c1", "refund_rate", {"channel": "web"}, bad)]
    result = check_fact_coverage(claims, required)
    assert result["passed"] is False
    assert result["wrong"][0]["reason"].startswith("malformed_value")


def test_control_character_filter_values_cannot_collide():
    """Distinct scopes must not collide via unescaped string delimiters."""
    required = [req("order_count", {"a": "x\x1fb\x1fS:y"}, 5)]
    claims = [claim("c1", "order_count", {"a": "x", "b": "y"}, 5)]
    result = check_fact_coverage(claims, required)
    assert result["passed"] is False
    assert result["matched_facts"] == 0
    assert len(result["missing"]) == 1
    assert result["wrong"] == []


def test_filter_values_int_bool_float_stay_distinct():
    """1, True and 1.0 are distinct filter values."""
    required = [req("order_count", {"store": 1}, 3)]
    assert check_fact_coverage(
        [claim("c1", "order_count", {"store": True}, 3)], required
    )["passed"] is False
    assert check_fact_coverage(
        [claim("c1", "order_count", {"store": 1.0}, 3)], required
    )["passed"] is False
    assert check_fact_coverage(
        [claim("c1", "order_count", {"store": 1}, 3)], required
    )["passed"] is True


def test_malformed_claim_filter_does_not_crash_and_is_not_missing_match():
    required = [req("order_count", {"channel": "web"}, 5)]
    claims = [
        {
            "claim_id": "c1",
            "artifact_id": "art-1",
            "column": "order_count",
            "filter": {"channel": ["web"]},  # nested -> unusable
            "aggregation": "sum",
            "value": 5,
            "display_unit": "raw",
        }
    ]
    result = check_fact_coverage(claims, required)
    assert result["passed"] is False
    assert result['matched_facts'] == 0
    assert result['wrong'][0]['reason'] == 'invalid_filter'


def test_malformed_filter_claim_cannot_pass_with_correct_claim_same_column():
    """A wrong malformed-filter claim blocks a pass even beside a good claim."""
    required = [req("order_count", {"channel": "web"}, 5)]
    claims = [
        {
            "claim_id": "bad",
            "artifact_id": "art-1",
            "column": "order_count",
            "filter": {"channel": ["web"]},  # malformed -> invalid_filter
            "aggregation": "sum",
            "value": 5,
            "display_unit": "raw",
        },
        claim("good", "order_count", {"channel": "web"}, 5),
    ]
    result = check_fact_coverage(claims, required)
    assert result["passed"] is False
    assert "invalid_filter" in reasons(result["wrong"])
    import json

    json.dumps(result, allow_nan=False)


def test_malformed_extra_column_filter_is_ignored():
    """A malformed filter on an unrelated column is not a diagnostic."""
    required = [req("order_count", {"channel": "web"}, 5)]
    claims = [
        claim("good", "order_count", {"channel": "web"}, 5),
        {
            "claim_id": "extra",
            "artifact_id": "art-1",
            "column": "basket_size",
            "filter": {"channel": ["app"]},
            "aggregation": "sum",
            "value": 3,
            "display_unit": "raw",
        },
    ]
    result = check_fact_coverage(claims, required)
    assert result["passed"] is True
    assert result["wrong"] == []


def test_missing_artifact_binding_is_wrong():
    for bad_artifact in (None, "", "   "):
        required = [req("order_count", {"channel": "web"}, 5)]
        claims = [claim("c1", "order_count", {"channel": "web"}, 5, artifact_id=bad_artifact)]
        result = check_fact_coverage(claims, required)
        assert result["passed"] is False
        assert result["wrong"][0]["reason"] == "missing_or_blank_artifact_id"


def test_wrong_display_unit_and_invalid_aggregation_rejected():
    required = [req("net_revenue_cents", {"channel": "all"}, 99)]

    unit_claims = [claim("c1", "net_revenue_cents", {"channel": "all"}, 99, unit="raw")]
    res = check_fact_coverage(unit_claims, required)
    assert res["passed"] is False
    assert res["wrong"][0]["reason"].startswith("wrong_display_unit")

    agg_claims = [claim("c1", "net_revenue_cents", {"channel": "all"}, 99, agg="average")]
    res = check_fact_coverage(agg_claims, required)
    assert res["passed"] is False
    assert res["wrong"][0]["reason"].startswith("invalid_aggregation")


def test_missing_value_and_missing_claim_id_rejected():
    required = [req("order_count", {"channel": "web"}, 5)]
    no_value = [claim("c1", "order_count", {"channel": "web"}, None)]
    no_value[0].pop('value')
    res = check_fact_coverage(no_value, required)
    assert res["passed"] is False
    assert res["wrong"][0]["reason"] == "missing_value"

    blank_id = [claim("  ", "order_count", {"channel": "web"}, 5)]
    res = check_fact_coverage(blank_id, required)
    assert res["passed"] is False
    assert res["wrong"][0]["reason"] == "missing_claim_id"


def test_malformed_claim_beside_correct_duplicate_is_still_wrong():
    required = [req("order_count", {"channel": "web"}, 5)]
    claims = [
        claim("bad", "order_count", {"channel": "web"}, 5, artifact_id=None),
        claim("good", "order_count", {"channel": "web"}, 5),
    ]
    result = check_fact_coverage(claims, required)
    assert result["passed"] is False
    assert result["wrong"][0]["reason"] == "ambiguous_duplicate_scope"


def test_malformed_claims_container_raises():
    with pytest.raises(ValueError):
        check_fact_coverage({"not": "a list"}, [req("order_count", {"channel": "web"}, 5)])
    with pytest.raises(ValueError):
        check_fact_coverage([["not a dict"]], [req("order_count", {"channel": "web"}, 5)])


# --------------------------------------------------------------------------
# required validation
# --------------------------------------------------------------------------
def test_required_must_be_nonempty_list():
    with pytest.raises(ValueError):
        check_fact_coverage([], [])
    with pytest.raises(ValueError):
        check_fact_coverage([], "nope")


def test_required_exact_keys_enforced():
    with pytest.raises(ValueError):
        check_fact_coverage([], [{"column": "a", "filter": {}, "value": 1}])
    with pytest.raises(ValueError):
        check_fact_coverage(
            [], [{"column": "a", "filter": {}, "value": 1, "tolerance": 0, "extra": 1}]
        )


def test_required_duplicate_scope_rejected():
    dup = [
        req("order_count", {"channel": "web"}, 5),
        req("order_count", {"channel": "web"}, 6),
    ]
    with pytest.raises(ValueError):
        check_fact_coverage([], dup)


def test_required_duplicate_scope_with_type_distinct_filter_is_allowed():
    distinct = [
        req("order_count", {"store": 1}, 5),
        req("order_count", {"store": 1.0}, 6),
    ]
    result = check_fact_coverage([], distinct)
    assert result["passed"] is False
    assert len(result["missing"]) == 2


def test_required_column_and_tolerance_types_enforced():
    with pytest.raises(ValueError):
        check_fact_coverage([], [req("", {}, 1)])
    with pytest.raises(ValueError):
        check_fact_coverage([], [req("order_count", {}, 1, True)])  # bool tolerance
    with pytest.raises(ValueError):
        check_fact_coverage([], [req("order_count", {}, 1, -0.1)])
    with pytest.raises(ValueError):
        check_fact_coverage([], [req("order_count", {}, 1.5)])  # count needs int
    with pytest.raises(ValueError):
        check_fact_coverage([], [req("net_revenue_cents", {}, 1, 0.001)])  # money tol != 0
    with pytest.raises(ValueError):
        check_fact_coverage([], [req("refund_rate", {}, float("nan"), 0.001)])
    with pytest.raises(ValueError):
        check_fact_coverage([], [req("refund_rate", {}, True, 0.001)])
    with pytest.raises(ValueError):
        check_fact_coverage([], [req("refund_rate", {}, 0.5, float("inf"))])


def test_required_money_None_rejected():
    """Money/count required values are mandatory integers; None is rejected."""
    with pytest.raises(ValueError):
        check_fact_coverage([], [req("net_revenue_cents", {"channel": "all"}, None)])
    with pytest.raises(ValueError):
        check_fact_coverage([], [req("order_count", {"channel": "web"}, None)])


def test_required_filter_must_be_flat_scalar_dict():
    with pytest.raises(ValueError):
        check_fact_coverage([], [req("order_count", "web", 1)])
    with pytest.raises(ValueError):
        check_fact_coverage([], [req("order_count", {"a": {"b": 1}}, 1)])
    with pytest.raises(ValueError):
        check_fact_coverage([], [req("order_count", {"a": float("nan")}, 1)])


# --------------------------------------------------------------------------
# no mutation / diagnostics shape
# --------------------------------------------------------------------------
def test_inputs_are_not_mutated():
    required = [
        req("net_revenue_cents", {"channel": "all"}, 42),
        req("order_count", {"channel": "app"}, 3),
    ]
    claims = [
        claim("c1", "net_revenue_cents", {"channel": "all"}, 42),
        claim("c2", "order_count", {"channel": "app"}, 4),
    ]
    r_snap = copy.deepcopy(required)
    c_snap = copy.deepcopy(claims)
    result = check_fact_coverage(claims, required)
    assert required == r_snap
    assert claims == c_snap
    assert result["passed"] is False

    import json

    json.dumps(result, allow_nan=False)


def test_diagnostics_carry_required_scope_fields():
    required = [req("order_count", {"channel": "web"}, 5)]
    result = check_fact_coverage([], required)
    entry = result["missing"][0]
    for field in ("required_index", "column", "filter", "value"):
        assert field in entry
    assert entry["required_index"] == 0
    assert entry["column"] == "order_count"
    assert entry["filter"] == {"channel": "web"}

    wrong = check_fact_coverage(
        [claim("c1", "order_count", {"channel": "web"}, 9)], required
    )["wrong"][0]
    assert wrong["claim_id"] == "c1"
    assert wrong["required_index"] == 0
    assert wrong["actual"] == 9
