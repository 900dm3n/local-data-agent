"""Independent evaluation boundaries: equivalent views, scope and budget."""
from pathlib import Path
import importlib.util
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('holdout_eval_coordinator', SCRIPTS/'evaluate_holdout.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def claim(column, filt, value):
    return dict(claim_id='c1', artifact_id='a1', column=column, filter=filt, value=value,
                aggregation='single_row', display_unit='CNY_yuan')


def req(column, filt, value):
    return dict(column=column, filter=filt, value=value, tolerance=0)


def test_equivalent_period_cell_covers_monthly_fact_without_changing_original():
    c = claim('base_net_cents', {'base_month':'2026-02','current_month':'2026-03','channel_code':'web'}, 123)
    facts = [req('net_cents', {'month':'2026-02','channel_code':'web'}, 123)]
    normalized, trace = module.normalize_presentation([c], facts)
    assert module.check_fact_coverage(normalized, facts)['passed']
    assert c['column'] == 'base_net_cents' and len(trace) == 1


def test_channel_scope_cannot_be_promoted_to_all_channels():
    c = claim('net_cents', {'month':'2026-02','channel_code':'web'}, 123)
    facts = [req('base_net_cents', {'base_month':'2026-02','current_month':'2026-03'}, 123)]
    normalized, trace = module.normalize_presentation([c], facts)
    assert not module.check_fact_coverage(normalized, facts)['passed'] and trace == []


def test_wrong_native_fact_cannot_be_masked_by_equivalent_alternate():
    native = claim('net_cents', {'month':'2026-02'}, 999)
    alternate = claim('base_net_cents', {'base_month':'2026-02','current_month':'2026-03'}, 123)
    facts = [req('net_cents', {'month':'2026-02'}, 123)]
    normalized, trace = module.normalize_presentation([native, alternate], facts)
    assert not module.check_fact_coverage(normalized, facts)['passed'] and trace == []


def test_global_budget_counts_actual_responses_and_prevents_next_request():
    class Fake:
        def __init__(self): self.calls = 0
        def respond(self, *args):
            self.calls += 1
            return {'usage': {'prompt_tokens':1000,'prompt_cache_hit_tokens':990,'prompt_cache_miss_tokens':10,'completion_tokens':2}}
    fake = Fake()
    budget = module.BudgetedClient(fake, max_requests=3, max_new_tokens=12)
    budget.respond([], [])
    assert budget.new_tokens == 12 and budget.stopped and fake.calls == 1
    with pytest.raises(module.EvaluationBudgetExceeded): budget.respond([], [])
    assert fake.calls == 1


def test_oracle_fills_missing_channel_period_and_undefined_rate():
    ledger = [{'month':'2026-03','channel_code':'web','paid_cents':150,'refund_cents':20,'net_cents':130}]
    facts = module.expected_facts(ledger, ['web'], {'months':['2026-02','2026-03'],'scope':'channel-comparison'})
    values = {r['column']:r['value'] for r in facts}
    assert values == {'base_net_cents':0,'current_net_cents':130,'delta_cents':130,'change_pct':None}
