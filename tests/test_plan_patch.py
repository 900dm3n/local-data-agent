"""Tests for apply_plan_patch (pure function, no model or IO dependency)."""
import copy

import pytest

from local_data_agent.plan import affected_steps, apply_plan_patch, validate_plan

SOURCE_IDS = ['business']


def load(step_id='load_orders', output_name=None, table='orders', source_id='business'):
    return {
        'step_id': step_id,
        'type': 'LOAD',
        'output_name': output_name or step_id,
        'source_id': source_id,
        'table': table,
    }


def select(step_id, depends_on, sql='SELECT * FROM x', output_name=None):
    return {
        'step_id': step_id,
        'type': 'SELECT',
        'output_name': output_name or step_id,
        'depends_on': list(depends_on),
        'sql': sql,
    }


def run(step_id, depends_on, operator='classify_tickets', parameters=None, output_name=None):
    step = {
        'step_id': step_id,
        'type': 'RUN',
        'output_name': output_name or step_id,
        'depends_on': list(depends_on),
        'operator': operator,
    }
    if parameters is not None:
        step['parameters'] = parameters
    return step


def save(step_id, depends_on, output_name=None):
    return {
        'step_id': step_id,
        'type': 'SAVE',
        'output_name': output_name or step_id,
        'depends_on': list(depends_on),
    }


def base_plan():
    return [
        load('load_orders'),
        select('select_orders', ['load_orders']),
        save('save_orders', ['select_orders']),
    ]


def ids(steps):
    return [step['step_id'] for step in steps]


def test_append_retains_order_and_appends_new_ids():
    plan = base_plan()
    new_step = run('classify', ['select_orders'], parameters={'labels': ['a', 'b']})
    result = apply_plan_patch(plan, [new_step], [], SOURCE_IDS)
    assert ids(result) == ['load_orders', 'select_orders', 'save_orders', 'classify']
    assert result[3]['parameters'] == {'labels': ['a', 'b']}


def test_append_multiple_new_ids_follow_supplied_order():
    plan = base_plan()
    patch = [run('second', ['select_orders']), run('first', ['select_orders'])]
    result = apply_plan_patch(plan, patch, [], SOURCE_IDS)
    assert ids(result) == ['load_orders', 'select_orders', 'save_orders', 'second', 'first']


def test_replace_keeps_position_and_full_step():
    plan = base_plan()
    replacement = select('select_orders', ['load_orders'], sql='SELECT id FROM orders')
    result = apply_plan_patch(plan, [replacement], [], SOURCE_IDS)
    assert ids(result) == ['load_orders', 'select_orders', 'save_orders']
    assert result[1]['sql'] == 'SELECT id FROM orders'


def test_remove_step():
    plan = base_plan()
    result = apply_plan_patch(plan, [], ['save_orders'], SOURCE_IDS)
    assert ids(result) == ['load_orders', 'select_orders']


def test_replace_and_remove_together_preserve_order():
    plan = base_plan()
    patch = [run('classify', ['select_orders']), load('load_orders', table='orders_v2')]
    result = apply_plan_patch(plan, patch, ['save_orders'], SOURCE_IDS)
    assert ids(result) == ['load_orders', 'select_orders', 'classify']
    assert result[0]['table'] == 'orders_v2'


def test_replacement_drops_old_extra_fields_not_merged():
    plan = base_plan()
    replacement = select('select_orders', ['load_orders'], sql='SELECT 1')
    result = apply_plan_patch(plan, [replacement], [], SOURCE_IDS)
    assert 'source_id' not in result[1]
    assert 'operator' not in result[1]


def test_replacement_resets_parameters_default_when_omitted():
    plan = base_plan() + [run('classify', ['select_orders'], parameters={'k': 1})]
    replacement = run('classify', ['select_orders'])
    result = apply_plan_patch(plan, [replacement], [], SOURCE_IDS)
    assert result[3]['parameters'] == {}


def test_empty_patch_returns_normalized_equivalent_plan():
    plan = base_plan()
    result = apply_plan_patch(plan, [], [], SOURCE_IDS)
    assert ids(result) == ids(plan)
    assert result == apply_plan_patch(plan, [], [], SOURCE_IDS)
    assert result == validate_plan(plan, SOURCE_IDS)


def test_equivalent_upsert_returns_unchanged_semantics():
    plan = base_plan()
    same = select('select_orders', ['load_orders'])
    result = apply_plan_patch(plan, [same], [], SOURCE_IDS)
    assert result == validate_plan(plan, SOURCE_IDS)
    assert affected_steps(plan, result) == set()


def test_input_dicts_are_not_mutated():
    plan = base_plan()
    patch = [run('classify', ['select_orders'], parameters={'labels': ['a']})]
    plan_snapshot = copy.deepcopy(plan)
    patch_snapshot = copy.deepcopy(patch)
    apply_plan_patch(plan, patch, ['save_orders'], SOURCE_IDS)
    assert plan == plan_snapshot
    assert patch == patch_snapshot


def test_output_is_independent_for_nested_parameters():
    parameters = {'labels': ['a'], 'nested': {'weights': [1, 2]}}
    plan = base_plan() + [run('classify', ['select_orders'], parameters=parameters)]
    result = apply_plan_patch(plan, [], [], SOURCE_IDS)
    result[3]['parameters']['labels'].append('mutated')
    result[3]['parameters']['nested']['weights'].append(99)
    assert parameters == {'labels': ['a'], 'nested': {'weights': [1, 2]}}
    assert plan[3]['parameters']['labels'] == ['a']


def test_duplicate_upsert_ids_rejected():
    plan = base_plan()
    patch = [run('classify', ['select_orders']), run('classify', ['load_orders'])]
    with pytest.raises(ValueError):
        apply_plan_patch(plan, patch, [], SOURCE_IDS)


def test_upsert_and_removal_conflict_rejected():
    plan = base_plan()
    with pytest.raises(ValueError):
        apply_plan_patch(plan, [select('select_orders', ['load_orders'])],
                         ['select_orders'], SOURCE_IDS)


def test_unknown_removal_rejected():
    plan = base_plan()
    with pytest.raises(ValueError):
        apply_plan_patch(plan, [], ['missing'], SOURCE_IDS)


def test_duplicate_removal_rejected():
    plan = base_plan()
    with pytest.raises(ValueError):
        apply_plan_patch(plan, [], ['save_orders', 'save_orders'], SOURCE_IDS)


def test_removal_rejects_bool_int_and_objects():
    plan = base_plan()
    for bad in [True, 1, {'step_id': 'save_orders'}, None]:
        with pytest.raises(ValueError):
            apply_plan_patch(plan, [], [bad], SOURCE_IDS)


def test_bad_shapes_rejected():
    plan = base_plan()
    with pytest.raises(ValueError):
        apply_plan_patch(plan, 'not-a-list', [], SOURCE_IDS)
    with pytest.raises(ValueError):
        apply_plan_patch(plan, [], 'not-a-list', SOURCE_IDS)


def test_empty_old_plan_rejected():
    with pytest.raises(ValueError):
        apply_plan_patch([], [load('load_orders')], [], SOURCE_IDS)


def test_invalid_source_rejected():
    plan = base_plan()
    bad_load = load('load_other', source_id='unknown')
    with pytest.raises(ValueError):
        apply_plan_patch(plan, [bad_load], [], SOURCE_IDS)


def test_missing_dependency_rejected():
    plan = base_plan()
    with pytest.raises(ValueError):
        apply_plan_patch(plan, [select('select_x', ['ghost'])], [], SOURCE_IDS)


def test_cycle_rejected():
    plan = base_plan()
    patch = [run('classify', ['select_orders'])]
    result = apply_plan_patch(plan, patch, [], SOURCE_IDS)
    assert 'classify' in ids(result)
    cyclic = [select('select_a', ['select_b']), select('select_b', ['select_a'])]
    with pytest.raises(ValueError):
        apply_plan_patch([load('load_orders')], cyclic, [], SOURCE_IDS)


def test_output_name_collision_rejected():
    plan = base_plan()
    clash = run('classify', ['select_orders'], output_name='save_orders')
    with pytest.raises(ValueError):
        apply_plan_patch(plan, [clash], [], SOURCE_IDS)


def test_removing_dependency_without_repair_rejected():
    plan = base_plan()
    with pytest.raises(ValueError):
        apply_plan_patch(plan, [], ['load_orders'], SOURCE_IDS)


def test_removing_dependency_with_repair_allowed():
    plan = base_plan()
    repair = [load('load_other'), select('select_orders', ['load_other'])]
    result = apply_plan_patch(plan, repair, ['load_orders'], SOURCE_IDS)
    assert ids(result) == ['select_orders', 'save_orders', 'load_other']


def test_empty_final_plan_rejected():
    plan = [load('load_orders')]
    with pytest.raises(ValueError):
        apply_plan_patch(plan, [], ['load_orders'], SOURCE_IDS)


def test_more_than_64_upserts_rejected():
    plan = [load('load_orders')]
    too_many = [select('select_%d' % i, ['load_orders']) for i in range(65)]
    with pytest.raises(ValueError):
        apply_plan_patch(plan, too_many, [], SOURCE_IDS)


def test_more_than_64_removals_rejected():
    plan = [load('load_orders')]
    too_many = ['step_%d' % i for i in range(65)]
    with pytest.raises(ValueError):
        apply_plan_patch(plan, [], too_many, SOURCE_IDS)


def test_invalid_step_shapes_rejected():
    plan = base_plan()
    with pytest.raises(ValueError):
        apply_plan_patch(plan, [{'step_id': 'bad', 'type': 'LOAD', 'output_name': 'bad'}], [], SOURCE_IDS)
    with pytest.raises(ValueError):
        apply_plan_patch(plan, [{'step_id': 'bad', 'type': 'NOPE', 'output_name': 'bad'}], [], SOURCE_IDS)


def test_operation_specific_fields_enforced():
    plan = base_plan()
    bad_run = {'step_id': 'r', 'type': 'RUN', 'output_name': 'r', 'depends_on': ['select_orders']}
    with pytest.raises(ValueError):
        apply_plan_patch(plan, [bad_run], [], SOURCE_IDS)
    bad_save = save('s', ['select_orders', 'load_orders'])
    with pytest.raises(ValueError):
        apply_plan_patch(plan, [bad_save], [], SOURCE_IDS)


def test_affected_steps_propagates_across_old_and_new_graph():
    plan = base_plan()
    replacement = load('load_orders', table='orders_v2')
    result = apply_plan_patch(plan, [replacement], [], SOURCE_IDS)
    assert affected_steps(plan, result) == {'load_orders', 'select_orders', 'save_orders'}


def test_affected_steps_reports_appended_and_removed_ids():
    plan = base_plan() + [run('classify', ['select_orders'])]
    result = apply_plan_patch(plan, [], ['classify'], SOURCE_IDS)
    assert affected_steps(plan, result) == {'classify'}


def test_affected_steps_empty_for_semantically_unchanged_patch():
    plan = base_plan() + [run('classify', ['select_orders'], parameters={'labels': ['a']})]
    result = apply_plan_patch(plan, [run('classify', ['select_orders'], parameters={'labels': ['a']})],
                              [], SOURCE_IDS)
    assert affected_steps(plan, result) == set()
    assert result == validate_plan(plan, SOURCE_IDS)


def test_combined_plan_exceeds_64_even_with_one_upsert():
    plan = [load('load_%d' % i) for i in range(64)]
    with pytest.raises(ValueError):
        apply_plan_patch(plan, [load('one_more')], [], SOURCE_IDS)


def test_output_upsert_nested_parameters_are_independent():
    patch = [run('classify', ['select_orders'], parameters={'nested': {'values': [1]}})]
    result = apply_plan_patch(base_plan(), patch, [], SOURCE_IDS)
    result[-1]['parameters']['nested']['values'].append(2)
    assert patch[-1]['parameters'] == {'nested': {'values': [1]}}


def test_rewired_dependencies_and_unaffected_branch():
    plan = base_plan() + [load('other'), select('other_total', ['other'])]
    result = apply_plan_patch(plan, [load('load_orders', table='other_table'),
                                   select('select_orders', ['other'])], [], SOURCE_IDS)
    assert affected_steps(plan, result) == {'load_orders', 'select_orders', 'save_orders'}
