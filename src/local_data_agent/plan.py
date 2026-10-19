"""Plan validation and transitive invalidation, independent of model output."""
from .models import PlanStep

def validate_plan(steps: list[dict], source_ids: list[str]) -> list[dict]:
    if not isinstance(steps, list) or not 1 <= len(steps) <= 64:
        raise ValueError('Plan must contain between 1 and 64 steps')
    normalized = [PlanStep.model_validate(step).model_dump(exclude_none=True) for step in steps]
    by_id = {step['step_id']: step for step in normalized}
    if len(by_id) != len(normalized):
        raise ValueError('Duplicate step_id')
    if len({step['output_name'] for step in normalized}) != len(normalized):
        raise ValueError('Duplicate output_name')
    for step in normalized:
        if step['type'] == 'LOAD' and step['source_id'] not in source_ids:
            raise ValueError(f"Unknown source_id: {step['source_id']}")
        if any(dep not in by_id for dep in step['depends_on']):
            raise ValueError(f"Unknown dependency for {step['step_id']}")
    visiting, visited = set(), set()
    def visit(identifier):
        if identifier in visiting:
            raise ValueError('Dependency cycle')
        if identifier in visited:
            return
        visiting.add(identifier)
        for dep in by_id[identifier]['depends_on']:
            visit(dep)
        visiting.remove(identifier)
        visited.add(identifier)
    for identifier in by_id:
        visit(identifier)
    return normalized

def apply_plan_patch(old_steps: list[dict], upsert_steps: list[dict],
                     remove_step_ids: list[str], source_ids: list[str]) -> list[dict]:
    """Apply an explicit upsert/remove patch to a validated plan.

    Order is preserved for retained and replaced ids; new ids are appended in
    the order supplied. Steps are replaced wholesale (PlanStep defaults win),
    never field-merged, and dependent steps are never removed implicitly.
    """
    old = validate_plan(old_steps, source_ids)
    if not isinstance(upsert_steps, list) or len(upsert_steps) > 64:
        raise ValueError('upsert_steps must be a list of at most 64 steps')
    if not isinstance(remove_step_ids, list) or len(remove_step_ids) > 64:
        raise ValueError('remove_step_ids must be a list of at most 64 identifiers')
    for identifier in remove_step_ids:
        if type(identifier) is not str:
            raise ValueError('remove_step_ids must contain strings only')
    if len(set(remove_step_ids)) != len(remove_step_ids):
        raise ValueError('Duplicate removal step_id')
    old_ids = {step['step_id'] for step in old}
    for identifier in remove_step_ids:
        if identifier not in old_ids:
            raise ValueError(f'Unknown removal step_id: {identifier}')
    upserted = [PlanStep.model_validate(step).model_dump(exclude_none=True) for step in upsert_steps]
    upsert_ids = [step['step_id'] for step in upserted]
    if len(set(upsert_ids)) != len(upsert_ids):
        raise ValueError('Duplicate upsert step_id')
    for identifier in upsert_ids:
        if identifier in remove_step_ids:
            raise ValueError(f'Step cannot be both upserted and removed: {identifier}')
    replacements = {step['step_id']: step for step in upserted}
    fresh_ids = [identifier for identifier in upsert_ids if identifier not in old_ids]
    raw = [replacements.get(step['step_id'], step) for step in old
           if step['step_id'] not in remove_step_ids]
    raw.extend(replacements[identifier] for identifier in fresh_ids)
    return validate_plan(raw, source_ids)

def affected_steps(old_steps: list[dict], new_steps: list[dict]) -> set[str]:
    old = {s['step_id']: PlanStep.model_validate(s).model_dump(exclude_none=True) for s in old_steps}
    new = {s['step_id']: PlanStep.model_validate(s).model_dump(exclude_none=True) for s in new_steps}
    affected = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    changed = True
    while changed:
        changed = False
        for step in [*old.values(), *new.values()]:
            if step['step_id'] not in affected and affected.intersection(step['depends_on']):
                affected.add(step['step_id'])
                changed = True
    return affected
