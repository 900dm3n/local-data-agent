"""Coordinator-owned live holdout evaluation; oracle stays outside runtime."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
import time
import uuid

from audit_dataset_v2 import audit
from local_data_agent.artifacts import checked_run_dir, file_hash, verified_artifact, write_json
from local_data_agent.connectors import snapshot_sources
from local_data_agent.holdout_coverage import check_fact_coverage
from local_data_agent.holdout_fixtures import generate_holdout
from local_data_agent.model_client import APIModelClient, load_settings
from local_data_agent.runner import AgentRunner
from local_data_agent.state import StateStore
from local_data_agent.validation import validate_sources


CASES = [
    {'case_id': 'H01', 'profile': 'baseline', 'months': ['2026-05', '2026-06'], 'scope': 'total-comparison',
     'question': '按批准的现金净额口径，比较2026年5月和6月：列出两个月的总支付、总退款、总净额，以及净额变化金额和变化率。只根据冻结源计算，无需猜测业务原因。'},
    {'case_id': 'H02', 'profile': 'baseline', 'months': ['2026-08', '2026-09'], 'scope': 'monthly-channels',
     'question': '按现金净额口径核查2026年8月和9月，列出每月总支付、总退款、总净额及全部渠道每月净额。退款按成功退款月扣除并关联原订单渠道；一个订单可能有多次退款，支付不能在连接后重复计数。'},
    {'case_id': 'H03', 'profile': 'empty-base', 'months': ['2026-02', '2026-03'], 'scope': 'total-comparison',
     'question': '按现金净额口径比较2026年2月与3月，逐月列出总支付、总退款、总净额，并列出净额变化金额和变化率。没有成功事件的已覆盖月份也要展示；基期净额为零时变化率必须明确不可定义。'},
    {'case_id': 'H04', 'profile': 'channel-gap', 'months': ['2026-02', '2026-03'], 'scope': 'channel-comparison',
     'question': '按现金净额口径比较2026年2月至3月的全部注册渠道：每个渠道列出基期净额、本期净额、变化金额和变化率。某渠道某期没有成功事件也不能漏掉；基期为零时变化率不可定义。'},
]
FOLLOWUPS = [
    {'case_id': 'F01', 'months': ['2026-05', '2026-06'], 'scope': 'channel-comparison',
     'question': '沿用刚才的现金净额口径，补充全部渠道在2026年5月至6月的基期净额、本期净额、变化金额和变化率，保留可以复用的已有结果。'},
    {'case_id': 'F02', 'months': ['2026-05', '2026-06'], 'scope': 'filtered-comparison', 'channel': 'web',
     'question': '继续同一分析，这次只看官网渠道：列出2026年5月与6月各自支付、退款、现金净额，以及官网净额变化金额和变化率。明确渠道范围，沿用指标和快照。'},
    {'case_id': 'F03', 'months': ['2026-05', '2026-06'], 'scope': 'monthly-channels',
     'question': '现在恢复全部渠道范围：再次列出2026年5月与6月的总支付、总退款、总现金净额和所有渠道每月净额，不能把上轮官网范围当成全体；复用仍适用的结果，保留此前报告。'},
]
SEEDS = {'baseline': 104729, 'empty-base': 65537, 'channel-gap': 99991}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


class EvaluationBudgetExceeded(RuntimeError):
    pass


class BudgetedClient:
    def __init__(self, client, max_requests=220, max_new_tokens=300000):
        self.client, self.max_requests, self.max_new_tokens = client, max_requests, max_new_tokens
        self.request_attempts, self.new_tokens, self.usages = 0, 0, []
        self.stopped = False

    def metadata(self):
        return self.client.metadata()

    def _call(self, method, *args, **kwargs):
        if self.request_attempts >= self.max_requests or self.new_tokens >= self.max_new_tokens:
            self.stopped = True
            raise EvaluationBudgetExceeded('Global holdout evaluation budget reached')
        self.request_attempts += 1
        result = getattr(self.client, method)(*args, **kwargs)
        usage = result.get('usage', {})
        miss = usage.get('prompt_cache_miss_tokens')
        if miss is None:
            cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0
            miss = max(0, (usage.get('prompt_tokens') or 0) - cached)
        self.new_tokens += (miss or 0) + (usage.get('completion_tokens') or 0)
        self.usages.append({'kind': method, **usage})
        if self.new_tokens >= self.max_new_tokens:
            self.stopped = True  # The just-returned request is recorded; do not start another.
        return result

    def respond(self, *args, **kwargs):
        return self._call('respond', *args, **kwargs)

    def semantic(self, *args, **kwargs):
        return self._call('semantic', *args, **kwargs)

    def close(self):
        self.client.close()


def expected_facts(ledger, channels, spec):
    by_group = {(r['month'], r['channel_code']): r for r in ledger}
    first, second = spec['months']
    facts = []

    def group(month, channel=None):
        if channel is None:
            return {col: sum(by_group.get((month, ch), {}).get(col, 0) for ch in channels)
                    for col in ('paid_cents', 'refund_cents', 'net_cents')}
        return {col: by_group.get((month, channel), {}).get(col, 0)
                for col in ('paid_cents', 'refund_cents', 'net_cents')}

    def add(column, filt, value):
        facts.append({'column': column, 'filter': filt, 'value': value,
                      'tolerance': .005001 if column == 'change_pct' and value is not None else 0})

    def monthly(channel=None):
        for month in spec['months']:
            filt = {'month': month, **({'channel_code': channel} if channel else {})}
            for col, value in group(month, channel).items():
                add(col, filt, value)

    def comparison(channel=None):
        a, b = group(first, channel)['net_cents'], group(second, channel)['net_cents']
        filt = {'base_month': first, 'current_month': second, **({'channel_code': channel} if channel else {})}
        for column, value in {'base_net_cents': a, 'current_net_cents': b, 'delta_cents': b-a,
                              'change_pct': None if a == 0 else (b-a)/a*100}.items():
            add(column, filt, value)

    if spec['scope'] == 'total-comparison':
        monthly(); comparison()
    elif spec['scope'] == 'monthly-channels':
        monthly()
        for ch in channels:
            for month in spec['months']:
                add('net_cents', {'month': month, 'channel_code': ch}, group(month, ch)['net_cents'])
    elif spec['scope'] == 'channel-comparison':
        for ch in channels:
            comparison(ch)
    elif spec['scope'] == 'filtered-comparison':
        monthly(spec['channel']); comparison(spec['channel'])
    else:
        raise ValueError('Unknown evaluation scope')
    return facts


def capture_state(runtime, run_id):
    state = StateStore(runtime).load(run_id)
    run = checked_run_dir(runtime, run_id)
    def fp(path):
        path = Path(path)
        return {'path': str(path), 'sha256': file_hash(path), 'mtime_ns': path.stat().st_mtime_ns}
    return {'plan_version': state['plan_version'], 'history_length': len(state['history']),
            'history_digest': digest(state['history']), 'usage_length': len(state['usage']),
            'model_requests': state['model_requests'], 'report_path': state.get('report_path'),
            'plan': state['plan'],
            'artifacts': {name: {'artifact_id': a['artifact_id'], 'step_id': a['step_id'], **fp(a['path'])}
                          for name, a in state['artifacts'].items()},
            'publications': [fp(p) for p in (run/'publications').rglob('*') if p.is_file()],
            'sources': [fp(s['path']) for s in state['manifest']['sources']]}


def normalize_presentation(claims, required):
    """Accept equivalent monthly/period cells without changing scope or value.

    Exact displayed cells take priority, including their errors and duplicates.
    Only registered net/paid/refund period aliases are translated for evaluation;
    the original publication and artifact bindings stay untouched.
    """
    normalized, translations = list(claims), []
    for req in required:
        native = [c for c in claims if c.get('column') == req['column'] and digest(c.get('filter')) == digest(req['filter'])]
        if native:
            continue
        for c in claims:
            filt = c.get('filter')
            if not isinstance(filt, dict):
                continue
            column, target, translated = c.get('column'), req['column'], False
            extra = {'channel_code': req['filter']['channel_code']} if 'channel_code' in req['filter'] else {}
            if 'month' in req['filter'] and target in {'net_cents', 'paid_cents', 'refund_cents'}:
                if set(filt) != {'base_month', 'current_month', *extra}:
                    continue
                if digest({k: filt[k] for k in extra}) != digest(extra):
                    continue
                for prefix, period_key in (('base_', 'base_month'), ('current_', 'current_month')):
                    if column == prefix + target and filt[period_key] == req['filter']['month']:
                        translated = True
            elif target in {'base_net_cents', 'current_net_cents'}:
                period = 'base_month' if target.startswith('base_') else 'current_month'
                wanted = {'month': req['filter'][period], **extra}
                translated = column == 'net_cents' and digest(filt) == digest(wanted)
            if translated:
                normalized.append({**c, 'column': target, 'filter': req['filter']})
                translations.append({'claim_id': c.get('claim_id'), 'original_column': column,
                                     'original_filter': filt, 'required_column': target, 'required_filter': req['filter']})
    return normalized, translations


def inspect_result(runtime, run_id, required, baseline=None):
    state = StateStore(runtime).load(run_id)
    run = checked_run_dir(runtime, run_id)
    report = state.get('report_path')
    claims = json.loads((Path(report).parent/'claims.json').read_text(encoding='utf-8')) if report else []
    comparable, translations = normalize_presentation(claims, required)
    coverage = check_fact_coverage(comparable, required)
    baseline = baseline or {'history_length': 0, 'usage_length': 0, 'model_requests': 0, 'plan_version': 0}
    calls, outputs = [], {}
    for item in state['history'][baseline['history_length']:]:
        if item.get('type') == 'chat_message':
            for call in item['message'].get('tool_calls', []):
                calls.append({'call_id': call['id'], **call['function']})
        elif item.get('type') == 'function_call':
            calls.append(item)
        elif item.get('type') == 'function_call_output':
            outputs[item['call_id']] = json.loads(item['output'])
    errors = [{'name': c['name'], 'call_id': c['call_id'], 'output': outputs[c['call_id']]}
              for c in calls if c['call_id'] in outputs and
              ('error' in outputs[c['call_id']] or outputs[c['call_id']].get('passed') is False)]
    hashes_ok = all(file_hash(s['path']) == s['sha256'] for s in state['manifest']['sources'])
    binding_checks = []
    for claim in claims:
        bound = next((a for a in state['artifacts'].values() if a['artifact_id'] == claim.get('artifact_id')), None)
        if bound is None:
            binding_checks.append(False)
        else:
            try:
                verified_artifact(run, bound)
                binding_checks.append(True)
            except (ValueError, OSError):
                binding_checks.append(False)
    old_files_ok, history_ok, reusable_checks = True, True, []
    if 'publications' in baseline:
        old_files_ok = all(Path(p['path']).is_file() and file_hash(p['path']) == p['sha256']
                           and Path(p['path']).stat().st_mtime_ns == p['mtime_ns'] for p in baseline['publications'])
        history_ok = digest(state['history'][:baseline['history_length']]) == baseline['history_digest']
        from local_data_agent.plan import affected_steps
        affected = affected_steps(baseline['plan'], state['plan'])
        # A newly added downstream node does not affect its unchanged ancestors.
        for name, old in baseline['artifacts'].items():
            if old['step_id'] in affected:
                continue
            current = state['artifacts'].get(name)
            ok = bool(current and current['artifact_id'] == old['artifact_id'] and
                      file_hash(current['path']) == old['sha256'] and
                      Path(current['path']).stat().st_mtime_ns == old['mtime_ns'])
            reusable_checks.append({'output_name': name, 'passed': ok})
    checks = {'run_succeeded': state['status'] == 'SUCCEEDED', 'required_displayed_facts': coverage['passed'],
              'frozen_sources_unchanged': hashes_ok, 'bound_artifact_hashes': bool(binding_checks) and all(binding_checks),
              'old_publications_unchanged': old_files_ok, 'prior_protocol_history_unchanged': history_ok,
              'unaffected_artifacts_reused': all(x['passed'] for x in reusable_checks)}
    if 'publications' in baseline:
        checks['new_publication'] = bool(report and report != baseline['report_path'])
    usage = state['usage'][baseline['usage_length']:]
    return {'run_id': run_id, 'status': 'passed' if all(checks.values()) else 'failed',
            'run_status': state['status'], 'stop_reason': state.get('stop_reason'), 'checks': checks,
            'coverage': coverage, 'presentation_translations': translations, 'report_path': report, 'tool_names': [c['name'] for c in calls],
            'tool_errors': errors, 'reusable_artifact_checks': reusable_checks,
            'plan_version_before': baseline['plan_version'], 'plan_version_after': state['plan_version'],
            'model_requests': state['model_requests'] - baseline['model_requests'],
            'usage': {k: sum(u.get(k, 0) or 0 for u in usage) for k in
                      ('prompt_tokens', 'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens', 'completion_tokens')},
            'narrative_review': 'pending_manual_review', 'error': state.get('error')}


def evaluate(root, repeats=3, max_requests=220, max_new_tokens=300000, selection=None):
    root = Path(root).resolve()
    if repeats < 3:
        raise ValueError('At least three repetitions required')
    identifier = 'holdout_' + uuid.uuid4().hex
    folder = root/'var/evaluations'/identifier
    runtime = folder/'runtime'
    folder.mkdir(parents=True)
    code = root/'src/local_data_agent'
    code_hashes = {p.name: file_hash(p) for p in sorted(code.glob('*.py'))}
    report = {'evaluation_id': identifier, 'started_at': datetime.now(timezone.utc).isoformat(),
              'status': 'preparing', 'runtime_root': str(runtime), 'code_sha256': code_hashes,
              'prompt_sha256': hashlib.sha256(__import__('local_data_agent.agent_loop', fromlist=['SYSTEM_PROMPT']).SYSTEM_PROMPT.encode()).hexdigest(),
              'packages': {k: metadata.version(k) for k in ('pyarrow', 'duckdb', 'pydantic', 'sqlglot', 'openai')},
              'gold_hidden_from_agent': True, 'memory_isolated': True, 'cases': [], 'source_audits': {},
              'catalog': CASES, 'followup_catalog': FOLLOWUPS,
              'scope': 'Fixed new question families on held-out synthetic seeds and financial-calendar stress distributions. Not general-purpose accuracy.',
              'budget': {'max_requests': max_requests, 'max_uncached_input_plus_output': max_new_tokens}}
    report_path = folder/'report.json'
    def checkpoint():
        write_json(report_path, report)
    checkpoint()
    registries, ledgers = {}, {}
    for profile, seed in SEEDS.items():
        fixture = 'holdout-' + profile
        registry = generate_holdout(runtime, fixture, seed, profile=profile)
        registry_path = runtime/'var/sources'/fixture/'registry.json'
        source_audit = audit(registry_path)
        write_json(folder/'audits'/f'{profile}.json', source_audit)
        manifest = snapshot_sources(runtime, runtime/'var/workspaces'/('audit_'+profile), fixture)
        actual_contract = validate_sources(manifest)
        write_json(folder/'audits'/f'{profile}-source-contract.json', actual_contract)
        passed = source_audit['passed'] and actual_contract['passed']
        report['source_audits'][profile] = {'passed': passed, 'checks': len(source_audit['checks']), 'path': str(folder/'audits'/f'{profile}.json')}
        checkpoint()
        if not passed:
            report.update(status='source_audit_failed', finished_at=datetime.now(timezone.utc).isoformat())
            checkpoint(); return report
        registries[profile], ledgers[profile] = registry, source_audit['ledger']
    selected = [s for s in CASES if selection is None or s['case_id'] in selection]
    if selection is not None and set(selection)-{s['case_id'] for s in CASES}:
        raise ValueError('Unknown selected case')
    settings = load_settings(root)
    client = BudgetedClient(APIModelClient(settings), max_requests, max_new_tokens)
    report['model'] = client.metadata()
    report['status'] = 'running'
    channels = ['app', 'marketplace', 'partner', 'web']
    unexecuted = []
    try:
        for spec in selected:
            for repeat in range(1, repeats+1):
                if client.stopped:
                    unexecuted.append({'case_id': spec['case_id'], 'repeat': repeat}); continue
                runner = AgentRunner(runtime, model=client)
                started = time.perf_counter()
                result = runner.run(spec['question'], mode='live', fixture='holdout-'+spec['profile'])
                observed = inspect_result(runtime, result['run_id'], expected_facts(ledgers[spec['profile']], channels, spec))
                observed.update(case_id=spec['case_id'], profile=spec['profile'], repeat=repeat, question=spec['question'], seconds=round(time.perf_counter()-started, 3))
                report['cases'].append(observed)
                checkpoint()
                print(json.dumps({k: observed[k] for k in ('case_id', 'repeat', 'status', 'run_id', 'model_requests')}, ensure_ascii=False), flush=True)
                if spec['case_id'] == 'H01':
                    chain_ok = observed['run_status'] == 'SUCCEEDED'
                    for followup in FOLLOWUPS:
                        if not chain_ok or client.stopped:
                            unexecuted.append({'case_id': followup['case_id'], 'repeat': repeat, 'reason': 'prior_run_not_completed_or_budget'})
                            continue
                        before = capture_state(runtime, result['run_id'])
                        write_json(folder/'baselines'/f'{repeat}-{followup["case_id"]}.json', before)
                        started = time.perf_counter()
                        result = runner.resume(result['run_id'], question=followup['question'])
                        next_result = inspect_result(runtime, result['run_id'], expected_facts(ledgers['baseline'], channels, followup), before)
                        next_result.update(case_id=followup['case_id'], profile='baseline', repeat=repeat, question=followup['question'], seconds=round(time.perf_counter()-started, 3))
                        report['cases'].append(next_result)
                        chain_ok = next_result['run_status'] == 'SUCCEEDED'
                        checkpoint()
                        print(json.dumps({k: next_result[k] for k in ('case_id', 'repeat', 'status', 'run_id', 'model_requests')}, ensure_ascii=False), flush=True)
    finally:
        client.close()
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report['unexecuted'] = unexecuted
        report['budget']['requests_attempted'] = client.request_attempts
        report['budget']['uncached_input_plus_output'] = client.new_tokens
        report['budget']['stopped'] = client.stopped
        report['totals'] = {'executed_turns': len(report['cases']),
                            'passed': sum(c['status'] == 'passed' for c in report['cases']),
                            'failed': sum(c['status'] == 'failed' for c in report['cases'])}
        report['usage'] = {k: sum(u.get(k, 0) or 0 for u in client.usages) for k in
                           ('prompt_tokens', 'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens', 'completion_tokens')}
        report['status'] = 'budget_stopped' if client.stopped else 'completed_with_failures' if report['totals']['failed'] or unexecuted else 'passed_pending_narrative_review'
        checkpoint()
    return report


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--max-requests', type=int, default=220)
    parser.add_argument('--max-new-tokens', type=int, default=300000)
    parser.add_argument('--cases', help='Optional fixed case IDs comma separated; all four by default')
    args = parser.parse_args()
    result = evaluate(args.project_root, args.repeats, args.max_requests, args.max_new_tokens, args.cases.split(',') if args.cases else None)
    print(json.dumps({k: result[k] for k in ('evaluation_id', 'status', 'runtime_root')}, ensure_ascii=False))
    raise SystemExit(0 if result['status'] == 'passed_pending_narrative_review' else 1)
