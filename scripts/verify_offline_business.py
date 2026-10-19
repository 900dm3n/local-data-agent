"""Large, declared business contracts through the real native agent graph.

No autonomous model is evaluated here. SQL/actions are supplied by this harness;
source audits and timestamp-derived oracle values stay outside agent tools.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from audit_dataset_v2 import audit
from evaluate_holdout import capture_state, expected_facts, inspect_result
from verify_langchain import independent_ledger
from local_data_agent.artifacts import file_hash, write_json
from local_data_agent.fixtures import bootstrap
from local_data_agent.holdout_fixtures import generate_holdout
from local_data_agent.replay import load, select
from langchain_data_agent.model import ModelGateway
from langchain_data_agent.runner import LangChainRunner


def action(name, **arguments):
    return name, arguments


class DeclaredBusinessChat(BaseChatModel):
    """Finite native tool-call script, explicitly independent of oracle values."""
    actions: list = Field(default_factory=list)
    calls: int = 0
    prefix: str = Field(default_factory=lambda: uuid.uuid4().hex)
    observer: object = Field(default=None, exclude=True)

    @property
    def _llm_type(self):return 'declared-offline-business-contract'

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.observer is not None:self.observer(self.calls, messages)
        if self.calls >= len(self.actions):
            raise AssertionError('Declared protocol exhausted before verified publication')
        batch=self.actions[self.calls]
        if isinstance(batch,tuple):batch=[batch]
        calls=[{'name':name,'args':deepcopy(args),'id':f'{self.prefix}_{self.calls}_{i}'}
               for i,(name,args) in enumerate(batch)]
        self.calls+=1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content='',tool_calls=calls))])


class OfflineGateway(ModelGateway):
    def metadata(self):
        return {**super().metadata(),'mode':'offline-contract','autonomous':False,'provider_requests':0}


def financial_plan(months):
    """Aggregate at the source before bounded extraction, then combine in DuckDB."""
    base,current=months
    selected=f"('{base}','{current}')"
    steps=[load('payments_agg','business','orders',
                'SELECT local_month AS month,channel_code,SUM(paid_cents) AS paid_cents '
                f"FROM orders WHERE status='paid' AND local_month IN {selected} GROUP BY local_month,channel_code"),
           load('refunds_agg','business','refunds',
                'SELECT r.local_month AS month,o.channel_code,SUM(r.refund_cents) AS refund_cents '
                'FROM refunds r JOIN orders o ON o.order_id=r.order_id '
                f"WHERE r.status='succeeded' AND r.local_month IN {selected} GROUP BY r.local_month,o.channel_code"),
           load('channels','channels','channels')]
    steps.append(select('channel_totals',['payments_agg','refunds_agg','channels'],
        f"WITH periods AS (SELECT '{base}' AS month UNION ALL SELECT '{current}' AS month), "
        'grid AS (SELECT p.month,c.channel_code FROM periods p CROSS JOIN channels c) '
        'SELECT g.month,g.channel_code,COALESCE(CAST(p.paid_cents AS BIGINT),0) AS paid_cents,'
        'COALESCE(CAST(r.refund_cents AS BIGINT),0) AS refund_cents,'
        'COALESCE(CAST(p.paid_cents AS BIGINT),0)-COALESCE(CAST(r.refund_cents AS BIGINT),0) AS net_cents '
        'FROM grid g LEFT JOIN payments_agg p ON p.month=g.month AND p.channel_code=g.channel_code '
        'LEFT JOIN refunds_agg r ON r.month=g.month AND r.channel_code=g.channel_code ORDER BY g.month,g.channel_code'))
    steps.append(select('month_totals',['channel_totals'],
        'SELECT month,CAST(SUM(paid_cents) AS BIGINT) AS paid_cents,'
        'CAST(SUM(refund_cents) AS BIGINT) AS refund_cents,CAST(SUM(net_cents) AS BIGINT) AS net_cents '
        'FROM channel_totals GROUP BY month ORDER BY month'))
    for name,table,channel in [('comparison','month_totals',False),('channel_compare','channel_totals',True)]:
        steps.append(select(name,[table],
            f'SELECT a.month AS base_month,b.month AS current_month,{"a.channel_code," if channel else ""}'
            'a.net_cents AS base_net_cents,b.net_cents AS current_net_cents,b.net_cents-a.net_cents AS delta_cents,'
            'CASE WHEN a.net_cents=0 THEN NULL ELSE CAST(b.net_cents-a.net_cents AS DOUBLE)/a.net_cents*100 END AS change_pct '
            f'FROM {table} a JOIN {table} b ON {"a.channel_code=b.channel_code" if channel else "1=1"} '
            f"WHERE a.month='{base}' AND b.month='{current}'"))
    return steps


def execute_and_publish(version, artifacts=None):
    return [[action('execute_step',plan_version=version,step_id='comparison'),
             action('execute_step',plan_version=version,step_id='channel_compare')],
            action('check_result',artifact_ids=['month_totals','channel_totals'],
                   check_ids=['integrity','finite','monthly_reconciliation']),
            action('publish_verified_tables',plan_version=version,
                   artifact_ids=artifacts or ['month_totals','channel_totals','comparison'],open_questions=[])]


def initial_actions(months, broken=False):
    plan=financial_plan(months)
    if broken:
        # Explicitly provoke the real 10,000-row extraction boundary.
        plan[0]=load('payments_agg','business','orders','SELECT * FROM orders')
    actions=[action('inspect_source_checks'),action('bind_metric',metric_id='CASH_NET_v1'),
             action('propose_plan',steps_json=json.dumps(plan),reason='Declared offline contract; aggregate independent events before joining.')]
    if broken:
        actions.extend([action('execute_step',plan_version=1,step_id='refunds_agg'),
                        action('execute_step',plan_version=1,step_id='comparison'),
                        action('patch_plan',base_plan_version=1,
                               upsert_steps_json=json.dumps([financial_plan(months)[0]]),remove_step_ids=[],
                               reason='Repair the actual upstream extraction overflow with source aggregation.')])
    actions.extend(execute_and_publish(2 if broken else 1))
    return actions


def required_all(ledger,channels,months):
    facts=expected_facts(ledger,channels,{'months':months,'scope':'total-comparison'})
    for ch in channels:
        facts.extend(expected_facts(ledger,channels,{'months':months,'scope':'filtered-comparison','channel':ch})[:6])
    return facts


def verify(root):
    folder=root/'var/evaluations'/('offline_business_'+uuid.uuid4().hex)
    runtime=folder/'runtime';folder.mkdir(parents=True)
    report={'started_at':datetime.now(timezone.utc).isoformat(),'runtime_root':str(runtime),
            'scope':'Declared offline business contracts, real LangChain/tools/SQLite/DuckDB/Parquet. No autonomous planning or semantic model claims.',
            'mode':'offline-contract','autonomous':False,'provider_requests':0,'oracle_hidden_from_tools':True,
            'source_sha256':{str(p.relative_to(root)):file_hash(p) for p in sorted((root/'src').rglob('*.py'))},
            'harness_sha256':file_hash(Path(__file__)),'cases':[],'source_audits':{},'errors':[]}
    bootstrap(runtime)

    def save():write_json(folder/'report.json',report)

    def run_case(case_id,actions,fixture,question,required,previous=None,observer=None,allowed_errors=0):
        model=DeclaredBusinessChat(actions=actions,observer=observer)
        gateway=OfflineGateway(runtime,chat_model=model)
        runner=LangChainRunner(runtime,model=gateway)
        before=capture_state(runtime,previous) if previous else None
        if before:write_json(folder/'baselines'/f'{case_id}.json',before)
        start=time.perf_counter()
        result=runner.resume(previous,question=question) if previous else runner.run(question,mode='live',fixture=fixture)
        # The existing resume contract uses mode=live for follow-ups; actual
        # provider metadata above and this report explicitly identify injection.
        observed=inspect_result(runtime,result['run_id'],required,before)
        observed.update(case_id=case_id,seconds=round(time.perf_counter()-start,3),
                        script_calls=model.calls,provider_requests=0,autonomous=False,
                        framework_checkpoint=result.get('langchain_checkpoint'),
                        narrative_verification=result.get('narrative_verification'))
        observed['checks']['native_checkpoint_present']=bool(result.get('langchain_checkpoint',{}).get('checkpoint_id'))
        observed['checks']['host_deterministic_narrative']=result.get('narrative_verification')=='host_deterministic_tables'
        observed['checks']['protocol_consumed']=model.calls==len(actions)
        observed['checks']['only_expected_tool_errors']=len(observed['tool_errors'])==allowed_errors
        observed['status']='passed' if all(observed['checks'].values()) else 'failed'
        report['cases'].append(observed);save()
        print(json.dumps({k:observed[k] for k in ['case_id','status','script_calls','seconds']},ensure_ascii=False),flush=True)
        return result,observed,runner

    try:
        # Any accidental provider configuration/HTTP request is an immediate
        # failure. No credentials are loaded by this evaluation.
        with patch('langchain_data_agent.model.load_settings',side_effect=AssertionError('Provider access forbidden in offline verification')), \
             patch('httpx.Client.send',side_effect=AssertionError('HTTP forbidden in offline verification')):
            for index,profile in enumerate(['baseline','empty-base','channel-gap']):
                fixture='offline-'+profile
                generate_holdout(runtime,fixture,20261020+index,profile=profile)
                registry=runtime/'var/sources'/fixture/'registry.json'
                audited=audit(registry);write_json(folder/'audits'/f'{profile}.json',audited)
                report['source_audits'][profile]={'passed':audited['passed'],'checks':len(audited['checks']),
                                                 'orders':12000,'tickets':600,'documents':60,'seed':20261020+index}
                if not audited['passed']:raise AssertionError(f'{profile} independent source audit failed')
                ledger,channels=independent_ledger(registry)
                write_json(folder/'audits'/f'{profile}-ledger.json',ledger)
                months=['2026-08','2026-09'] if profile=='baseline' else ['2026-02','2026-03']
                result,_,_=run_case(profile,initial_actions(months),fixture,
                    f'按现金净额核查{months}每月及全部渠道支付、退款、净额与总变化；无成功事件的已覆盖范围也展示。',
                    required_all(ledger,channels,months))
                if result['status']!='SUCCEEDED':continue
                if profile=='channel-gap':
                    run_case('channel-gap-rates',[action('publish_verified_tables',plan_version=1,
                        artifact_ids=['channel_compare'],open_questions=[])],fixture,
                        '在同一冻结快照上补充全部渠道变化率；零基期明确不可定义。',
                        expected_facts(ledger,channels,{'months':months,'scope':'channel-comparison'}),result['run_id'])
                if profile!='baseline':continue
                run_id=result['run_id']
                web=[select('web_monthly',['channel_totals'],"SELECT * FROM channel_totals WHERE channel_code='web'"),
                     select('web_compare',['channel_compare'],"SELECT * FROM channel_compare WHERE channel_code='web'")]
                run_case('followup-web',[
                    action('patch_plan',base_plan_version=1,upsert_steps_json=json.dumps(web),remove_step_ids=[],reason='Only web scope.'),
                    [action('execute_step',plan_version=2,step_id='web_monthly'),action('execute_step',plan_version=2,step_id='web_compare')],
                    action('publish_verified_tables',plan_version=2,artifact_ids=['web_monthly','web_compare'],open_questions=[])],
                    fixture,'只看官网，沿用指标、快照和既有产物。',
                    expected_facts(ledger,channels,{'months':months,'scope':'filtered-comparison','channel':'web'}),run_id)
                run_case('followup-restore',[
                    action('patch_plan',base_plan_version=2,upsert_steps_json='[]',remove_step_ids=['web_monthly','web_compare'],reason='Restore all registered channels.'),
                    action('publish_verified_tables',plan_version=3,artifact_ids=['month_totals','channel_totals','comparison'],open_questions=[])],
                    fixture,'恢复全部渠道，保留此前报告。',required_all(ledger,channels,months),run_id)
                new_months=['2026-05','2026-06']
                run_case('followup-period',[
                    action('patch_plan',base_plan_version=3,upsert_steps_json=json.dumps(financial_plan(new_months)),remove_step_ids=[],reason='Change comparison periods; retain unchanged channel dimension.')
                    ]+execute_and_publish(4),fixture,'改为五月与六月，复用仍适用的产物。',
                    required_all(ledger,channels,new_months),run_id)

                # Capture a real committed ancestor just before and after patch;
                # the observer never supplies decisions or expected money values.
                retained={}
                def observe(cursor,messages):
                    if cursor not in {5,6}:return
                    tools=[m for m in messages if m.type=='tool']
                    if cursor==5:
                        outputs=[json.loads(m.content) for m in tools]
                        retained['failure']=outputs[-1]
                        a=outputs[-2]['artifact']
                        paths=list((runtime/'var/workspaces').glob(f"run_*/artifacts/{a['artifact_id']}.parquet"))
                        if len(paths)!=1:raise AssertionError('Expected one committed refund artifact')
                        path=paths[0]
                        retained['before']={'artifact_id':a['artifact_id'],'sha256':file_hash(path),
                                            'mtime_ns':path.stat().st_mtime_ns,'path':str(path)}
                    else:retained['patch']=json.loads(tools[-1].content)
                repaired,observed,runner=run_case('upstream-overflow-repair',initial_actions(months,broken=True),fixture,
                    '按现金净额分析：先触发已声明的上游行数故障，再局部修复并复用已完成的退款聚合。',
                    required_all(ledger,channels,months),observer=observe,allowed_errors=1)
                state=runner.store.load(repaired['run_id']);a=state['artifacts'].get('refunds_agg',{})
                old=retained.get('before',{})
                checks={'overflow_was_observed':bool(retained.get('failure',{}).get('error')),
                        'upstream_correctly_attributed':retained.get('failure',{}).get('failed_step',{}).get('step_id')=='payments_agg',
                        'completed_refunds_reused':bool(old and a.get('artifact_id')==old['artifact_id']
                            and file_hash(a['path'])==old['sha256'] and Path(a['path']).stat().st_mtime_ns==old['mtime_ns']),
                        'repair_increments_plan':state['plan_version']==2}
                observed['checks'].update(checks);observed['repair_evidence']=retained
                observed['status']='passed' if all(observed['checks'].values()) else 'failed';save()
    except Exception as error:
        report['errors'].append({'type':type(error).__name__,'message':str(error)[:1000]})
    finally:
        report['passed']=not report['errors'] and len(report['cases'])==8 and all(c['status']=='passed' for c in report['cases'])
        report['finished_at']=datetime.now(timezone.utc).isoformat();save()
        print(json.dumps({'report_path':str(folder/'report.json'),'passed':report['passed'],'cases':len(report['cases']),
                          'errors':report['errors']},ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(0 if verify(Path(__file__).resolve().parents[1])['passed'] else 1)
