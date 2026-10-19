"""Independent native-framework verification with optional bounded real calls."""
import argparse
from collections import defaultdict
from contextlib import closing
import csv
from datetime import datetime
from importlib import metadata
import json
from pathlib import Path
import sqlite3
import sys
import time
import uuid
from zoneinfo import ZoneInfo

from evaluate_holdout import capture_state,expected_facts,inspect_result
from audit_dataset_v2 import audit
from local_data_agent.artifacts import file_hash,write_json
from local_data_agent.fixtures import bootstrap
from local_data_agent.holdout_fixtures import generate_holdout
from langchain_data_agent.model import ModelGateway
from langchain_data_agent.runner import LangChainRunner
from langchain_data_agent.runtime import ModelBudgetExceeded


class SharedBudget:
    def __init__(self,max_requests=64,max_new_tokens=100000):
        self.max_requests=max_requests;self.max_new_tokens=max_new_tokens
        self.attempts=0;self.new_tokens=0;self.usage=[]

    def before_request(self):
        if self.attempts>=self.max_requests or self.new_tokens>=self.max_new_tokens:
            raise ModelBudgetExceeded('Shared evaluation budget reached')
        self.attempts+=1

    def record(self,usage):
        self.usage.append(usage)
        self.new_tokens+=usage.get('prompt_cache_miss_tokens',usage.get('prompt_tokens',0))+usage.get('completion_tokens',0)


def independent_ledger(registry_path):
    """Python event accumulation from raw aware timestamps, outside model tools."""
    registry=json.loads(registry_path.read_text(encoding='utf-8'));folder=registry_path.parent
    sources={s['source_id']:s for s in registry['sources']}
    with closing(sqlite3.connect((folder/sources['business']['path']).resolve().as_uri()+'?mode=ro',uri=True)) as db:
        db.row_factory=sqlite3.Row
        orders={r['order_id']:dict(r) for r in db.execute('SELECT * FROM orders')}
        refunds=[dict(r) for r in db.execute('SELECT * FROM refunds')]
    groups=defaultdict(lambda:{'paid_cents':0,'refund_cents':0})
    for rows,status,stamp,amount in [(list(orders.values()),'paid','paid_at','paid_cents'),(refunds,'succeeded','refunded_at','refund_cents')]:
        for row in rows:
            if row['status']!=status:continue
            when=datetime.fromisoformat(row[stamp].replace('Z','+00:00')).astimezone(ZoneInfo('Australia/Sydney')).strftime('%Y-%m')
            channel=row['channel_code'] if status=='paid' else orders[row['order_id']]['channel_code']
            groups[(when,channel)][amount]+=row[amount]
    ledger=[{'month':month,'channel_code':channel,**amounts,'net_cents':amounts['paid_cents']-amounts['refund_cents']} for (month,channel),amounts in sorted(groups.items())]
    with (folder/sources['channels']['path']).open(encoding='utf-8-sig',newline='') as f:
        channels=sorted({r['channel_code'] for r in csv.DictReader(f)})
    return ledger,channels


def verify(root,live=False,*,max_requests=64,max_new_tokens=100000):
    folder=root/'var/evaluations'/('langchain_'+uuid.uuid4().hex);runtime=folder/'runtime'
    folder.mkdir(parents=True)
    report={'runtime_root':str(runtime),'framework':'langchain.create_agent','mode':'live' if live else 'replay',
            'source_sha256':{str(p.relative_to(root/'src')):file_hash(p) for p in sorted((root/'src').rglob('*.py'))},
            'packages':{n:metadata.version(n) for n in ['langchain','langchain-deepseek','langgraph','langgraph-checkpoint-sqlite','duckdb','sqlglot']},
            'scope':'Independent framework edition; replay is declared protocol execution, live cases are limited integration validation, not general accuracy.',
            'cases':[],'oracle_hidden_from_model':True,'errors':[]}
    bootstrap(runtime)
    gateway=ModelGateway(root) if live else None
    budget=SharedBudget(max_requests,max_new_tokens)
    if gateway is not None:report['model']=gateway.metadata()
    if gateway is not None:gateway.budget=budget
    try:
        if not live:
            for case in ['cash-net-v1','repair','budget','clarification']:
                result=LangChainRunner(runtime).run('按现金净额分析' if case!='clarification' else '分析收入',mode='replay',case=case)
                expected='WAITING_INPUT' if case=='clarification' else 'FAILED' if case=='budget' else 'SUCCEEDED'
                report['cases'].append({'case_id':case,'run_id':result['run_id'],'status':result['status'],
                                        'expected_status':expected,'passed':result['status']==expected,'model_requests':result['model_requests']})
        else:
            specs=[{'case_id':'small','fixture':'cash-net-v1','months':['2026-08','2026-09'],'scope':'total-comparison',
                    'question':'按批准的现金净额口径比较2026年8月与9月，列出每月总支付、总退款、总净额，以及净额变化金额和变化率，无需推测业务原因。'},
                   {'case_id':'large','fixture':'lc-baseline','months':['2026-08','2026-09'],'scope':'monthly-channels',
                    'question':'按现金净额口径核查2026年8月与9月：列出每月总支付、总退款、总净额，以及全部渠道每月净额。退款按成功退款事件月及原订单渠道归集，避免多次退款重复累计支付。'},
                   {'case_id':'zero','fixture':'lc-empty','months':['2026-02','2026-03'],'scope':'total-comparison',
                    'question':'按现金净额口径比较2026年2月和3月，列出每月总支付、总退款、总净额、净额变化金额与变化率。已覆盖但无成功事件的月份也应展示，零基期变化率应不可定义。'}]
            generate_holdout(runtime,'lc-baseline',20261015,profile='baseline')
            generate_holdout(runtime,'lc-empty',20261016,profile='empty-base')
            for spec in specs:
                registry=runtime/'var/sources'/spec['fixture']/'registry.json'
                if spec['fixture']!='cash-net-v1':
                    audited=audit(registry);write_json(folder/'audits'/f"{spec['fixture']}.json",audited)
                    if not audited['passed']:raise ValueError('Independent source audit failed')
                ledger,channels=independent_ledger(registry)
                write_json(folder/'audits'/f"{spec['fixture']}-ledger.json",ledger)
                runner=LangChainRunner(runtime,model=gateway)
                start=time.perf_counter();result=runner.run(spec['question'],mode='live',fixture=spec['fixture'])
                observed=inspect_result(runtime,result['run_id'],expected_facts(ledger,channels,spec))
                observed.update(case_id=spec['case_id'],seconds=round(time.perf_counter()-start,3),framework_checkpoint=result.get('langchain_checkpoint'))
                observed['narrative_verification']=result.get('narrative_verification')
                report['cases'].append(observed);write_json(folder/'report.json',report)
                print(json.dumps({k:observed[k] for k in ['case_id','run_id','status','model_requests']},ensure_ascii=False),flush=True)
                if spec['case_id']=='small' and result['status']=='SUCCEEDED':
                    before=capture_state(runtime,result['run_id']);write_json(folder/'small-followup-baseline.json',before)
                    follow={'months':['2026-08','2026-09'],'scope':'channel-comparison'}
                    result=runner.resume(result['run_id'],question='沿用相同口径和快照，补充全部注册渠道在这两个月的基期净额、本期净额、变化金额和变化率，复用已有结果并保留原报告。')
                    next_result=inspect_result(runtime,result['run_id'],expected_facts(ledger,channels,follow),before)
                    next_result.update(case_id='small-followup',framework_checkpoint=result.get('langchain_checkpoint'),narrative_verification=result.get('narrative_verification'))
                    report['cases'].append(next_result);write_json(folder/'report.json',report)
                if budget.new_tokens>=budget.max_new_tokens or budget.attempts>=budget.max_requests:break
    except Exception as error:
        report['errors'].append({'type':type(error).__name__,'message':str(error)[:500]})
    finally:
        if gateway is not None:gateway.close()
        report['usage']={'attempts':budget.attempts,'new_tokens_conservative_if_cache_unknown':budget.new_tokens,
                         'reported_requests':len(budget.usage),'unreported_attempts':budget.attempts-len(budget.usage),
                         'max_requests':budget.max_requests,'max_new_tokens':budget.max_new_tokens,
                         'totals':{k:sum(u.get(k,0) for u in budget.usage) for k in ['prompt_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens','completion_tokens']}}
        expected_count=4
        report['passed']=not report['errors'] and len(report['cases'])==expected_count and all(c.get('passed',c.get('status')=='passed') for c in report['cases'])
        write_json(folder/'report.json',report)
        print(json.dumps({'report_path':str(folder/'report.json'),'passed':report['passed']}),flush=True)
    return report


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--live',action='store_true')
    parser.add_argument('--max-requests',type=int,default=64);parser.add_argument('--max-new-tokens',type=int,default=100000)
    args=parser.parse_args()
    if not 1<=args.max_requests<=128 or not 1<=args.max_new_tokens<=300000:parser.error('Evaluation budget exceeds supported bounds')
    report=verify(Path(__file__).resolve().parents[1],live=args.live,max_requests=args.max_requests,max_new_tokens=args.max_new_tokens)
    raise SystemExit(0 if report['passed'] else 1)
