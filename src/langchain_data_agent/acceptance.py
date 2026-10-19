"""Installed CLI acceptance entry; fixed protocols never imply model accuracy."""
from pathlib import Path
import uuid

from local_data_agent.artifacts import file_hash,write_json
from local_data_agent.fixtures import bootstrap,compute_gold
from .model import ModelGateway
from .runner import LangChainRunner


def evaluate(root,suite='replay',repeats=1):
    if suite not in {'replay','live'} or type(repeats) is not int or not 1<=repeats<=3:
        raise ValueError('Suite replay/live and repeats 1..3 required')
    root=Path(root);folder=root/'var/evaluations'/('cli_'+uuid.uuid4().hex);runtime=folder/'runtime'
    registry=bootstrap(runtime);gold=compute_gold(registry)
    report={'suite':suite,'autonomous_model':suite=='live','cases':[],
            'scope':'Small native framework acceptance only; larger source/turn/repair matrix lives in scripts/verify_offline_business.py.',
            'provider_requests':0,'source_sha256':{str(p.relative_to(root/'src')):file_hash(p) for p in sorted((root/'src').rglob('*.py'))}}
    gateway=ModelGateway(root) if suite=='live' else None
    try:
        cases={'cash-net-v1':'SUCCEEDED'} if suite=='live' else {'cash-net-v1':'SUCCEEDED','repair':'SUCCEEDED','budget':'FAILED','clarification':'WAITING_INPUT'}
        for repeat in range(repeats):
            for case,expected in cases.items():
                runner=LangChainRunner(runtime,model=gateway)
                question='分析收入' if case=='clarification' else '按现金净额口径比较2026年8月和9月净额，并列出九月全部渠道净额；不得推测业务原因。'
                result=runner.run(question,mode=suite,case=case)
                state=runner.store.load(result['run_id']);coverage=True
                if expected=='SUCCEEDED':
                    if result.get('report_path'):
                        import json
                        claims=json.loads((Path(result['report_path']).parent/'claims.json').read_text(encoding='utf-8'))
                        required=[({'month':month},gold['monthly_cents'][month]) for month in ['2026-08','2026-09']]
                        required.extend(({'month':r['month'],'channel_code':r['channel_code']},r['net_cents']) for r in gold['monthly_channel_cents'] if r['month']=='2026-09')
                        for filters,value in required:
                            matches=[c for c in claims if c.get('column')=='net_cents' and c.get('filter')==filters]
                            coverage=coverage and len(matches)==1 and matches[0].get('value')==value
                    else:coverage=False
                report['cases'].append({'repeat':repeat+1,'case':case,'run_id':result['run_id'],'status':result['status'],
                    'expected_status':expected,'displayed_required_facts_verified':coverage if expected=='SUCCEEDED' else None,
                    'passed':result['status']==expected and coverage,'model_requests':result['model_requests']})
                if suite=='live':report['provider_requests']+=result['model_requests']
                write_json(folder/'report.json',report)
    finally:
        if gateway is not None:gateway.close()
    report['passed']=all(c['passed'] for c in report['cases']) and len(report['cases'])==len(cases)*repeats
    report['status']='SUCCEEDED' if report['passed'] else 'FAILED';report['report_path']=str(folder/'report.json')
    write_json(folder/'report.json',report);return report
