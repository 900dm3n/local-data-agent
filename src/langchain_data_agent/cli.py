"""The independent entry point: run/resume always use LangChain."""
from importlib import metadata
import argparse
import json
from pathlib import Path
import sys

from local_data_agent import cli as core_cli
from local_data_agent.doctor import diagnose

from .runner import LangChainRunner,independent_root,default_root


def dispatch(args):
    root=independent_root(args.project_root)
    if args.command=='serve':
        from .server import serve
        return serve(root,args.port)
    if args.command=='datasets':
        from .datasets import catalog,import_package
        if args.dataset_command=='list':return {'status':'succeeded','datasets':catalog(root)}
        if args.dataset_command=='inspect':return {'status':'succeeded',**catalog(root,args.fixture)}
        if args.dataset_command=='import':return import_package(root,args.registry,args.fixture)
        from local_data_agent.holdout_fixtures import generate_holdout
        generate_holdout(root,args.fixture,args.seed,profile=args.profile)
        return {'status':'succeeded','fixture':args.fixture}
    if args.command=='runs':
        from .workspace import Workspace
        return {'status':'succeeded','runs':Workspace(root).runs(args.limit)}
    if args.command=='export':
        from .workspace import Workspace
        destination=args.output.resolve()
        independent_root(destination.parent)
        protected=Path(__file__).resolve().parents[2].parent/'local_data_agent_demo'
        if destination.is_relative_to(protected.resolve()):raise ValueError('Original project is protected')
        # Exclusive creation also prevents overwriting a configured credential
        # or an existing report supplied accidentally as the output path.
        data=Workspace(root).export_bundle(args.run_id)
        with destination.open('xb') as stream:stream.write(data)
        return {'status':'succeeded','path':str(destination),'bytes':len(data)}
    if args.command in {'run','resume','inspect','approve-metric'}:
        runner=LangChainRunner(root)
        if args.command=='run':return runner.run(args.question,mode=args.mode,case=args.case,fixture=args.fixture)
        if args.command=='resume':return runner.resume(args.run_id,answer=args.answer,question=args.question)
        if args.command=='approve-metric':return runner.approve_metric(args.run_id,args.metric,args.approved_by)
        return runner.inspect(args.run_id)
    if args.command=='doctor':
        from .model import public_model_settings
        result=diagnose(root,probe=False)
        result['model'].update(public_model_settings(root))
        result['framework']={'runtime':'langchain.create_agent','checkpointer':'langgraph.checkpoint.sqlite.SqliteSaver',
                             'packages':{n:metadata.version(n) for n in ['langchain','langchain-core','langchain-deepseek','langchain-openai','langgraph','langgraph-checkpoint-sqlite']}}
        if args.probe:
            from .model import ModelGateway
            from langchain.agents import create_agent
            from langchain_core.tools import StructuredTool
            from langchain_core.messages import AIMessage, ToolMessage
            from langsmith import tracing_context
            from .messages import normalized_usage
            gateway=ModelGateway(root)
            try:
                def probe(value:str)->str:return value
                tool=StructuredTool.from_function(probe,name='local_probe',description='Echo the supplied value',return_direct=True)
                with tracing_context(enabled=False):
                    observed=create_agent(gateway.chat,[tool]).invoke({'messages':[{'role':'user','content':"Call local_probe with value 'ready'. Do not answer directly."}]},config={'recursion_limit':8,'max_concurrency':1})
                valid=any(isinstance(m,ToolMessage) and m.name=='local_probe' and m.content=='ready' for m in observed['messages'])
                result['model'].update(network_called=True,verification='verified' if valid else 'failed',
                    usage=[normalized_usage(m) for m in observed['messages'] if isinstance(m,AIMessage)],
                    probe_scope='LangChain native function-call loop only')
                if not valid:result['status']='failed'
            finally:gateway.close()
        return result
    if args.command=='evaluate':
        from .acceptance import evaluate
        return evaluate(root,args.suite,args.repeats)
    return core_cli.dispatch(args)


def parser():
    root=default_root()
    result=argparse.ArgumentParser(description='LangChain 本地 Data Agent：独立框架版本')
    result.add_argument('--project-root',type=Path,default=root)
    commands=result.add_subparsers(dest='command',required=True)
    def add(name,help_text):
        p=commands.add_parser(name,help=help_text);p.add_argument('--project-root',type=Path,default=argparse.SUPPRESS);return p
    p=add('serve','启动仅本机访问的工作台');p.add_argument('--port',type=int,default=8765)
    p=add('doctor','检查本地依赖与配置');p.add_argument('--probe',action='store_true')
    p=add('bootstrap','创建小型合成演示数据');p.add_argument('--fixture',default='cash-net-v1');p.add_argument('--seed',type=int,default=0);p.add_argument('--variant')
    p=add('run','开始分析');p.add_argument('--question',default='按现金净额口径比较八月和九月，并分析下降');p.add_argument('--fixture',default='cash-net-v1');p.add_argument('--mode',choices=['replay','live'],default='replay');p.add_argument('--case',default='cash-net-v1')
    p=add('resume','恢复、回答澄清或连续追问');p.add_argument('run_id');group=p.add_mutually_exclusive_group();group.add_argument('--answer');group.add_argument('--question')
    p=add('inspect','查看运行详情');p.add_argument('run_id')
    p=add('runs','列出最近运行');p.add_argument('--limit',type=int,default=50)
    p=add('export','导出结果与审计包，拒绝覆盖');p.add_argument('run_id');p.add_argument('--output',type=Path,required=True)
    p=add('approve-metric','用户明确批准现金净额口径');p.add_argument('run_id');p.add_argument('--metric',choices=['CASH_NET_v1'],required=True);p.add_argument('--approved-by',required=True)
    p=add('memory','列出或明确批准记忆草稿');subs=p.add_subparsers(dest='memory_command',required=True)
    q=subs.add_parser('list');q.add_argument('--status',choices=['DRAFT','APPROVED','STALE','REJECTED'])
    q=subs.add_parser('approve');q.add_argument('memory_id');q.add_argument('--approved-by',required=True)
    p=add('datasets','管理数据集');subs=p.add_subparsers(dest='dataset_command',required=True)
    subs.add_parser('list')
    q=subs.add_parser('inspect');q.add_argument('fixture')
    q=subs.add_parser('import');q.add_argument('registry',type=Path);q.add_argument('--fixture',required=True)
    q=subs.add_parser('generate');q.add_argument('--fixture',required=True);q.add_argument('--seed',type=int,default=73);q.add_argument('--profile',choices=['baseline','empty-base','channel-gap'],default='baseline')
    p=add('evaluate','在隔离目录验证真正的框架执行');p.add_argument('--suite',choices=['replay','live'],default='replay');p.add_argument('--repeats',type=int,default=1)
    return result


def main(argv=None):
    for stream in [sys.stdout,sys.stderr]:
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='replace')
    args=parser().parse_args(argv)
    try:result=dispatch(args)
    except (Exception,KeyboardInterrupt) as error:
        result={'status':'cancelled' if isinstance(error,KeyboardInterrupt) else 'failed',
                'error_type':type(error).__name__,'error':core_cli._safe_error(error)}
    print(json.dumps(core_cli._redact_result(result),ensure_ascii=False,indent=2,default=str))
    return 1 if str(result.get('status','')).lower() in {'failed','cancelled','unverified','blocked'} or result.get('passed') is False else 0
