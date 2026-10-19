"""LangChain-native orchestration with durable local domain transactions."""
from contextvars import ContextVar
from contextlib import closing
from pathlib import Path
import sqlite3
import threading
import uuid

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import tracing_context

from local_data_agent.agent_loop import SYSTEM_PROMPT, current_policy_history, execute_calls, pending_calls
from local_data_agent.artifacts import dumps, write_json
from local_data_agent.context_budget import check_history, prepare_history
from local_data_agent.tools import TOOL_DEFINITIONS
from local_data_agent.model_client import ProviderError

from .messages import archive_ai, history_messages, normalized_usage


TERMINAL={'WAITING_INPUT','SUCCEEDED','FAILED','CANCELLED'}
_tool_call: ContextVar[dict|None]=ContextVar('domain_tool_call',default=None)


class ModelBudgetExceeded(RuntimeError):pass


class RunCancelled(RuntimeError):pass


def check_cancel(context):
    check=getattr(context,'cancel_check',None)
    if context.state['status']=='SUCCEEDED' or check is None or not check():return False
    context.state.update(status='CANCELLED',stop_reason='user_cancelled')
    execute_calls(context,pending_calls(context.state['history']))
    context.store.save(context.state,'cancellation_observed')
    return True


def registered_tools(context):
    """Real StructuredTools; middleware supplies the framework's actual call ID."""
    def make(definition):
        def invoke(**arguments):
            call=_tool_call.get()
            if call is None or call['name']!=definition['name']:
                raise RuntimeError('Domain tools must execute inside the LangChain tool middleware')
            execute_calls(context,[{'call_id':call['id'],'name':call['name'],'arguments':arguments}])
            return context.state['history'][-1]['output']
        return StructuredTool(name=definition['name'],description=definition['description'],
                              args_schema=definition['parameters'],func=invoke,
                              handle_tool_error=False,handle_validation_error=False)
    return [make(d) for d in TOOL_DEFINITIONS]


class VerifiedDataMiddleware(AgentMiddleware):
    """Keep domain correctness and budgets around LangChain's native agent loop."""
    def __init__(self, context, max_requests=16):
        self.context=context
        self.baseline=context.state['model_requests']
        self.max_requests=max_requests
        self.tool_lock=threading.RLock()

    @hook_config(can_jump_to=['end'])
    def before_model(self, state, runtime):
        if check_cancel(self.context) or self.context.state['status'] in TERMINAL:
            return {'jump_to':'end'}
        return None

    def wrap_model_call(self, request, handler):
        state,store=self.context.state,self.context.store
        if check_cancel(self.context):raise RunCancelled()
        if state['model_requests']-self.baseline>=self.max_requests:
            raise ModelBudgetExceeded('Per-interaction model request budget exceeded')
        budget=getattr(self.context.model,'budget',None)
        prepared=prepare_history(current_policy_history(state['history']),state=state)
        visible=history_messages(prepared)
        state['context_budget']=check_history(prepared)
        state['context_budget']['provider_view']='langchain_native_messages_from_bounded_domain_view'
        if budget is not None:budget.before_request()
        state['model_requests']+=1  # Actual attempts, including requests that error.
        store.save(state,'model_request',{'request_number':state['model_requests'],'framework':'langchain'})
        try:
            response=handler(request.override(messages=[m for m in visible if not isinstance(m,SystemMessage)],
                                              system_message=SystemMessage(content=SYSTEM_PROMPT)))
        except Exception as error:
            if check_cancel(self.context):
                store.save(state,'cancelled_model_request_error',{'type':type(error).__name__,'http_status':getattr(error,'status_code',None)})
                raise RunCancelled() from None
            raise ProviderError(f'LangChain model request failed: {type(error).__name__}; HTTP={getattr(error,"status_code",None)}') from None
        for message in response.result:
            if isinstance(message,AIMessage):
                if not message.id:message.id='ai_'+uuid.uuid4().hex
                usage=normalized_usage(message)
                state['usage'].append({'kind':'agent',**usage})
                if budget is not None:budget.record(usage)
                # Invalid provider tool arguments are retained as an error event,
                # never reclassified as a successful publication.
                state['history'].append(archive_ai(message))
                finish=message.response_metadata.get('finish_reason')
                if finish is not None and finish not in {'stop','tool_calls'}:
                    state['status']='FAILED';state['stop_reason']='model_response_incomplete'
                    # Pair but permanently skip any calls from this incomplete
                    # response. Resume must not execute a truncated decision.
                    execute_calls(self.context,[{'call_id':c['id'],'name':c['name'],'arguments':c['args']}
                                                for c in message.tool_calls])
        store.save(state,'model_response',{'framework':'langchain','messages':len(response.result)})
        return response

    @hook_config(can_jump_to=['model','end'])
    def after_model(self, state, runtime):
        domain=self.context.state
        if check_cancel(self.context):return {'jump_to':'end'}
        if domain['status'] in TERMINAL:return {'jump_to':'end'}
        last=state['messages'][-1]
        if isinstance(last,AIMessage) and not last.tool_calls and domain['status'] not in TERMINAL:
            notice='请通过工具取得证据并正式发布；自由文字答复不能跳过验证，缺少定义时使用 ask_user。'
            identifier='notice_'+uuid.uuid4().hex
            domain['history'].append({'role':'user','content':notice,'lc_message_id':identifier})
            self.context.store.save(domain,'unpublished_model_text')
            return {'messages':[HumanMessage(content=notice,id=identifier)],'jump_to':'model'}
        return None

    def wrap_tool_call(self, request, handler):
        with self.tool_lock:
            return self._execute_tool(request,handler)

    def _execute_tool(self, request, handler):
        if check_cancel(self.context):
            output=next(i['output'] for i in self.context.state['history']
                        if i.get('type')=='function_call_output' and i.get('call_id')==request.tool_call['id'])
            return ToolMessage(content=output,tool_call_id=request.tool_call['id'],
                               name=request.tool_call['name'],id='tool_'+request.tool_call['id'])
        token=_tool_call.set(request.tool_call)
        try:
            result=handler(request)
            # Framework validation/unknown-tool errors can short-circuit the
            # registered function. Still preserve a paired domain reply so the
            # next provider request never has a dangling tool call.
            if not any(item.get('type')=='function_call_output' and item.get('call_id')==request.tool_call['id']
                       for item in self.context.state['history']):
                call=request.tool_call
                execute_calls(self.context,[{'call_id':call['id'],'name':call['name'],'arguments':call['args']}])
                result=ToolMessage(content=self.context.state['history'][-1]['output'],
                                   tool_call_id=call['id'],name=call['name'])
            if isinstance(result,ToolMessage):result.id='tool_'+request.tool_call['id']
            return result
        finally:_tool_call.reset(token)


def run_agent(context, *, max_requests=16):
    if not 1<=max_requests<=16:raise ValueError('Model request budget must be 1..16')
    state,store=context.state,context.store
    if not check_cancel(context):execute_calls(context,pending_calls(state['history']))
    if state['status'] in TERMINAL:return
    model=context.model.chat
    if hasattr(model,'protocol_history'):model.protocol_history=state['history']
    baseline=state['model_requests']
    checkpoint_path=context.root/'var/langchain_checkpoints.sqlite'
    checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
    config={'configurable':{'thread_id':state['run_id']},'max_concurrency':1,'recursion_limit':256}
    try:
        with closing(sqlite3.connect(checkpoint_path,check_same_thread=False,timeout=15)) as db:
            db.execute('PRAGMA journal_mode=WAL')
            saver=SqliteSaver(db)
            graph=create_agent(model=model,tools=registered_tools(context),system_prompt=SYSTEM_PROMPT,
                               middleware=[VerifiedDataMiddleware(context,max_requests)],checkpointer=saver)
            (context.run_dir/'langchain_graph.mmd').write_text(graph.get_graph().draw_mermaid(),encoding='utf-8')
            # Local transcripts/checkpoints only. No LangSmith outbound tracing,
            # even if a parent shell happens to enable that environment variable.
            try:
                with tracing_context(enabled=False):
                    graph.invoke({'messages':history_messages(state['history'])},config=config)
            finally:
                # A failed provider request can still leave a durable graph
                # checkpoint. Keep that evidence without masking its exception.
                try:
                    checkpoint=graph.get_state(config)
                    state.pop('checkpoint_inspection_error',None)
                    state['langchain_checkpoint']={**checkpoint.config.get('configurable',{}),
                                                  'path':str(checkpoint_path),
                                                  'messages':len(checkpoint.values.get('messages',[])),
                                                  'next':list(checkpoint.next)}
                except Exception as error:
                    state['checkpoint_inspection_error']=type(error).__name__
            if state['status'] not in TERMINAL:
                state['status']='FAILED';state['stop_reason']='framework_ended_without_verified_publication'
    except ModelBudgetExceeded:
        state['status']='FAILED';state['stop_reason']='model_request_budget_exceeded'
    except RunCancelled:
        state['status']='CANCELLED';state['stop_reason']='user_cancelled'
    finally:
        state['session_requests']=state['model_requests']-baseline
        store.save(state,'langchain_loop_stopped',{'status':state['status'],'framework':'langchain.create_agent'})
