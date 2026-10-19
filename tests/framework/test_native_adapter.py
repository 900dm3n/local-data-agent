import json
import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage,ToolMessage
from langchain_core.tools import StructuredTool
from langsmith import tracing_context

from langchain_data_agent.model import ModelGateway
from langchain_data_agent.messages import normalized_usage


def test_actual_chatdeepseek_streaming_wire_protocol_and_usage(monkeypatch,tmp_path):
    from langchain_data_agent import model as module
    observed=[]
    def transport(request):
        body=json.loads(request.content);observed.append(body)
        assert request.url.host=='api.deepseek.com'
        assert body['thinking']['type']=='disabled' and body['stream']
        assert body['stream_options']['include_usage']
        chunks=[{'id':'synthetic','object':'chat.completion.chunk','model':'deepseek-flash','created':0,
                 'choices':[{'index':0,'delta':{'role':'assistant','tool_calls':[{'index':0,'id':'probe','type':'function',
                     'function':{'name':'local_probe','arguments':'{"value":"ready"}'}}]},'finish_reason':None}]},
                {'id':'synthetic','object':'chat.completion.chunk','model':'deepseek-flash','created':0,
                 'choices':[{'index':0,'delta':{},'finish_reason':'tool_calls'}],
                 'usage':{'prompt_tokens':20,'completion_tokens':5,'total_tokens':25,
                          'prompt_tokens_details':{'cached_tokens':10},'prompt_cache_hit_tokens':10,'prompt_cache_miss_tokens':10}}]
        content=''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n'
        return httpx.Response(200,headers={'content-type':'text/event-stream'},content=content)
    monkeypatch.setattr(module,'load_settings',lambda root:{'provider':'deepseek','api_mode':'chat','model':'deepseek-flash','base_url':'https://api.deepseek.com','api_key':'sk-synthetic-fixture'})
    monkeypatch.setattr(module,'bounded_http_client',lambda:httpx.Client(transport=httpx.MockTransport(transport)))
    gateway=ModelGateway(tmp_path)
    def echo(value:str)->str:return value
    tool=StructuredTool.from_function(echo,name='local_probe',description='Return the given text',return_direct=True)
    try:
        with tracing_context(enabled=False):result=create_agent(gateway.chat,[tool]).invoke({'messages':[{'role':'user','content':'Probe'}]})
        assert len(observed)==1
        assert any(isinstance(m,ToolMessage) and m.content=='ready' for m in result['messages'])
        message=next(m for m in result['messages'] if isinstance(m,AIMessage))
        assert normalized_usage(message)=={'prompt_tokens':20,'completion_tokens':5,'prompt_cache_hit_tokens':10,'prompt_cache_miss_tokens':10}
    finally:gateway.close()


def test_actual_mimo_chatopenai_wire_protocol_and_usage(monkeypatch,tmp_path):
    from langchain_data_agent import model as module
    observed=[]
    def transport(request):
        body=json.loads(request.content);observed.append(body)
        assert request.url==httpx.URL('https://api.xiaomimimo.com/v1/chat/completions')
        assert request.headers['authorization']=='Bearer sk-synthetic-mimo'
        assert body['model']=='mimo-v2.5' and body['thinking']=={'type':'disabled'}
        assert body['max_completion_tokens']==4096 and body['stream_options']['include_usage']
        chunks=[{'id':'synthetic','object':'chat.completion.chunk','model':'mimo-v2.5','created':0,
            'choices':[{'index':0,'delta':{'role':'assistant','tool_calls':[{'index':0,'id':'mimo_probe','type':'function',
                'function':{'name':'local_probe','arguments':'{"value":"ready"}'}}]},'finish_reason':None}]},
            {'id':'synthetic','object':'chat.completion.chunk','model':'mimo-v2.5','created':0,
             'choices':[{'index':0,'delta':{},'finish_reason':'tool_calls'}]},
            {'id':'synthetic','object':'chat.completion.chunk','model':'mimo-v2.5','created':0,'choices':[],
             'usage':{'prompt_tokens':20,'completion_tokens':5,'total_tokens':25,'prompt_tokens_details':{'cached_tokens':12}}}]
        return httpx.Response(200,headers={'content-type':'text/event-stream'},
            content=''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n')
    monkeypatch.setattr(module,'load_settings',lambda root:{'provider':'mimo','api_mode':'chat','model':'mimo-v2.5','base_url':'https://api.xiaomimimo.com/v1','api_key':'sk-synthetic-mimo'})
    monkeypatch.setattr(module,'bounded_http_client',lambda:httpx.Client(transport=httpx.MockTransport(transport)))
    gateway=ModelGateway(tmp_path)
    def echo(value:str)->str:return value
    tool=StructuredTool.from_function(echo,name='local_probe',description='Return given text',return_direct=True)
    try:
        with tracing_context(enabled=False):result=create_agent(gateway.chat,[tool]).invoke({'messages':[{'role':'user','content':'Probe'}]})
        assert len(observed)==1 and any(isinstance(m,ToolMessage) and m.content=='ready' for m in result['messages'])
        message=next(m for m in result['messages'] if isinstance(m,AIMessage))
        assert normalized_usage(message)=={'prompt_tokens':20,'completion_tokens':5,'prompt_cache_hit_tokens':12,'prompt_cache_miss_tokens':8}
        assert gateway.metadata()['provider']=='mimo' and gateway.metadata()['chat_adapter']=='langchain_openai.ChatOpenAI'
    finally:gateway.close()


@pytest.mark.parametrize('base_url',['https://api.deepseek.com/v1','https://api.xiaomimimo.com.evil.invalid/v1',
    'http://api.xiaomimimo.com/v1','https://api.xiaomimimo.com/v1?api_key=bad','https://api.xiaomimimo.com/other'])
def test_mimo_key_cannot_be_sent_to_another_endpoint(monkeypatch,tmp_path,base_url):
    from langchain_data_agent import model as module
    monkeypatch.setattr(module,'load_settings',lambda root:{'provider':'mimo','api_mode':'chat','model':'mimo-v2.5','base_url':base_url,'api_key':'sk-synthetic-mimo'})
    with pytest.raises(ValueError,match='official HTTPS'):ModelGateway(tmp_path)
