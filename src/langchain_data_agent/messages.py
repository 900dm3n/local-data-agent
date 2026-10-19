"""Strict, non-mutating bridge between the domain ledger and native messages.

Coordinator-reviewed implementation of the delegated bridge contract. Stable
message IDs allow repeated synchronization into LangGraph without duplication.
"""
from copy import deepcopy
import hashlib
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def _string(value, field):
    if not isinstance(value,str) or not value:raise ValueError(field+' must be nonempty text')
    return value


def _arguments(value):
    if isinstance(value,str):
        try:value=json.loads(value)
        except ValueError:raise ValueError('Malformed tool argument JSON') from None
    if not isinstance(value,dict):raise ValueError('Tool arguments must be a JSON object')
    try:return json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False))
    except (ValueError,TypeError):raise ValueError('Tool arguments must be finite JSON') from None


def _content(value):
    if value is None:return ''
    if not isinstance(value,(str,list)):raise ValueError('Message content must be text or blocks')
    return deepcopy(value)


def _identifier(item,index):
    explicit=item.get('lc_message_id')
    if explicit is not None:return _string(explicit,'lc_message_id')
    encoded=json.dumps(item,sort_keys=True,ensure_ascii=False,allow_nan=False)
    return 'history_'+str(index)+'_'+hashlib.sha256(encoded.encode()).hexdigest()[:24]


def _calls(raw):
    if raw is None:return []
    if not isinstance(raw,list):raise ValueError('tool_calls must be a list')
    calls=[]
    for entry in raw:
        if not isinstance(entry,dict) or entry.get('type','function')!='function':raise ValueError('Malformed function call')
        fn=entry.get('function')
        if not isinstance(fn,dict):raise ValueError('Function object required')
        calls.append({'id':_string(entry.get('id'),'call id'),'name':_string(fn.get('name'),'function name'),
                      'args':_arguments(fn.get('arguments'))})
    return calls


def history_messages(history):
    if not isinstance(history,(list,tuple)):raise ValueError('History must be a list')
    messages=[]
    for index,item in enumerate(history):
        if not isinstance(item,dict):raise ValueError('History item must be an object')
        kind=item.get('type')
        if kind=='reasoning':continue
        if kind=='function_call_output':
            call_id=_string(item.get('call_id'),'call id')
            if not isinstance(item.get('output'),str):raise ValueError('Persisted tool output must be text')
            messages.append(ToolMessage(content=item['output'],tool_call_id=call_id,id='tool_'+call_id))
        elif kind=='function_call':
            messages.append(AIMessage(content='',id=_identifier(item,index),tool_calls=[{
                'id':_string(item.get('call_id'),'call id'),'name':_string(item.get('name'),'function name'),
                'args':_arguments(item.get('arguments'))}]))
        else:
            raw=item.get('message') if kind=='chat_message' else item
            if kind not in {None,'chat_message'} or not isinstance(raw,dict):raise ValueError('Unknown history item type')
            role=raw.get('role');content=_content(raw.get('content'));identifier=_identifier(item,index)
            if role=='system':message=SystemMessage(content=content,id=identifier)
            elif role=='user':message=HumanMessage(content=content,id=identifier)
            elif role=='assistant':message=AIMessage(content=content,id=identifier,tool_calls=_calls(raw.get('tool_calls')))
            else:raise ValueError('Unsupported persisted message role')
            messages.append(message)
    return messages


def archive_ai(message):
    if not isinstance(message,AIMessage) or message.invalid_tool_calls:raise ValueError('Valid AIMessage required')
    calls=[{'id':_string(c.get('id'),'call id'),'type':'function',
            'function':{'name':_string(c.get('name'),'function name'),
                        'arguments':json.dumps(_arguments(c.get('args')),ensure_ascii=False,allow_nan=False)}}
           for c in message.tool_calls]
    return {'type':'chat_message','lc_message_id':message.id,
            'message':{'role':'assistant','content':_content(message.content),'tool_calls':calls}}


def _count(value):
    if isinstance(value,bool) or not isinstance(value,int) or value<0:raise ValueError('Token counts must be nonnegative integers')
    return value


def _raw_usage(raw):
    result={key:_count(raw[key]) for key in ['prompt_tokens','completion_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens'] if raw.get(key) is not None}
    details=raw.get('prompt_tokens_details') or {}
    if details.get('cached_tokens') is not None:
        cached=_count(details['cached_tokens'])
        if 'prompt_cache_hit_tokens' in result and result['prompt_cache_hit_tokens']!=cached:raise ValueError('Contradictory cached counts')
        result['prompt_cache_hit_tokens']=cached
    prompt=result.get('prompt_tokens');hit=result.get('prompt_cache_hit_tokens');miss=result.get('prompt_cache_miss_tokens')
    if prompt is not None:
        if hit is not None and hit>prompt or miss is not None and miss>prompt:raise ValueError('Cache counts exceed prompt')
        if hit is not None and miss is not None and hit+miss!=prompt:raise ValueError('Inconsistent cache split')
        if hit is not None:result['prompt_cache_miss_tokens']=prompt-hit
        elif miss is not None:result['prompt_cache_hit_tokens']=prompt-miss
    return result


def normalized_usage(message):
    raw=(getattr(message,'response_metadata',None) or {}).get('token_usage') or {}
    parsed=_raw_usage(raw)
    normalized=getattr(message,'usage_metadata',None) or {}
    if not normalized:return parsed
    result={}
    for old,new in [('input_tokens','prompt_tokens'),('output_tokens','completion_tokens')]:
        if normalized.get(old) is not None:result[new]=_count(normalized[old])
    prompt=result.get('prompt_tokens')
    details=normalized.get('input_token_details') or {}
    cached=details.get('cache_read')
    if cached is not None:
        cached=_count(cached)
        if prompt is None or cached>prompt:raise ValueError('Invalid normalized cache split')
        if parsed.get('prompt_tokens')==prompt and parsed.get('prompt_cache_hit_tokens',cached)!=cached:raise ValueError('Conflicting provider usage')
    elif parsed.get('prompt_tokens')==prompt and prompt is not None:
        cached=parsed.get('prompt_cache_hit_tokens')
    if cached is not None:result.update(prompt_cache_hit_tokens=cached,prompt_cache_miss_tokens=prompt-cached)
    return result
