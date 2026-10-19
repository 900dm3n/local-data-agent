from copy import deepcopy
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_data_agent.messages import archive_ai,history_messages,normalized_usage


def test_multiple_call_roundtrip_unicode_and_stable_ids():
    message=AIMessage(content='',id='ai-1',tool_calls=[{'name':'search','args':{'q':'退款'},'id':'c1'},
                                                    {'name':'inspect','args':{},'id':'c2'}])
    archived=archive_ai(message)
    history=[{'role':'system','content':'instructions'},{'role':'user','content':'问题'},archived,
             {'type':'function_call_output','call_id':'c1','output':'{"结果":"原文"}'},
             {'type':'function_call_output','call_id':'c2','output':'{}'}]
    before=deepcopy(history);first=history_messages(history);second=history_messages(history)
    assert first[2].tool_calls==message.tool_calls and first[2].id=='ai-1'
    assert first[-1].id=='tool_c2' and first[-1].tool_call_id=='c2'
    assert [m.id for m in first]==[m.id for m in second] and history==before


def test_nested_blocks_are_isolated_in_both_directions():
    message=AIMessage(content=[{'type':'text','text':'真实内容'}])
    archived=archive_ai(message);archived['message']['content'][0]['text']='修改'
    assert message.content[0]['text']=='真实内容'
    before=deepcopy(archived);native=history_messages([archived])[0]
    native.content[0]['text']='再修改'
    assert archived==before


@pytest.mark.parametrize('arguments',[None,'','[]','{"x":NaN}',{'x':float('inf')}])
def test_invalid_tool_argument_payloads_rejected(arguments):
    with pytest.raises(ValueError):history_messages([{'type':'function_call','call_id':'c','name':'t','arguments':arguments}])


def test_unknown_items_fail_but_reasoning_metadata_is_explicitly_ignored():
    assert history_messages([{'type':'reasoning','summary':[]}])==[]
    with pytest.raises(ValueError):history_messages([{'type':'unknown'}])
    with pytest.raises(ValueError):history_messages({'role':'user','content':'x'})


def usage(normal=None,raw=None):return normalized_usage(SimpleNamespace(usage_metadata=normal,response_metadata={'token_usage':raw}))


def test_normalized_cache_and_provider_fallback_do_not_doublecount():
    expected={'prompt_tokens':100,'completion_tokens':20,'prompt_cache_hit_tokens':80,'prompt_cache_miss_tokens':20}
    assert usage({'input_tokens':100,'output_tokens':20,'input_token_details':{'cache_read':80}})==expected
    assert usage({'input_tokens':100,'output_tokens':20},{'prompt_tokens':100,'prompt_cache_hit_tokens':80})==expected
    assert usage(None,{'prompt_tokens':100,'completion_tokens':20,'prompt_tokens_details':{'cached_tokens':80}})==expected
    assert usage({'input_tokens':100,'output_tokens':20},{'prompt_tokens':99,'prompt_cache_hit_tokens':80})=={'prompt_tokens':100,'completion_tokens':20}
    assert usage()=={}


@pytest.mark.parametrize('raw',[{'prompt_tokens':True},{'prompt_tokens':-1},
                              {'prompt_tokens':10,'prompt_cache_hit_tokens':11},
                              {'prompt_tokens':10,'prompt_cache_hit_tokens':5,'prompt_cache_miss_tokens':6}])
def test_invalid_usage_counters_are_rejected(raw):
    with pytest.raises(ValueError):usage(None,raw)
