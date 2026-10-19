"""Coordinator acceptance: actual interfaces and independently specified cases."""
import copy
import pytest
from local_data_agent.explanation import build_explanation, render_explanation

def rows(base,current):
    channels=[dict(month=m,channel_code=c,paid_cents=p,refund_cents=r,net_cents=p-r) for m,d in [('2026-08',base),('2026-09',current)] for c,(p,r) in d.items()]
    months=[dict(month=m,**{k:sum(r[k] for r in channels if r['month']==m) for k in ('paid_cents','refund_cents','net_cents')}) for m in ('2026-08','2026-09')]
    return months,channels

def build(base=None,current=None,**kw):
    m,c=rows(base or {'web':(100,20),'app':(100,20)},current or {'web':(90,120),'app':(80,20)})
    return build_explanation(m,c,'2026-08','2026-09',artifact_refs={'month':['artifact_m'],'channel':['artifact_c']},**kw)

def statement(result,kind):
    return next(s for s in result['statements'] if s['kind']==kind)

def tickets(text='refund request'):
    e=dict(type='ticket',evidence_id=f'ticket:tickets:T1:0-{len(text)}',source_id='tickets',ticket_id='T1',source_hash='a'*64,record_hash='b'*64,source_line=9,text=text,start_line=1,end_line=1,line_count=1,start_char=0,end_char=len(text),created_at='2026-09-01T00:00:00Z',ticket_month='2026-09',ticket_channel='web',association={'status':'unlinked','order_id':None,'order_channel':None,'link_origin':None,'association_source_id':None,'association_source_hash':None,'duplicate_warning':False,'detail':'no reference'})
    return dict(source_id='tickets',query='refund',filters={'month':'2026-09','channel_code':'web'},population_ticket_count=100,matched_ticket_count=20,returned_ticket_count=1,association_counts={'verified':30,'unlinked':70,'unverified':0,'conflicting':0},evidence=[e])

def test_negative_net_zero_refund_exact_decomposition():
    r=build(base={'web':(100,0),'app':(100,0)},current={'web':(90,120),'app':(80,0)})
    d=statement(r,'month_delta')['values']
    assert (d['paid_delta_cents'],d['refund_delta_cents'],d['net_delta_cents'])==(-30,120,-150)
    assert statement(r,'channel_decline')['values']['all_channels_negative'] is True

def test_mixed_channels_ties_and_overall_gain():
    r=build(current={'web':(30,20),'app':(120,20)})
    assert statement(r,'channel_decline')['values']['all_channels_negative'] is False
    tied=build(current={'web':(80,20),'app':(80,20)})
    assert statement(tied,'channel_decline')['values']['top_negative_channels']==['app','web']
    gain=build(current={'web':(80,20),'app':(140,20)})
    assert statement(gain,'month_delta')['values']['net_delta_cents']==20
    assert statement(gain,'channel_decline')['values']['negative_contributions'][0]['channel_code']=='web'

@pytest.mark.parametrize('bad',['missing','duplicate','mismatch','bool','float'])
def test_bad_finance_rejected(bad):
    m,c=rows({'web':(100,20)},{'web':(90,20)})
    if bad=='missing':m.pop()
    elif bad=='duplicate':c.append(dict(c[0]))
    elif bad=='mismatch':m[1]['paid_cents']+=1;m[1]['net_cents']+=1
    elif bad=='bool':c[0]['paid_cents']=True
    else:c[0]['paid_cents']=100.0
    with pytest.raises(ValueError):build_explanation(m,c,'2026-08','2026-09')

@pytest.mark.parametrize('value,state',[(True,'passed'),(False,'failed'),(None,'unverified')])
def test_actual_source_check_list(value,state):
    r=build(source_validation={'checks':[{'name':'derived_event_months','passed':value}],'passed':value})
    assert r['verification_scope']['derived_event_months']==state
    assert len([s for s in r['statements'] if s['kind']=='source_validation'])==1
    assert statement(r,'source_validation')['source_refs']==['source_validation:derived_event_months']

def test_full_population_and_exact_deepcopy():
    t=tickets();orig=copy.deepcopy(t);r=build(ticket_result=t)
    assert r['ticket_evidence']==t['evidence']
    assert r['ticket_scope']['population_ticket_count']==100
    assert r['ticket_scope']['returned_ticket_count']==1
    assert r['ticket_scope']['association_counts']['verified']==30
    r['ticket_evidence'][0]['association']['detail']='edited'
    assert t==orig

def test_empty_search_and_verified_reference_noncausal():
    empty=tickets();empty.update(evidence=[],matched_ticket_count=0,returned_ticket_count=0)
    assert build(ticket_result=empty)['ticket_scope']['status']=='searched'
    linked=tickets();linked['evidence'][0]['association'].update(status='verified',order_id='O1',order_channel='web',link_origin='explicit',association_source_id='business',association_source_hash='c'*64)
    assert statement(build(ticket_result=linked),'ticket_warning')['values']['causal'] is False

@pytest.mark.parametrize('bad',['sample_as_population','bool_count','wrong_offsets','duplicate_ticket','oversize'])
def test_bad_ticket_evidence_rejected(bad):
    t=tickets()
    if bad=='sample_as_population':t['association_counts']['unlinked']=0
    elif bad=='bool_count':t['matched_ticket_count']=True
    elif bad=='wrong_offsets':t['evidence'][0]['end_char']+=1
    elif bad=='duplicate_ticket':t['evidence']*=2;t['returned_ticket_count']=2
    else:t=tickets('x'*1201)
    with pytest.raises(ValueError):build(ticket_result=t)

def test_unsafe_quote_is_literal_not_html_or_image():
    t=tickets('<script>alert(1)</script> ![image](https://evil.invalid/a) | `do this`')
    r=build(ticket_result=t);text=render_explanation(r)
    assert '<script>' not in text and '![image](' not in text
    assert '\\|' in text and '\\`' in text
    assert r['ticket_evidence'][0]['text']==t['evidence'][0]['text']
