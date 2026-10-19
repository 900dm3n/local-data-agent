"""Independent acceptance of deterministic wording and cross-table consistency."""
from copy import deepcopy

import pytest

from local_data_agent.table_narrative import compile_table_narrative


def table(rows, name='a'):
    return {'artifact_id': name, 'rows': rows}


def monthly(net, month='2025-01', **extra):
    return {'month': month, 'net_cents': net, **extra}


def pair(base=100, current=120, **extra):
    return {'base_month': '2025-01', 'current_month': '2025-02',
            'base_net_cents': base, 'current_net_cents': current,
            'delta_cents': current-base, **extra}


def test_overall_growth_and_mixed_channels_have_one_collective_statement():
    rows=[monthly(100),monthly(110,'2025-02'),monthly(60,channel_code='app'),
          monthly(40,'2025-02',channel_code='app'),monthly(40,channel_code='web'),
          monthly(70,'2025-02',channel_code='web')]
    result=compile_table_narrative([table(rows)])
    assert len(result['groups'])==3
    assert '全体汇总的净额上升' in result['summary']
    assert '所选渠道净额方向不一致' in result['summary']
    assert '全部上升' not in result['summary']
    assert not any(c.isdigit() for c in result['summary'])


def test_zero_base_and_negative_base_direction():
    zero=compile_table_narrative([table([pair(0,30,change_pct=None)])])
    assert zero['groups'][0]['change_pct'] is None
    assert '不可定义' in zero['summary']
    negative=compile_table_narrative([table([pair(-100,-50,change_pct=-50)])])
    assert negative['groups'][0]['net_direction']=='up'
    assert '净额上升' in negative['summary']


@pytest.mark.parametrize('row',[pair(0,30,change_pct=0),pair(change_pct=None),
                              pair(change_pct=float('nan')),pair(change_pct=True),
                              pair(change_pct=99),pair(current_month='2025-01')])
def test_invalid_comparisons_are_rejected(row):
    with pytest.raises(ValueError):compile_table_narrative([table([row])])


def test_component_directions_and_compatible_duplicates_do_not_mutate_input():
    tables=[table([monthly(100),monthly(120,'2025-02')]),
            table([monthly(100,paid_cents=150,refund_cents=50),
                   monthly(120,'2025-02',paid_cents=160,refund_cents=40)],'b'),
            table([pair(change_pct=20)],'c')]
    before=deepcopy(tables)
    result=compile_table_narrative(tables)
    assert tables==before and len(result['groups'])==1
    group=result['groups'][0]
    assert group['source_artifact_ids']==['a','b','c']
    assert group['base_paid_cents']==150 and group['current_refund_cents']==40
    assert '支付金额上升' in result['summary'] and '退款金额下降' in result['summary']


def test_cross_artifact_conflicts_reject():
    with pytest.raises(ValueError):
        compile_table_narrative([table([monthly(100)]),table([monthly(101)],'b')])
    with pytest.raises(ValueError):
        compile_table_narrative([table([monthly(100),monthly(120,'2025-02')]),table([pair(100,130)],'b')])


def test_more_than_two_months_infers_no_pair_but_checks_explicit_overlap():
    rows=[monthly(100),monthly(120,'2025-02'),monthly(80,'2025-03')]
    assert compile_table_narrative([table(rows)])['groups']==[]
    with pytest.raises(ValueError):
        compile_table_narrative([table(rows),table([pair(100,130)],'b')])


def test_partial_channel_scope_never_implies_whole_company():
    result=compile_table_narrative([table([pair(channel_code='web')])])
    assert '全体' not in result['summary'] and '所选渠道的净额上升' in result['summary']


@pytest.mark.parametrize('row',[monthly(True),monthly(1.0),monthly(None),
                              monthly(1,'2025-13'),monthly(1,'0000-01'),
                              monthly(100,paid_cents=None,refund_cents=None),
                              monthly(100,paid_cents=100),
                              monthly(100,paid_cents=90,refund_cents=-10),
                              monthly(100,paid_cents=120,refund_cents=10),
                              monthly(100,business_cause='promotion')])
def test_invalid_monthly_money_scope_or_shape_rejected(row):
    with pytest.raises(ValueError):compile_table_narrative([table([row])])


def test_benign_columns_ignored_and_multiple_overall_pairs_not_misstated():
    result=compile_table_narrative([table([monthly(100,channel_name='label',payment_count=2),
                                         monthly(120,'2025-02',refund_count=1)])])
    assert len(result['groups'])==1
    result=compile_table_narrative([table([pair(),pair(120,100,base_month='2025-02',current_month='2025-03')])])
    assert '各比较期间的净额方向不一致' in result['summary']
