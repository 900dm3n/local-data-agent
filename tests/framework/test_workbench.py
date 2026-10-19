"""Product lifecycle, imported source integrity, HTTP isolation and exports."""
from contextlib import contextmanager
import csv
import io
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import zipfile

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from local_data_agent.artifacts import file_hash,write_json
from local_data_agent.fixtures import bootstrap
from langchain_data_agent.cli import dispatch,parser
from langchain_data_agent.datasets import catalog,import_package
from langchain_data_agent.jobs import JobManager,JobStore,validated_job
from langchain_data_agent.runner import LangChainRunner
from langchain_data_agent.server import WorkbenchServer
from langchain_data_agent.workspace import Workspace


def wait_for(store,identifier,statuses,timeout=40):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        job=store.get(identifier)
        if job['status'] in statuses:return job
        time.sleep(.03)
    raise AssertionError(store.get(identifier))


def test_import_is_atomic_independent_and_keeps_sqlite_wal_commits(tmp_path):
    source=tmp_path/'package';registry=bootstrap(source)
    original=Path(registry['sources'][0]['path'])
    with sqlite3.connect(original) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute("INSERT INTO orders VALUES('EXTRA','2026-09-01T01:00:00Z','web',123,'paid')");db.commit()
        original_hash=file_hash(original)
        root=tmp_path/'runtime'
        result=import_package(root,source/'var/sources/cash-net-v1/registry.json','imported')
        assert result['validation']['passed']
        assert file_hash(original)==original_hash
    detail=catalog(root,'imported');assert detail['sources'][0]['schema']['orders']
    copied=json.loads((root/'var/sources/imported/registry.json').read_text(encoding='utf-8'))
    assert all(not Path(s['path']).is_absolute() for s in copied['sources'])
    imported=root/'var/sources/imported'/copied['sources'][0]['path']
    with sqlite3.connect(imported) as db:assert db.execute("SELECT paid_cents FROM orders WHERE order_id='EXTRA'").fetchone()[0]==123
    assert imported!=original
    assert not list((root/'var/sources').glob('.import-*'))
    with pytest.raises(ValueError,match='already exists'):import_package(root,source/'var/sources/cash-net-v1/registry.json','imported')


@pytest.mark.parametrize('variant',['duplicate-channels','orphan-refund','null-amount'])
def test_invalid_financial_package_never_becomes_visible(tmp_path,variant):
    bootstrap(tmp_path/'package',variant=variant)
    root=tmp_path/'runtime'
    with pytest.raises(ValueError,match='validation failed'):
        import_package(root,tmp_path/'package/var/sources/cash-net-v1/registry.json','bad')
    assert catalog(root)==[] and not (root/'var/sources/bad').exists()
    assert not list((root/'var/sources').glob('.import-*'))


def test_package_cannot_read_files_outside_its_directory(tmp_path):
    bootstrap(tmp_path/'package');path=tmp_path/'package/var/sources/cash-net-v1/registry.json'
    registry=json.loads(path.read_text(encoding='utf-8'));registry['documents'][0]['path']=str(tmp_path/'outside.md')
    (tmp_path/'outside.md').write_text('private',encoding='utf-8');write_json(path,registry)
    with pytest.raises(ValueError,match='outside'):import_package(tmp_path/'runtime',path,'escape')


def test_import_rejects_an_approved_contract_with_different_event_semantics(tmp_path):
    bootstrap(tmp_path/'package');path=tmp_path/'package/var/sources/cash-net-v1/registry.json'
    registry=json.loads(path.read_text(encoding='utf-8'))
    registry['metric_contracts'][0]['refund_channel']='refund_request.channel_code';write_json(path,registry)
    with pytest.raises(ValueError,match='implemented CASH_NET_v1'):import_package(tmp_path/'runtime',path,'wrong-metric')
    assert catalog(tmp_path/'runtime')==[]


def test_queue_cancels_pending_and_active_tasks_without_running_pending(tmp_path):
    started=threading.Event();executed=[]
    def executor(root,job,cancel,attach):
        executed.append(job['job_id']);started.set()
        deadline=time.monotonic()+5
        while not cancel() and time.monotonic()<deadline:time.sleep(.01)
        return {'status':'CANCELLED'}
    manager=JobManager(tmp_path,executor=executor)
    try:
        first=manager.submit('bootstrap',{'fixture':'first'});assert started.wait(2)
        second=manager.submit('bootstrap',{'fixture':'second'});manager.store.cancel(second['job_id'])
        assert manager.store.get(second['job_id'])['status']=='CANCELLED'
        manager.store.cancel(first['job_id']);wait_for(manager.store,first['job_id'],{'CANCELLED'})
        assert executed==[first['job_id']]
        with pytest.raises(RuntimeError,match='already running'):JobManager(tmp_path)
    finally:manager.close()


def test_restart_marks_inflight_interrupted_and_never_retries_it(tmp_path):
    store=JobStore(tmp_path);old=store.submit('bootstrap',{'fixture':'old'});store.claim()
    queued=store.submit('bootstrap',{'fixture':'queued'});seen=[]
    def executor(root,job,cancel,attach):seen.append(job['job_id']);return {'status':'SUCCEEDED'}
    manager=JobManager(tmp_path,executor=executor)
    try:
        assert store.get(old['job_id'])['status']=='INTERRUPTED'
        wait_for(store,queued['job_id'],{'SUCCEEDED'})
        assert seen==[queued['job_id']]
    finally:manager.close()


def test_duplicate_run_operations_and_unknown_payloads_are_rejected(tmp_path):
    store=JobStore(tmp_path);identifier='run_'+'a'*32
    store.submit('resume',{'run_id':identifier,'question':'只看官网'})
    with pytest.raises(ValueError,match='active job'):store.submit('resume',{'run_id':identifier})
    with pytest.raises(ValueError):validated_job('run',{'fixture':'cash-net-v1','mode':'live','question':'x','api_key':'secret'})
    with pytest.raises(ValueError):validated_job('run',{'fixture':'large','mode':'replay','question':'x'})
    with pytest.raises(ValueError):validated_job('generate',{'fixture':'x','document_count':4})


@contextmanager
def running_server(root):
    server=WorkbenchServer(root,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{server.server_port}',timeout=10) as client:yield server,client
    finally:server.shutdown();server.server_close();thread.join(timeout=2)


def test_http_host_origin_and_write_token_are_enforced(tmp_path):
    with running_server(tmp_path) as (server,client):
        assert client.get('/api/session',headers={'Host':'attacker.invalid'}).status_code==403
        assert client.get('/api/session',headers={'Origin':'https://attacker.invalid'}).status_code==403
        assert client.get('/api/session',headers={'Sec-Fetch-Site':'cross-site'}).status_code==403
        assert client.post('/api/jobs',json={'kind':'bootstrap','payload':{'fixture':'bad'}}).status_code==403
        token=client.get('/api/session').json()['token']
        headers={'X-Workspace-Token':token}
        assert client.post('/api/jobs',json={'kind':'shell','payload':{}},headers=headers).status_code==400
        response=client.get('/');assert response.status_code==200 and 'frame-ancestors' in response.headers['content-security-policy']
        assert client.get('/app.js').status_code==200 and client.get('/style.css').status_code==200
        assert client.get('/config.local.json').status_code==404
        assert client.get('/api/runs/../../config.local.json').status_code in {400,404}
        assert client.post('/api/jobs',json={'kind':'bootstrap','payload':{'fixture':'created'}},headers=headers).status_code==202


def test_http_replay_to_publication_preview_csv_and_audit_bundle(tmp_path):
    bootstrap(tmp_path)
    with running_server(tmp_path) as (server,client):
        token=client.get('/api/session').json()['token'];headers={'X-Workspace-Token':token}
        response=client.post('/api/jobs',json={'kind':'run','payload':{'fixture':'cash-net-v1','mode':'replay','question':'现金净额比较'}},headers=headers)
        assert response.status_code==202,response.text
        job=wait_for(server.manager.store,response.json()['job_id'],{'SUCCEEDED','FAILED'},timeout=60)
        assert job['status']=='SUCCEEDED',job
        run=job['run_id'];detail=client.get(f'/api/runs/{run}').json()
        assert detail['status']=='SUCCEEDED' and detail['model']['mode']=='replay'
        assert client.get(f'/api/runs/{run}/report').json()['claims']
        preview=client.get(f'/api/runs/{run}/artifacts/monthly_total').json()
        assert preview['rows']==[{'month':'2026-08','net_cents':38000},{'month':'2026-09','net_cents':31000}]
        csv_response=client.get(f'/api/runs/{run}/artifacts/monthly_total/download')
        assert csv_response.status_code==200 and '2026-08,38000' in csv_response.text
        events=client.get(f'/api/runs/{run}/events?after=0').json();assert events
        assert all(e['event_id']>events[0]['event_id'] for e in client.get(f'/api/runs/{run}/events?after={events[0]["event_id"]}').json())
        content=client.get(f'/api/runs/{run}/export').content
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            assert 'export_manifest.json' in archive.namelist()
            assert any(n.endswith('/report.md') for n in archive.namelist())
            assert not any(n.endswith('config.local.json') or '/inputs/' in n for n in archive.namelist())
            manifest=json.loads(archive.read('export_manifest.json'))
            import hashlib
            assert all(hashlib.sha256(archive.read(f['name'])).hexdigest()==f['sha256'] for f in manifest['files'])
            assert {f['name'] for f in manifest['files']}==set(archive.namelist())-{'export_manifest.json'}
        saved=server.workspace.store.load(run);old=saved['artifacts'].pop('monthly_total');server.workspace.store.save(saved)
        historical=client.get(f'/api/runs/{run}/export')
        assert historical.status_code==200
        with zipfile.ZipFile(io.BytesIO(historical.content)) as archive:
            assert f'artifacts/{old["artifact_id"]}.parquet' in archive.namelist()
        saved['artifacts']['monthly_total']=old;server.workspace.store.save(saved)
        artifact=detail['artifacts']['monthly_total'];path=Path(artifact['path']);path.write_bytes(b'tampered')
        assert client.get(f'/api/runs/{run}/artifacts/monthly_total').status_code==400
        assert client.get(f'/api/runs/{run}/export').status_code==400


def test_csv_formula_strings_are_inert_but_negative_money_stays_numeric(tmp_path):
    bootstrap(tmp_path);runner=LangChainRunner(tmp_path)
    state=runner.store.create('csv export','replay','cash-net-v1','cash-net-v1')
    folder=Path(tmp_path)/'var/workspaces'/state['run_id']/'artifacts';folder.mkdir()
    path=folder/'artifact_sample.parquet';pq.write_table(pa.table({'text':['=1+1'],'net_cents':[-123]}),path)
    state['artifacts']['sample']={'artifact_id':'artifact_sample','path':str(path),'sha256':file_hash(path)};runner.store.save(state)
    data,_,_=Workspace(tmp_path).artifact_download(state['run_id'],'sample')
    rows=list(csv.reader(io.StringIO(data.decode('utf-8-sig'))))
    assert rows[1]==["'=1+1",'-123']


def test_new_cli_parser_and_export_refuse_overwrite(tmp_path):
    args=parser().parse_args(['--project-root',str(tmp_path),'datasets','list'])
    assert dispatch(args)['datasets']==[]
    with pytest.raises(SystemExit):parser().parse_args(['resume','run_'+'a'*32,'--answer','x','--question','y'])
    state=LangChainRunner(tmp_path).store.create('export','replay','cash-net-v1','cash-net-v1')
    destination=tmp_path/'existing.zip';destination.write_bytes(b'preserve')
    args=parser().parse_args(['--project-root',str(tmp_path),'export',state['run_id'],'--output',str(destination)])
    with pytest.raises(FileExistsError):dispatch(args)
    assert destination.read_bytes()==b'preserve'
