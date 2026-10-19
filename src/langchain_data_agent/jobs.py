"""Single-worker durable local jobs; interrupted API calls are never auto-retried."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import uuid

from local_data_agent.artifacts import dumps
from local_data_agent.cli import _redact_result, _safe_error
from local_data_agent.state import pid_alive, utc_now
from .runner import LangChainRunner, independent_root
from .datasets import dataset_id

FINAL={'SUCCEEDED','FAILED','CANCELLED','WAITING_INPUT','INTERRUPTED'}


def validated_job(kind,payload):
    fields={'run':({'question','fixture','mode','case'},{'question','fixture','mode'}),
            'resume':({'run_id','answer','question'},{'run_id'}),
            'bootstrap':({'fixture','seed'},{'fixture'}),
            'generate':({'fixture','seed','profile','order_count','ticket_count','document_count'},{'fixture'}),
            'import':({'fixture','registry_path'},{'fixture','registry_path'}),
            'approve_metric':({'run_id','approved_by'},{'run_id','approved_by'}),
            'approve_memory':({'memory_id','approved_by'},{'memory_id','approved_by'})}
    if kind not in fields or not isinstance(payload,dict):raise ValueError('Unknown job kind')
    allowed,required=fields[kind]
    if not required.issubset(payload) or set(payload)-allowed:raise ValueError('Invalid job fields')
    value=dict(payload)
    for key in ['question','answer','approved_by','registry_path']:
        if key in value and (not isinstance(value[key],str) or not value[key].strip() or len(value[key])>8000):
            raise ValueError(f'Invalid {key}')
    if 'fixture' in value:dataset_id(value['fixture'])
    if 'run_id' in value and not re.fullmatch(r'run_[a-f0-9]{32}',str(value['run_id'])):raise ValueError('Invalid run_id')
    if 'memory_id' in value and not re.fullmatch(r'mem_[a-f0-9]{32}',str(value['memory_id'])):raise ValueError('Invalid memory_id')
    if kind=='run':
        if value['mode'] not in {'replay','live'}:raise ValueError('Invalid mode')
        case=value.setdefault('case','cash-net-v1')
        if case not in {'cash-net-v1','exploration','repair','budget','clarification'}:raise ValueError('Invalid declared replay case')
        if value['mode']=='replay' and value['fixture']!='cash-net-v1':
            raise ValueError('Declared replay is bound to cash-net-v1; use live for other datasets')
    if kind=='resume' and 'question' in value and 'answer' in value:raise ValueError('Choose answer or follow-up, not both')
    for key,lo,hi in [('seed',0,2**31-1),('order_count',12000,100000),('ticket_count',0,3000),('document_count',40,200)]:
        if key in value and (type(value[key]) is not int or not lo<=value[key]<=hi):raise ValueError(f'Invalid {key}')
    if 'profile' in value and value['profile'] not in {'baseline','empty-base','channel-gap'}:raise ValueError('Invalid profile')
    return value


class JobStore:
    def __init__(self,root):
        self.root=independent_root(root);self.path=self.root/'var/jobs.sqlite'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.db() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs(
                  job_id TEXT PRIMARY KEY,kind TEXT NOT NULL,payload TEXT NOT NULL,status TEXT NOT NULL,
                  created_at TEXT NOT NULL,updated_at TEXT NOT NULL,owner_pid INTEGER,
                  cancel_requested INTEGER NOT NULL DEFAULT 0,run_id TEXT,result TEXT,error TEXT);
                CREATE TABLE IF NOT EXISTS service_lock(slot INTEGER PRIMARY KEY,owner_pid INTEGER,token TEXT);
            ''')

    @contextmanager
    def db(self):
        db=sqlite3.connect(self.path,timeout=15);db.row_factory=sqlite3.Row
        try:
            with db:yield db
        finally:db.close()

    @staticmethod
    def row(row):
        value=dict(row)
        for field in ['payload','result']:
            if value[field] is not None:value[field]=json.loads(value[field])
        value['cancel_requested']=bool(value['cancel_requested'])
        return value

    def get(self,job_id):
        with self.db() as db:row=db.execute('SELECT * FROM jobs WHERE job_id=?',(job_id,)).fetchone()
        if row is None:raise ValueError('Unknown job_id')
        return self.row(row)

    def list(self,limit=50):
        if type(limit) is not int or not 1<=limit<=200:raise ValueError('Invalid limit')
        with self.db() as db:rows=db.execute('SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?',(limit,)).fetchall()
        return [self.row(row) for row in rows]

    def submit(self,kind,payload):
        payload=validated_job(kind,payload);identifier='job_'+uuid.uuid4().hex;stamp=utc_now()
        with self.db() as db:
            # A second queued follow-up must not accidentally run against a
            # different intermediate question/version than the user observed.
            run_id=payload.get('run_id')
            if run_id and db.execute("SELECT 1 FROM jobs WHERE run_id=? AND status IN ('QUEUED','RUNNING')",(run_id,)).fetchone():
                raise ValueError('This run already has an active job')
            db.execute('INSERT INTO jobs(job_id,kind,payload,status,created_at,updated_at,run_id) VALUES(?,?,?,?,?,?,?)',
                       (identifier,kind,dumps(payload),'QUEUED',stamp,stamp,run_id))
        return self.get(identifier)

    def acquire_service(self):
        token=uuid.uuid4().hex
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT owner_pid FROM service_lock WHERE slot=1').fetchone()
            if row and pid_alive(row[0]):raise RuntimeError('A workspace worker is already running for this project')
            db.execute('INSERT OR REPLACE INTO service_lock VALUES(1,?,?)',(os.getpid(),token))
            # Preserve runs and effects. Never automatically retry a model
            # attempt for which the old process did not record a final result.
            db.execute("UPDATE jobs SET status='INTERRUPTED',updated_at=?,error=? WHERE status='RUNNING'",
                       (utc_now(),'Workspace process ended; inspect the run and resume explicitly.'))
        return token

    def release_service(self,token):
        with self.db() as db:db.execute('DELETE FROM service_lock WHERE slot=1 AND token=?',(token,))

    def claim(self):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute("SELECT job_id FROM jobs WHERE status='QUEUED' ORDER BY created_at LIMIT 1").fetchone()
            if row is None:return None
            db.execute("UPDATE jobs SET status='RUNNING',owner_pid=?,updated_at=? WHERE job_id=?",(os.getpid(),utc_now(),row[0]))
        return self.get(row[0])

    def attach_run(self,job_id,run_id):
        with self.db() as db:db.execute('UPDATE jobs SET run_id=?,updated_at=? WHERE job_id=?',(run_id,utc_now(),job_id))

    def cancel(self,job_id):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT status FROM jobs WHERE job_id=?',(job_id,)).fetchone()
            if row is None:raise ValueError('Unknown job_id')
            if row[0] not in FINAL:
                db.execute("UPDATE jobs SET cancel_requested=1,status=CASE WHEN status='QUEUED' THEN 'CANCELLED' ELSE status END,updated_at=? WHERE job_id=?",(utc_now(),job_id))
        return self.get(job_id)

    def finish(self,job_id,result=None,error=None):
        result=_redact_result(result) if result is not None else None
        status='FAILED' if error is not None else str((result or {}).get('status','SUCCEEDED')).upper()
        if status not in FINAL:status='SUCCEEDED'
        with self.db() as db:
            db.execute('UPDATE jobs SET status=?,updated_at=?,owner_pid=NULL,result=?,error=?,run_id=COALESCE(?,run_id) WHERE job_id=?',
                       (status,utc_now(),dumps(result) if result is not None else None,error,(result or {}).get('run_id'),job_id))


def execute_job(root,job,cancel,attach):
    payload=job['payload'];kind=job['kind']

    class ManagedRunner(LangChainRunner):
        def _client(self,state):attach(state['run_id']);return super()._client(state)
        def _execute(self,state,model=None):attach(state['run_id']);return super()._execute(state,model)

    if cancel():return {'status':'CANCELLED'}
    if kind in {'run','resume','approve_metric'}:
        runner=ManagedRunner(root,cancel_check=cancel)
        if kind=='run':return runner.run(**payload)
        if kind=='resume':return runner.resume(**payload)
        return {**runner.approve_metric(payload['run_id'],'CASH_NET_v1',payload['approved_by']),'status':'SUCCEEDED'}
    if kind=='bootstrap':
        from local_data_agent.fixtures import bootstrap
        bootstrap(root,**payload);return {'status':'SUCCEEDED','fixture':payload['fixture']}
    if kind=='generate':
        from local_data_agent.holdout_fixtures import generate_holdout
        generate_holdout(root,**{'seed':73,'profile':'baseline',**payload})
        return {'status':'SUCCEEDED','fixture':payload['fixture']}
    if kind=='import':
        from .datasets import import_package
        return import_package(root,**payload)
    if kind=='approve_memory':
        from local_data_agent.memory import MemoryStore
        return {'status':'SUCCEEDED','card':MemoryStore(root).approve(**payload)}
    raise ValueError('Unknown job kind')


class JobManager:
    def __init__(self,root,*,executor=execute_job):
        self.root=independent_root(root);self.store=JobStore(self.root);self.executor=executor
        self.token=self.store.acquire_service();self.stopped=threading.Event();self.wake=threading.Event()
        self.thread=threading.Thread(target=self._work,name='data-agent-worker',daemon=True);self.thread.start()

    def submit(self,kind,payload):
        result=self.store.submit(kind,payload);self.wake.set();return result

    def _work(self):
        while not self.stopped.is_set():
            job=self.store.claim()
            if job is None:self.wake.wait(.5);self.wake.clear();continue
            identifier=job['job_id']
            try:
                result=self.executor(self.root,job,lambda:self.store.get(identifier)['cancel_requested'],
                                     lambda run_id:self.store.attach_run(identifier,run_id))
                self.store.finish(identifier,result=result)
            except Exception as error:self.store.finish(identifier,error=_safe_error(error))

    def close(self):
        self.stopped.set();self.wake.set()
        for job in self.store.list(200):
            if job['status']=='RUNNING':self.store.cancel(job['job_id'])
        self.thread.join(timeout=2)
        # Keep ownership while a cooperative cancellation is still in flight.
        if not self.thread.is_alive():self.store.release_service(self.token)
