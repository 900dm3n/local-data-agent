"""Human-facing dataset catalog and atomic local package import."""
from contextlib import closing
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time

from local_data_agent.artifacts import write_json
from local_data_agent.connectors import contained, discover_schema, readonly_sqlite, snapshot_sources
from local_data_agent.validation import validate_sources

IDENTIFIER=re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z')


def dataset_id(value):
    if not isinstance(value,str) or not IDENTIFIER.fullmatch(value):raise ValueError('Invalid dataset identifier')
    return value


def read_registry(root,fixture):
    base=(Path(root)/'var/sources').resolve()
    path=contained(base/dataset_id(fixture)/'registry.json',base)
    return path,json.loads(path.read_text(encoding='utf-8'))


def catalog(root,fixture=None):
    base=Path(root)/'var/sources'
    if fixture is None:
        rows=[]
        if not base.exists():return rows
        for path in sorted(base.glob('*/registry.json')):
            if not IDENTIFIER.fullmatch(path.parent.name):continue
            try:
                _,r=read_registry(root,path.parent.name)
                rows.append({'fixture':path.parent.name,'sources':len(r['sources']),
                             'documents':len(r.get('documents',[])),
                             'metrics':[c.get('metric_id') for c in r.get('metric_contracts',[]) if c.get('status')=='APPROVED']})
            except (OSError,ValueError,KeyError,TypeError):
                rows.append({'fixture':path.parent.name,'error':'Invalid or unreadable registry'})
        return rows
    path,registry=read_registry(root,fixture)
    sources=[]
    for source in registry['sources']:
        original=Path(source['path'])
        original=contained(original if original.is_absolute() else path.parent/original,base)
        sources.append({'source_id':source['source_id'],'kind':source['kind'],
                        'schema':discover_schema(original,source['kind'],source['source_id'])})
    return {'fixture':fixture,'sources':sources,'documents':[{k:v for k,v in d.items() if k!='path'} for d in registry.get('documents',[])],
            'metric_contracts':registry.get('metric_contracts',[])}


def import_package(root,registry_path,fixture):
    """Copy only declared, contained data assets; validate before atomic publish.

    Registry/documents are data. No scripts, shell commands or SQL instructions
    from a package are executed. Existing approved contracts are validated by
    the same domain core; missing approval remains a clarification requirement.
    """
    fixture=dataset_id(fixture);root=Path(root).resolve()
    source_registry=Path(registry_path).resolve()
    if source_registry.stat().st_size>1024*1024:raise ValueError('Registry exceeds 1 MiB')
    data=json.loads(source_registry.read_text(encoding='utf-8'))
    if not isinstance(data,dict) or not isinstance(data.get('sources'),list) or not 1<=len(data['sources'])<=32:
        raise ValueError('Package needs 1..32 registered sources')
    documents=data.get('documents',[])
    if not isinstance(documents,list) or len(documents)>128:raise ValueError('Package exceeds document limit')
    contracts=data.get('metric_contracts',[])
    if not isinstance(contracts,list) or len(contracts)>16 or any(not isinstance(c,dict) for c in contracts):
        raise ValueError('Invalid metric contracts')
    approved=[c for c in contracts if c.get('status')=='APPROVED']
    expected={'metric_id':'CASH_NET_v1','unit':'CNY_cents','timezone':'Australia/Sydney',
              'payment_status':'paid','refund_status':'succeeded','refund_channel':'original_order.channel_code',
              'event_time':{'orders':'paid_at','refunds':'refunded_at'},'missing_channel_policy':'reject'}
    if len(approved)>1 or any(any(c.get(k)!=v for k,v in expected.items()) for c in approved):
        raise ValueError('Imported approved contract must match the implemented CASH_NET_v1 definition')
    base=root/'var/sources';base.mkdir(parents=True,exist_ok=True)
    target=base/fixture
    if target.exists() or target.is_symlink():raise ValueError('Dataset already exists; choose another identifier')
    assets=[];total=0;identifiers=set()
    for group,items,key in [('sources',data['sources'],'source_id'),('documents',documents,'document_id')]:
        for index,item in enumerate(items):
            if not isinstance(item,dict):raise ValueError('Asset metadata must be an object')
            identifier=item.get(key)
            if not isinstance(identifier,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',identifier):raise ValueError('Invalid asset identifier')
            if (group,identifier) in identifiers:raise ValueError('Duplicate asset identifier')
            identifiers.add((group,identifier))
            original=Path(item['path'])
            original=contained(original if original.is_absolute() else source_registry.parent/original,source_registry.parent)
            if not original.is_file():raise ValueError('Missing asset')
            size=original.stat().st_size;total+=size
            if size>128*1024*1024 or total>256*1024*1024:raise ValueError('Package exceeds size limit')
            kind=item.get('kind') if group=='sources' else 'document'
            if kind not in {'sqlite','csv','parquet','jsonl','document'}:raise ValueError('Unsupported source kind')
            if kind=='document' and size>2*1024*1024:raise ValueError('Document exceeds 2 MiB')
            suffix={'sqlite':'.sqlite','csv':'.csv','parquet':'.parquet','jsonl':'.jsonl','document':'.md'}[kind]
            assets.append((group,item,original,kind,f'{group}_{index}{suffix}'))
    with tempfile.TemporaryDirectory(prefix='.import-',dir=base) as temporary:
        stage=Path(temporary)
        registry={'fixture':fixture,'sources':[],'documents':[],
                  'metric_contracts':deepcopy(data.get('metric_contracts',[]))}
        copied_bytes=0
        for group,item,original,kind,name in assets:
            destination=stage/name
            if kind=='sqlite':
                with closing(readonly_sqlite(original)) as source_db,closing(sqlite3.connect(destination)) as target_db:
                    deadline=time.monotonic()+30
                    def progress(status,remaining,total):
                        if time.monotonic()>deadline:raise TimeoutError('SQLite import backup exceeded 30 seconds')
                    source_db.backup(target_db,pages=1000,progress=progress)
            else:shutil.copyfile(original,destination)
            # A small SQLite main file can have a much larger committed WAL.
            # Enforce accepted package limits against the consistent copy too.
            copied_bytes+=destination.stat().st_size
            if destination.stat().st_size>128*1024*1024 or copied_bytes>256*1024*1024:
                raise ValueError('Consistent package copy exceeds size limit')
            if kind=='document':destination.read_text(encoding='utf-8-sig')
            registry[group].append({**{k:v for k,v in item.items() if k in {'source_id','document_id','kind','status','metric_id','title','version','tables'}},'path':name})
        write_json(stage/'registry.json',registry)
        validation_dir=stage/'validation'
        manifest=snapshot_sources(root,validation_dir,stage.name)
        validation=validate_sources(manifest)
        allowed={'approved_metric_available','metric_unit_and_timezone'} if validation.get('needs_clarification') else set()
        failures=[c for c in validation['checks'] if not c['passed'] and c['name'] not in allowed]
        if failures:raise ValueError('Source contract validation failed: '+','.join(c['name'] for c in failures))
        # Checked exact staging child; never remove a caller-supplied path.
        if validation_dir.resolve().parent!=stage.resolve():raise ValueError('Invalid validation directory')
        shutil.rmtree(validation_dir)
        write_json(stage/'import_validation.json',validation)
        if target.exists() or target.is_symlink():raise ValueError('Dataset already exists')
        os.rename(stage,target)
    return {'status':'SUCCEEDED','fixture':fixture,'validation':validation,'registry_path':str(target/'registry.json')}
