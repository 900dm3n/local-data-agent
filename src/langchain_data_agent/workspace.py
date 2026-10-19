"""Bounded workspace views and verified exports, shared by HTTP and CLI."""
import csv
import io
import hashlib
import json
from pathlib import Path
import re
import tempfile
import zipfile

import pyarrow.parquet as pq

from local_data_agent.artifacts import checked_run_dir, dumps, file_hash, jsonable, verified_artifact
from local_data_agent.cli import _redact_result
from local_data_agent.state import StateStore
from .runner import independent_root


class Workspace:
    def __init__(self,root):
        self.root=independent_root(root);self.store=StateStore(self.root)

    def runs(self,limit=50):
        if type(limit) is not int or not 1<=limit<=200:raise ValueError('Invalid limit')
        with self.store.connect() as db:
            rows=db.execute('SELECT payload FROM runs ORDER BY json_extract(payload,\'$.updated_at\') DESC LIMIT ?',(limit,)).fetchall()
        fields=['run_id','question','fixture','mode','status','created_at','updated_at','plan_version','model_requests','stop_reason','model']
        return [_redact_result({k:s.get(k) for k in fields}) for s in (json.loads(r[0]) for r in rows)]

    def run(self,run_id):
        state=self.store.load(run_id)
        fields=['run_id','question','fixture','mode','status','created_at','updated_at','plan_version','plan','artifacts',
                'model_requests','stop_reason','error','model','pending_question','metric_binding','publications',
                'langchain_checkpoint','narrative_verification','repair_counts','observed_facts','open_questions','memory_draft','model_sessions']
        usage=state['usage'];agent=[u for u in usage if u.get('kind')=='agent']
        return _redact_result({**{k:state.get(k) for k in fields},
            'usage_summary':{'received_entries':len(usage),'agent_attempts':state['model_requests'],
                'agent_entries_with_reported_usage':sum('prompt_tokens' in u for u in agent),
                'totals':{k:sum(u.get(k,0) or 0 for u in usage) for k in ['prompt_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens','completion_tokens']},
                'note':'Only returned provider usage is summed; missing usage is unknown. Replay has no API usage.'}})

    def events(self,run_id,after=0,limit=100):
        self.store.load(run_id)
        if type(after) is not int or after<0 or type(limit) is not int or not 1<=limit<=200:raise ValueError('Invalid event cursor')
        with self.store.connect() as db:
            rows=db.execute('SELECT event_id,kind,payload,created_at FROM events WHERE run_id=? AND event_id>? ORDER BY event_id LIMIT ?',
                            (run_id,after,limit)).fetchall()
        return _redact_result([{'event_id':r[0],'kind':r[1],'payload':json.loads(r[2]),'created_at':r[3]} for r in rows])

    def artifact(self,run_id,identifier):
        state=self.store.load(run_id)
        a=next((a for key,a in state['artifacts'].items() if identifier in {key,a['artifact_id']}),None)
        if a is None:raise ValueError('Unknown registered artifact')
        return a,verified_artifact(checked_run_dir(self.root,run_id),a)

    def preview(self,run_id,identifier,limit=100):
        if type(limit) is not int or not 1<=limit<=200:raise ValueError('Invalid preview limit')
        a,path=self.artifact(run_id,identifier)
        file=pq.ParquetFile(path)
        batch=next(file.iter_batches(batch_size=limit),None)
        return _redact_result({'artifact_id':a['artifact_id'],'output_name':a['output_name'],'row_count':file.metadata.num_rows,
                              'columns':file.schema_arrow.names,'rows':jsonable(batch.to_pylist() if batch else [])})

    def artifact_download(self,run_id,identifier,format='csv'):
        _,path=self.artifact(run_id,identifier)
        if format=='parquet':return path.read_bytes(),'application/vnd.apache.parquet',path.name
        if format!='csv':raise ValueError('Supported export formats: csv, parquet')
        file=pq.ParquetFile(path)
        if file.metadata.num_rows>10000:raise ValueError('Artifact exceeds export row limit')
        def inert(value):
            return "'"+value if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')) else value
        stream=io.StringIO(newline='');writer=csv.writer(stream);writer.writerow([inert(n) for n in file.schema_arrow.names])
        for batch in file.iter_batches(batch_size=1000):
            for row in batch.to_pylist():
                values=[]
                for name in file.schema_arrow.names:
                    value=jsonable(row[name])
                    if isinstance(value,(dict,list)):value=dumps(value)
                    # Keep string data inert when CSV is opened by a spreadsheet.
                    values.append(inert(value))
                writer.writerow(values)
        return ('\ufeff'+stream.getvalue()).encode('utf-8'),'text/csv; charset=utf-8',path.stem+'.csv'

    def publication_files(self,run_id,publication_id=None):
        state=self.store.load(run_id);items=state.get('publications',[])
        if not items:raise ValueError('This run has no verified publication')
        publication=next((p for p in items if p['publication_id']==publication_id),None) if publication_id else items[-1]
        if publication is None:raise ValueError('Unknown publication')
        run=checked_run_dir(self.root,run_id)
        folder=(run/'publications'/publication['publication_id']).resolve()
        if folder.parent!=(run/'publications').resolve():raise ValueError('Invalid publication path')
        files={}
        for name in ['report.md','claims.json','validation.json','lineage.json','evidence.json','table_narrative.json','analysis.json']:
            path=(folder/name).resolve()
            if path.parent!=folder:raise ValueError('Publication path escapes its directory')
            if path.is_file():files[name]=path
        if 'report.md' not in files:raise ValueError('Missing publication report')
        return publication,files

    def report(self,run_id,publication_id=None):
        publication,files=self.publication_files(run_id,publication_id)
        result={'publication_id':publication['publication_id'],'markdown':files['report.md'].read_text(encoding='utf-8')}
        for name in ['claims.json','validation.json','table_narrative.json','analysis.json']:
            if name in files:result[name.removesuffix('.json')]=json.loads(files[name].read_text(encoding='utf-8'))
        return _redact_result(result)

    def export_bundle(self,run_id):
        """Downloadable audit/result bundle, excluding raw sources and secrets."""
        state=self.store.load(run_id);run=checked_run_dir(self.root,run_id)
        manifest=[];total=0
        artifacts={a['artifact_id']:a for a in state['artifacts'].values()}
        publications=[];missing_metadata=[]
        for publication in state.get('publications',[]):
            _,files=self.publication_files(run_id,publication['publication_id']);publications.append((publication,files))
            if 'lineage.json' not in files:continue
            lineage=json.loads(files['lineage.json'].read_text(encoding='utf-8'))
            for node in lineage.get('nodes',[]):
                identifier=node.get('id')
                if node.get('kind')!='artifact' or identifier in artifacts:continue
                if not isinstance(identifier,str) or not re.fullmatch(r'artifact_[a-f0-9]{32}',identifier):raise ValueError('Invalid historical artifact identifier')
                meta=(run/'artifacts'/f'{identifier}.json').resolve()
                if not meta.is_relative_to(run):raise ValueError('Artifact metadata escapes run')
                if not meta.is_file():missing_metadata.append(identifier);continue
                a=json.loads(meta.read_text(encoding='utf-8'))
                if a.get('artifact_id')!=identifier:raise ValueError('Historical artifact metadata mismatch')
                artifacts[identifier]=a
        with tempfile.SpooledTemporaryFile(max_size=8*1024*1024) as stream:
            with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
                def put_json(name,value):
                    encoded=dumps(value).encode('utf-8');archive.writestr(name,encoded)
                    manifest.append({'name':name,'sha256':hashlib.sha256(encoded).hexdigest()})
                put_json('run.json',self.run(run_id))
                put_json('source_fingerprints.json',{
                    'sources':[{k:v for k,v in s.items() if k in {'source_id','kind','sha256','schema'}} for s in state.get('manifest',{}).get('sources',[])],
                    'documents':[{k:v for k,v in d.items() if k in {'document_id','sha256','status','metric_id'}} for d in state.get('manifest',{}).get('documents',[])],
                    'metric_contract':state.get('bound_contract')})
                for a in artifacts.values():
                    path=verified_artifact(run,a);total+=path.stat().st_size
                    if total>256*1024*1024:raise ValueError('Export exceeds 256 MiB')
                    filename=f'artifacts/{a["artifact_id"]}.parquet'
                    archive.write(path,filename);manifest.append({'name':filename,'sha256':file_hash(path)})
                    put_json(f'artifacts/{a["artifact_id"]}.json',a)
                for record in sorted((run/'plans').glob('plan_*.json')):
                    if not record.resolve().is_relative_to(run):raise ValueError('Plan record escapes run')
                    total+=record.stat().st_size
                    if total>256*1024*1024:raise ValueError('Export exceeds 256 MiB')
                    name='plans/'+record.name;archive.write(record,name);manifest.append({'name':name,'sha256':file_hash(record)})
                for publication,files in publications:
                    for name,path in files.items():
                        total+=path.stat().st_size
                        if total>256*1024*1024:raise ValueError('Export exceeds 256 MiB')
                        filename=f'publications/{publication["publication_id"]}/{name}'
                        archive.write(path,filename);manifest.append({'name':filename,'sha256':file_hash(path)})
                archive.writestr('export_manifest.json',dumps({'run_id':run_id,'files':manifest,
                    'unavailable_historical_artifact_metadata':sorted(set(missing_metadata)),
                    'scope':'Registered result artifacts and publications; raw inputs, provider configuration and credentials excluded.'}))
            stream.seek(0);return stream.read()
