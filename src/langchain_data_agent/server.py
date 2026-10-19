"""Loopback-only workbench with a same-origin JSON API and durable jobs."""
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
from pathlib import Path
import secrets
from urllib.parse import parse_qs, unquote, urlsplit

from local_data_agent.artifacts import dumps
from local_data_agent.cli import _redact_result, _safe_error
from local_data_agent.doctor import diagnose
from local_data_agent.memory import MemoryStore
from .datasets import catalog
from .jobs import JobManager
from .workspace import Workspace
from .model import public_model_settings


class WorkbenchServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,root,port=8765,*,manager=None):
        if type(port) is not int or not 0<=port<=65535:raise ValueError('Invalid port')
        self.workspace=Workspace(root);self.root=self.workspace.root
        self.manager=manager;self.owns_manager=manager is None;self.token=secrets.token_urlsafe(32)
        super().__init__(('127.0.0.1',port),WorkbenchHandler)
        try:
            if self.manager is None:self.manager=JobManager(self.root)
        except Exception:
            super().server_close();raise

    def server_close(self):
        super().server_close()
        if self.owns_manager and self.manager is not None:self.manager.close()


class WorkbenchHandler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.0'

    def log_message(self,*args):pass  # User questions, local paths and session tokens are not access logs.

    def send_data(self,data,status=200,mime='application/json; charset=utf-8',filename=None):
        if not isinstance(data,bytes):data=dumps(_redact_result(data)).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        if filename:self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
        self.end_headers();self.wfile.write(data)

    def trusted_request(self,write=False):
        port=self.server.server_port;host=self.headers.get('Host','')
        if host not in {f'127.0.0.1:{port}',f'localhost:{port}'}:return False
        origin=self.headers.get('Origin')
        if origin is not None and origin!=f'http://{host}':return False
        if self.headers.get('Sec-Fetch-Site')=='cross-site':return False
        if write:
            token=self.headers.get('X-Workspace-Token','')
            if not hmac.compare_digest(token,self.server.token):return False
            if self.headers.get('Content-Type','').split(';')[0].strip()!='application/json':return False
        return True

    def do_GET(self):self.handle_api(False)
    def do_POST(self):self.handle_api(True)

    def handle_api(self,write):
        try:
            if not self.trusted_request(write):self.send_data({'error':'Local same-origin access required'},403);return
            url=urlsplit(self.path);path=unquote(url.path);query=parse_qs(url.query)
            if write:
                if self.headers.get('Transfer-Encoding'):raise ValueError('Chunked bodies are unsupported')
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=65536:raise ValueError('JSON body must be 1..65536 bytes')
                payload=json.loads(self.rfile.read(size))
                if not isinstance(payload,dict):raise ValueError('JSON object required')
                self.post(path,payload);return
            self.get(path,query)
        except (BrokenPipeError,ConnectionResetError):pass
        except (ValueError,KeyError,TypeError,FileNotFoundError) as error:
            self.send_data({'error':_safe_error(error),'error_type':type(error).__name__},400)
        except Exception as error:
            self.send_data({'error':_safe_error(error),'error_type':type(error).__name__},500)

    def get(self,path,query):
        workspace=self.server.workspace
        if path in {'/','/app.js','/style.css'}:
            name={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}[path]
            mime={'index.html':'text/html; charset=utf-8','app.js':'text/javascript; charset=utf-8','style.css':'text/css; charset=utf-8'}[name]
            self.send_data(files('langchain_data_agent').joinpath('web',name).read_bytes(),mime=mime);return
        if path=='/api/session':
            self.send_data({'token':self.server.token,'framework':'langchain.create_agent','local_only':True,
                            'model':{**diagnose(self.server.root,probe=False)['model'],**public_model_settings(self.server.root)}});return
        if path=='/api/datasets':self.send_data(catalog(self.server.root));return
        if path.startswith('/api/datasets/'):
            self.send_data(catalog(self.server.root,path.removeprefix('/api/datasets/')));return
        if path=='/api/jobs':self.send_data(self.server.manager.store.list());return
        if path.startswith('/api/jobs/'):
            self.send_data(self.server.manager.store.get(path.removeprefix('/api/jobs/')));return
        if path=='/api/runs':self.send_data(workspace.runs());return
        if path=='/api/memory':self.send_data(MemoryStore(self.server.root).list_cards());return
        parts=path.strip('/').split('/')
        if len(parts)>=3 and parts[:2]==['api','runs']:
            run_id=parts[2]
            if len(parts)==3:self.send_data(workspace.run(run_id));return
            if len(parts)==4 and parts[3]=='events':
                self.send_data(workspace.events(run_id,int(query.get('after',['0'])[0])));return
            if len(parts)==4 and parts[3]=='report':
                self.send_data(workspace.report(run_id,query.get('publication_id',[None])[0]));return
            if len(parts)==4 and parts[3]=='export':
                self.send_data(workspace.export_bundle(run_id),mime='application/zip',filename=run_id+'.zip');return
            if len(parts) in {5,6} and parts[3]=='artifacts':
                if len(parts)==5:self.send_data(workspace.preview(run_id,parts[4]));return
                if parts[5]=='download':
                    data,mime,name=workspace.artifact_download(run_id,parts[4],query.get('format',['csv'])[0])
                    self.send_data(data,mime=mime,filename=name);return
        self.send_data({'error':'Unknown endpoint'},404)

    def post(self,path,payload):
        if path=='/api/jobs':
            if set(payload)!={'kind','payload'}:raise ValueError('Expected kind and payload')
            self.send_data(self.server.manager.submit(payload['kind'],payload['payload']),202);return
        parts=path.strip('/').split('/')
        if len(parts)==4 and parts[:2]==['api','jobs'] and parts[3]=='cancel':
            if payload:raise ValueError('Cancel body must be empty object')
            self.send_data(self.server.manager.store.cancel(parts[2]));return
        self.send_data({'error':'Unknown endpoint'},404)


def serve(root,port=8765):
    server=WorkbenchServer(root,port)
    print(json.dumps({'status':'serving','url':f'http://127.0.0.1:{server.server_port}',
                      'project_root':str(server.root),'network_probe':False}),flush=True)
    try:server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:pass
    finally:server.server_close()
    return {'status':'stopped'}
