"""Independent runner using LangChain for both live and replay execution."""
from pathlib import Path

from local_data_agent.artifacts import checked_run_dir, file_hash
from local_data_agent.runner import AgentRunner as DomainRunner
from local_data_agent.state import RunBusy
from local_data_agent.tools import ToolContext

from .model import ModelGateway
from .runtime import run_agent


class LangChainRunner(DomainRunner):
    def __init__(self, project_root=None, model=None, *, max_requests=16, cancel_check=None):
        root=independent_root(project_root or default_root())
        super().__init__(root,model=model)
        if not 1<=max_requests<=16:raise ValueError('max_requests must be 1..16')
        self.max_requests=max_requests
        self.cancel_check=cancel_check

    def _client(self,state):
        if isinstance(self.model,ModelGateway):return self.model
        return ModelGateway(self.root,mode=state['mode'],case=state['case'],chat_model=self.model)

    def result(self,state):
        return {**super().result(state),'framework':'langchain.create_agent',
                'langchain_checkpoint':state.get('langchain_checkpoint'),
                'narrative_verification':state.get('narrative_verification'),
                'model_sessions':state.get('model_sessions',[])}

    def _execute(self,state,model=None):
        with self.store.acquire(state['run_id']) as attempt:
            latest=self.store.load(state['run_id'])
            if latest['updated_at']!=state['updated_at']:raise RunBusy('Run changed; reload before resuming')
            for snapshot in state['manifest']['sources']:
                if file_hash(snapshot['path'])!=snapshot['sha256']:raise ValueError('Frozen source hash mismatch')
            if model is None:model=self._client(state)
            identity=model.metadata()
            state['model']=identity
            state.setdefault('model_sessions',[]).append({'attempt_id':attempt,'model':identity,
                                                        'starting_request_number':state['model_requests']})
            state.update(attempt_id=attempt,status='RUNNING',stop_reason=None,framework='langchain.create_agent')
            state.pop('error',None)
            context=ToolContext(self.root,checked_run_dir(self.root,state['run_id']),state,state['manifest'],self.store,model)
            context.cancel_check=self.cancel_check
            self.store.save(state,'attempt_started',{'attempt_id':attempt,'framework':'langchain'})
            try:run_agent(context,max_requests=self.max_requests)
            finally:
                context.retriever.close()
                if self.model is None:model.close()


def independent_root(root):
    resolved=Path(root).resolve()
    if any(p.name.casefold()=='local_data_agent_demo' for p in [resolved,*resolved.parents]):
        raise ValueError('The original project is protected; choose a separate runtime directory')
    original=Path(__file__).resolve().parents[2].parent/'local_data_agent_demo'
    if resolved.is_relative_to(original.resolve()):
        raise ValueError('The original project is protected; choose this edition or a separate runtime directory')
    return resolved


def default_root():
    source_root=Path(__file__).resolve().parents[2]
    if (source_root/'src/langchain_data_agent').resolve()==Path(__file__).resolve().parent and (source_root/'pyproject.toml').is_file():
        return source_root
    return Path.cwd().resolve()
