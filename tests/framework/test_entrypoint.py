from pathlib import Path
import pytest
from langchain_data_agent.cli import main
from langchain_data_agent.runner import LangChainRunner,independent_root


def test_original_project_is_rejected_before_any_state_database_is_opened():
    original=Path(__file__).resolve().parents[3]/'local_data_agent_demo'
    with pytest.raises(ValueError,match='protected'):LangChainRunner(original)
    with pytest.raises(ValueError,match='protected'):independent_root(original/'var'/'new-run')


def test_doctor_entrypoint_reports_native_framework_without_network(tmp_path,capsys):
    assert main(['doctor','--project-root',str(tmp_path)])==0
    import json
    result=json.loads(capsys.readouterr().out)
    assert result['framework']['runtime']=='langchain.create_agent'
    assert not result['model']['network_called']


def test_mimo_public_identity_never_loads_credential(tmp_path,monkeypatch):
    import json
    from langchain_data_agent import model
    (tmp_path/'config.local.json').write_text(json.dumps({'provider':'mimo','model':'mimo-v2.5',
        'key_file':'/nonexistent/test-credential.txt'}),encoding='utf-8')
    def forbidden(*args,**kwargs):raise AssertionError('Credential loader must not run for display metadata')
    monkeypatch.setattr(model,'load_settings',forbidden)
    assert model.public_model_settings(tmp_path)=={'provider':'mimo','model':'mimo-v2.5'}
