import pytest


@pytest.fixture(autouse=True)
def isolate_local_workspaces(tmp_path, monkeypatch):
    monkeypatch.setenv("WEIXIN_PROJECT_DIR", str(tmp_path / "projects"))
