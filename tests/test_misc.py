import sys

from smart_photo_edit import config, paths
from smart_photo_edit.launcher import split_command


def test_config_roundtrip_and_corrupt_file(tmp_path):
    p = tmp_path / "c.json"
    s = config.Settings(comfy_url="127.0.0.1:9", workflow="x", workflow_params={"x": {"a": 1}})
    config.save(s, p)
    again = config.load(p)
    assert again.comfy_url == "http://127.0.0.1:9" and again.workflow_params == {"x": {"a": 1}}
    p.write_text("lixo{", encoding="utf-8")
    assert config.load(p).workflow == config.DEFAULT_WORKFLOW


def test_unknown_keys_in_config_are_ignored(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"workflow": "a", "campo_futuro": 1}', encoding="utf-8")
    assert config.load(p).workflow == "a"


def test_env_override_for_url(tmp_path, monkeypatch):
    monkeypatch.setenv("SPE_COMFY_URL", "http://10.0.0.5:8188/")
    assert config.load(tmp_path / "none.json").comfy_url == "http://10.0.0.5:8188"


def test_user_dir_follows_override(isolated_home):
    assert paths.user_data_dir() == isolated_home


def test_user_dir_per_platform(monkeypatch):
    monkeypatch.delenv("SPE_HOME")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", r"C:\Users\x\AppData\Roaming")
    assert paths.user_data_dir().name == "SmartPhotoEdit"
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/xdg")
    assert str(paths.user_data_dir()) == "/tmp/xdg/smart-photo-edit"


def test_split_command_keeps_quoted_paths(monkeypatch):
    assert split_command('python "/a b/main.py" --port 1') == ["python", "/a b/main.py", "--port", "1"]
