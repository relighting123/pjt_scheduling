"""
tests/test_db_registry_envs.py

config/databases.yaml 하나에 envs: 블록으로 prd/dev 접속정보를 함께 넣고
APP_ENV 로 골라 쓸 수 있어야 한다. envs: 없는 레거시 단일 파일도 그대로 동작해야 한다.
"""
from pathlib import Path

import data.db_registry as db_registry

_SINGLE_FILE_WITH_ENVS = """
default: Prd

envs:
  production:
    Prd:
      user: prd_user
      password: prd_password
      dsn: prd-host:1521/ORCL
  development:
    Prd:
      user: dev_user
      password: dev_password
      dsn: dev-host:1521/ORCL

Common:
  user: shared_user
  password: shared_password
  dsn: shared-host:1521/ORCL
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config" / "databases.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_envs_section_selects_production_credentials(tmp_path, monkeypatch):
    path = _write(tmp_path, _SINGLE_FILE_WITH_ENVS)
    monkeypatch.setenv("APP_ENV", "production")

    buckets, yaml_default = db_registry.load_db_aliases_from_yaml(path)

    assert buckets["prd"] == {
        "user": "prd_user", "password": "prd_password", "dsn": "prd-host:1521/ORCL",
    }
    assert buckets["common"]["user"] == "shared_user"
    assert yaml_default == "prd"


def test_envs_section_selects_development_credentials(tmp_path, monkeypatch):
    path = _write(tmp_path, _SINGLE_FILE_WITH_ENVS)
    monkeypatch.setenv("APP_ENV", "development")

    buckets, _ = db_registry.load_db_aliases_from_yaml(path)

    assert buckets["prd"] == {
        "user": "dev_user", "password": "dev_password", "dsn": "dev-host:1521/ORCL",
    }
    assert buckets["common"]["user"] == "shared_user"


def test_envs_section_aliases_prod_and_dev_synonyms(tmp_path, monkeypatch):
    path = _write(tmp_path, _SINGLE_FILE_WITH_ENVS)
    monkeypatch.setenv("APP_ENV", "prod")

    buckets, _ = db_registry.load_db_aliases_from_yaml(path)

    assert buckets["prd"]["user"] == "prd_user"


def test_no_app_env_leaves_envs_section_unmerged(tmp_path, monkeypatch):
    path = _write(tmp_path, _SINGLE_FILE_WITH_ENVS)
    monkeypatch.delenv("APP_ENV", raising=False)

    buckets, _ = db_registry.load_db_aliases_from_yaml(path)

    assert "prd" not in buckets
    assert buckets["common"]["user"] == "shared_user"


def test_diagnose_flags_app_env_missing_when_envs_section_present(tmp_path, monkeypatch):
    path = _write(tmp_path, _SINGLE_FILE_WITH_ENVS)
    monkeypatch.delenv("APP_ENV", raising=False)

    report = db_registry.diagnose_db_config(yaml_path=path)

    assert any("envs:" in issue and "APP_ENV" in issue for issue in report["issues"])


def test_default_db_config_path_is_always_the_single_file(tmp_path, monkeypatch):
    monkeypatch.setattr(db_registry, "BASE_DIR", tmp_path)
    _write(tmp_path, _SINGLE_FILE_WITH_ENVS)
    monkeypatch.delenv("DB_CONFIG", raising=False)
    monkeypatch.setenv("APP_ENV", "production")

    resolved = db_registry.default_db_config_path()

    assert resolved == tmp_path / "config" / "databases.yaml"


def test_db_config_env_var_overrides_the_default_path(tmp_path, monkeypatch):
    monkeypatch.setattr(db_registry, "BASE_DIR", tmp_path)
    other_path = tmp_path / "elsewhere" / "custom.yaml"
    monkeypatch.setenv("DB_CONFIG", str(other_path))
    monkeypatch.setenv("APP_ENV", "production")

    resolved = db_registry.default_db_config_path()

    assert resolved == other_path
