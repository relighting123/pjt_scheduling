"""
data/db_registry.py – SQL 쿼리별 DB alias 및 계층 DB 설정

설정 파일 (권장): ``config/databases.yaml``
```yaml
default: Prd

Prd:
  user: ...
  password: ...
  dsn: ...

Dev:
  user: ...
  password: ...
  dsn: ...
  Mes:          # → alias ``dev.mes``
    user: ...
    password: ...
```

SQL: ``-- @db: Prd`` / ``-- @db: Dev.Mes``

.env:
    DB_CONFIG=config/databases.yaml
    DB_DEFAULT_ALIAS=Prd

하위 호환: ORACLE_USER/PASSWORD/DSN → alias ``main``
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import BASE_DIR

_DB_HEADER_RE = re.compile(
    r"^\s*--\s*@db:\s*([A-Za-z][\w.-]*(?:\.[A-Za-z][\w.-]*)*)\s*$",
    re.MULTILINE,
)
_ALIAS_ENV_RE = re.compile(r"^DB_([A-Z][A-Z0-9_]*)_(USER|PASSWORD|DSN)$")
_META_KEYS = frozenset({"default", "_default"})
_FIELD_ALIASES = {
    "user": "user",
    "userid": "user",
    "username": "user",
    "pw": "password",
    "pwd": "password",
    "password": "password",
    "pass": "password",
    "dsn": "dsn",
    "connect": "dsn",
    "connection": "dsn",
    "host": "dsn",
}


@dataclass(frozen=True)
class DbCredentials:
    alias: str
    user: str
    password: str
    dsn: str

    def connect(self):
        try:
            import oracledb
        except ImportError as e:
            raise ImportError(
                "oracledb 패키지가 필요합니다. pip install oracledb"
            ) from e
        if not self.user or not self.password or not self.dsn:
            raise ValueError(
                f"DB alias '{self.alias}' 접속 정보가 없습니다. "
                f"상위 alias 상속 후에도 user/password/dsn 이 부족합니다."
            )
        return oracledb.connect(
            user=self.user, password=self.password, dsn=self.dsn,
        )


def _normalize_alias(name: str) -> str:
    parts = [
        p.strip().lower().replace("-", "_")
        for p in name.strip().replace("/", ".").split(".")
    ]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError(f"DB alias 가 비어 있습니다: {name!r}")
    return ".".join(parts)


def _normalize_field(key: str) -> Optional[str]:
    return _FIELD_ALIASES.get(key.strip().lower().replace("-", "_"))


def _is_credential_leaf(node: dict) -> bool:
    return any(_normalize_field(k) for k in node)


def _walk_yaml_node(
    node: dict,
    prefix: List[str],
    buckets: Dict[str, Dict[str, str]],
) -> None:
    """YAML 트리 → alias별 credential 필드 (경로 = alias)."""
    direct: Dict[str, str] = {}
    children: Dict[str, dict] = {}

    for key, value in node.items():
        if key in _META_KEYS:
            continue
        field = _normalize_field(key)
        if field and not isinstance(value, dict):
            direct[field] = str(value).strip()
            continue
        if isinstance(value, dict):
            children[key] = value

    if direct:
        alias = _normalize_alias(".".join(prefix)) if prefix else "root"
        bucket = buckets.setdefault(alias, {})
        for field, val in direct.items():
            if val:
                bucket[field] = val

    for child_name, child_node in children.items():
        if not isinstance(child_node, dict):
            continue
        _walk_yaml_node(child_node, prefix + [child_name], buckets)


def default_db_config_path() -> Path:
    raw = os.environ.get("DB_CONFIG", "config/databases.yaml").strip()
    path = Path(raw)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def load_db_aliases_from_yaml(
    path: Optional[Path] = None,
) -> Tuple[Dict[str, Dict[str, str]], Optional[str]]:
    """YAML 계층 설정 → alias buckets + default alias."""
    cfg_path = path or default_db_config_path()
    if not cfg_path.exists():
        return {}, None

    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML 패키지가 필요합니다. pip install pyyaml"
        ) from e

    with open(cfg_path, encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"DB 설정 YAML 최상위는 mapping 이어야 합니다: {cfg_path}")

    yaml_default = raw.get("default") or raw.get("_default")
    buckets: Dict[str, Dict[str, str]] = {}

    for key, value in raw.items():
        if key in _META_KEYS:
            continue
        if isinstance(value, dict):
            _walk_yaml_node(value, [str(key)], buckets)

    return buckets, (
        _normalize_alias(str(yaml_default)) if yaml_default else None
    )


def load_db_aliases_from_env(
    environ: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, str]]:
    """평면 .env DB_* / ORACLE_* (하위 호환)."""
    env = environ if environ is not None else os.environ
    buckets: Dict[str, Dict[str, str]] = {}

    for key, value in env.items():
        m = _ALIAS_ENV_RE.match(key)
        if not m:
            continue
        segment = m.group(1)
        if "__" in segment:
            alias = ".".join(
                p.lower().replace("-", "_") for p in segment.split("__")
            )
        else:
            alias = segment.lower().replace("-", "_")
        field = m.group(2).lower()
        buckets.setdefault(alias, {})[field] = value.strip()

    legacy_user = env.get("ORACLE_USER", "").strip()
    legacy_password = env.get("ORACLE_PASSWORD", "").strip()
    legacy_dsn = env.get("ORACLE_DSN", "").strip()
    if legacy_user or legacy_password or legacy_dsn:
        main = buckets.setdefault("main", {})
        main.setdefault("user", legacy_user)
        main.setdefault("password", legacy_password)
        main.setdefault("dsn", legacy_dsn)

    return buckets


def load_db_aliases(
    yaml_path: Optional[Path] = None,
    environ: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Dict[str, str]], Optional[str]]:
    """YAML 우선, .env 평면 설정으로 보완·병합."""
    yaml_buckets, yaml_default = load_db_aliases_from_yaml(yaml_path)
    env_buckets = load_db_aliases_from_env(environ)

    merged: Dict[str, Dict[str, str]] = {}
    for alias, fields in {**env_buckets, **yaml_buckets}.items():
        merged.setdefault(alias, {}).update(fields)

    return merged, yaml_default


def alias_ancestors(alias: str) -> List[str]:
    norm = _normalize_alias(alias)
    parts = norm.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def parse_sql_db_alias(sql_text: str, default_alias: str = "main") -> str:
    head = "\n".join(sql_text.splitlines()[:20])
    match = _DB_HEADER_RE.search(head)
    if match:
        return _normalize_alias(match.group(1))
    return _normalize_alias(default_alias)


def resolve_db_credentials(
    alias: str,
    buckets: Dict[str, Dict[str, str]],
) -> DbCredentials:
    key = _normalize_alias(alias)
    merged = {"user": "", "password": "", "dsn": ""}
    matched_any = False
    for anc in alias_ancestors(key):
        fields = buckets.get(anc)
        if not fields:
            continue
        matched_any = True
        for field in merged:
            if fields.get(field):
                merged[field] = fields[field]
    if not matched_any:
        known = ", ".join(sorted(buckets)) or "(없음)"
        raise KeyError(
            f"DB alias '{key}' 가 설정에 없습니다. "
            f"등록된 alias: {known}"
        )
    return DbCredentials(alias=key, **merged)


def default_db_alias(
    yaml_default: Optional[str] = None,
    environ: Optional[Dict[str, str]] = None,
) -> str:
    env = environ if environ is not None else os.environ
    if env.get("DB_DEFAULT_ALIAS", "").strip():
        return _normalize_alias(env["DB_DEFAULT_ALIAS"])
    if yaml_default:
        return _normalize_alias(yaml_default)
    return "main"


class DbRegistry:
    """alias → Oracle 연결 (fetch/collector 공용)."""

    def __init__(
        self,
        alias_buckets: Optional[Dict[str, Dict[str, str]]] = None,
        default_alias: Optional[str] = None,
        yaml_path: Optional[Path] = None,
    ):
        if alias_buckets is None:
            buckets, yaml_def = load_db_aliases(yaml_path=yaml_path)
        else:
            buckets, yaml_def = alias_buckets, None

        self._buckets = buckets
        self._default_alias = _normalize_alias(
            default_alias or default_db_alias(yaml_def),
        )
        self._connections: Dict[str, object] = {}

    @property
    def default_alias(self) -> str:
        return self._default_alias

    @property
    def known_aliases(self) -> List[str]:
        return sorted(self._buckets)

    def get_credentials(self, alias: Optional[str] = None) -> DbCredentials:
        return resolve_db_credentials(alias or self._default_alias, self._buckets)

    def connect(self, alias: Optional[str] = None):
        key = _normalize_alias(alias or self._default_alias)
        if key not in self._connections:
            self._connections[key] = self.get_credentials(key).connect()
        return self._connections[key]

    def resolve_sql_connection(self, sql_text: str):
        alias = parse_sql_db_alias(sql_text, self._default_alias)
        return self.connect(alias), alias

    def close_all(self) -> None:
        for conn in self._connections.values():
            try:
                conn.close()
            except Exception:
                pass
        self._connections.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close_all()


def read_sql_with_alias(sql_path: Path) -> Tuple[str, str]:
    text = sql_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"SQL 파일이 비어 있습니다: {sql_path}")
    alias = parse_sql_db_alias(text)
    return text, alias
