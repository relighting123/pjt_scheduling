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

SQL: ``-- @db: Prd`` / ``@db:WT_RTS`` / ``-- @db: Dev.Mes`` (``@db`` 대소문자 무관)

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
    r"^\s*(?:--\s*)?@db\s*:\s*"
    r"([A-Za-z][\w.-]*(?:\.[A-Za-z][\w.-]*)*)"
    r"(?:\s*--.*)?\s*$",
    re.IGNORECASE | re.MULTILINE,
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


def _ensure_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env")
    except ImportError:
        pass


def _read_dotenv_file() -> Dict[str, str]:
    """`.env` 파일 값 (프로세스 환경과 비교용, 비밀은 마스킹하지 않음)."""
    path = BASE_DIR / ".env"
    if not path.exists():
        return {}
    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def default_alias_source(
    yaml_default: Optional[str] = None,
    environ: Optional[Dict[str, str]] = None,
) -> str:
    env = environ if environ is not None else os.environ
    raw = env.get("DB_DEFAULT_ALIAS", "").strip()
    if raw:
        norm = _normalize_alias(raw)
        return f".env DB_DEFAULT_ALIAS={raw} (내부 키: {norm})"
    if yaml_default:
        return f"yaml default: {yaml_default} (내부 키: {yaml_default})"
    return "fallback: main (DB_DEFAULT_ALIAS·yaml default 미설정)"


def scan_sql_db_aliases(sql_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """runtime SQL 폴더의 @db 헤더 스캔."""
    from config import SQL_DIR

    target = sql_dir or SQL_DIR
    rows: List[Dict[str, Any]] = []
    if not target.exists():
        return rows
    for path in sorted(target.glob("*.sql")):
        head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:20])
        match = _DB_HEADER_RE.search(head)
        if match:
            raw = match.group(1)
            rows.append({
                "file": path.name,
                "raw_alias": raw,
                "alias": _normalize_alias(raw),
                "uses_default": False,
            })
        else:
            rows.append({
                "file": path.name,
                "raw_alias": None,
                "alias": None,
                "uses_default": True,
            })
    return rows


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
    _ensure_dotenv()
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

    # default alias 에 루트 수준 user/password/dsn (흔한 설정 실수 허용)
    root_fields: Dict[str, str] = {}
    for key, value in raw.items():
        if key in _META_KEYS or isinstance(value, dict):
            continue
        field = _normalize_field(str(key))
        if field and value is not None and not isinstance(value, (dict, list)):
            root_fields[field] = str(value).strip()
    if root_fields and yaml_default:
        buckets.setdefault(_normalize_alias(str(yaml_default)), {}).update(root_fields)

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
    _ensure_dotenv()
    yaml_buckets, yaml_default = load_db_aliases_from_yaml(yaml_path)
    env_buckets = load_db_aliases_from_env(environ)

    merged: Dict[str, Dict[str, str]] = {}
    for alias, fields in {**env_buckets, **yaml_buckets}.items():
        merged.setdefault(alias, {}).update(fields)

    return merged, yaml_default


def diagnose_db_config(
    yaml_path: Optional[Path] = None,
    environ: Optional[Dict[str, str]] = None,
) -> dict:
    """DB 설정 진단 – collector/fetch 오류 시 원인 확인용."""
    _ensure_dotenv()
    cfg_path = yaml_path or default_db_config_path()
    example_path = BASE_DIR / "config" / "databases.yaml.example"
    env = environ if environ is not None else os.environ
    buckets, yaml_default = load_db_aliases(yaml_path=yaml_path, environ=environ)
    default_alias, default_warn = resolve_default_db_alias(
        buckets, yaml_default, environ,
    )
    alias_source = default_alias_source(yaml_default, environ)
    dotenv_values = _read_dotenv_file()
    sql_aliases = scan_sql_db_aliases()

    issues: List[str] = []
    notes: List[str] = []
    if default_warn:
        notes.append(default_warn)
    if not cfg_path.exists():
        issues.append(f"YAML 파일 없음: {cfg_path}")
        if example_path.exists():
            try:
                rel_cfg = cfg_path.relative_to(BASE_DIR)
                rel_ex = example_path.relative_to(BASE_DIR)
            except ValueError:
                rel_cfg = cfg_path
                rel_ex = example_path
            issues.append(f"실행: cp {rel_ex} {rel_cfg}")
    elif not buckets:
        issues.append(
            f"YAML 은 있으나 DB alias 가 비어 있습니다: {cfg_path} "
            "(Prd: 아래에 user/password/dsn 이 Prd 블록 안에 있어야 합니다)"
        )

    if default_alias not in buckets and not any(
        anc in buckets for anc in alias_ancestors(default_alias)
    ):
        issues.append(
            f"default alias '{default_alias}' 가 buckets 에 없습니다. "
            f"등록된 alias: {', '.join(sorted(buckets)) or '(없음)'}"
        )

    for var in ("DB_CONFIG", "DB_DEFAULT_ALIAS"):
        file_val = dotenv_values.get(var, "").strip()
        proc_val = env.get(var, "").strip()
        if file_val and proc_val and file_val != proc_val:
            issues.append(
                f"환경 변수 {var} 불일치: 셸/서비스={proc_val!r}, .env={file_val!r} "
                "(셸 값이 우선 적용됩니다. systemd/cron 이면 WorkingDirectory·EnvironmentFile 확인)"
            )

    from config import SQL_DIR

    if not SQL_DIR.exists():
        issues.append(
            f"SQL 폴더 없음: {SQL_DIR} "
            "(실행: mkdir -p data/sql && cp data/sql.example/*.sql data/sql/)"
        )
    elif not sql_aliases:
        issues.append(f"SQL 폴더에 *.sql 없음: {SQL_DIR}")
    else:
        for row in sql_aliases:
            alias = row.get("alias")
            if alias is None:
                issues.append(
                    f"{row['file']}: -- @db: 헤더 없음 → default alias '{default_alias}' 사용"
                )
                continue
            if not any(anc in buckets for anc in alias_ancestors(alias)):
                issues.append(
                    f"{row['file']}: -- @db: {row['raw_alias']} "
                    f"(내부 키: {alias}) 가 databases.yaml 에 없습니다"
                )

    sql_alias_set = {row["alias"] for row in sql_aliases if row.get("alias")}
    if sql_alias_set == {default_alias} and default_alias:
        notes.append(
            f"모든 SQL 이 -- @db: 로 '{default_alias}' 를 직접 지정합니다. "
            "fetch/collect 시 default 가 아니라 SQL 헤더 alias 가 사용됩니다."
        )
    if default_alias == "prd":
        notes.append(
            "Prd → prd 는 정상입니다. alias 는 대소문자 구분 없이 소문자로 통일됩니다."
        )

    return {
        "config_path": str(cfg_path),
        "config_exists": cfg_path.exists(),
        "example_path": str(example_path),
        "default_alias": default_alias,
        "default_alias_source": alias_source,
        "yaml_default": yaml_default,
        "sql_dir": str(SQL_DIR),
        "sql_aliases": sql_aliases,
        "known_aliases": sorted(buckets),
        "buckets": {
            alias: {k: ("***" if k == "password" and v else v) for k, v in fields.items()}
            for alias, fields in buckets.items()
        },
        "issues": issues,
        "notes": notes,
        "ok": not issues,
    }


def print_db_config_report(report: dict) -> None:
    """diagnose_db_config() 결과를 사람이 읽기 좋게 출력."""
    print(f"config: {report['config_path']} ({'OK' if report['config_exists'] else 'MISSING'})")
    print(f"default alias: {report['default_alias']}")
    print(f"default source: {report.get('default_alias_source', '')}")
    print(f"known aliases: {', '.join(report['known_aliases']) or '(없음)'}")
    for alias, fields in report["buckets"].items():
        print(f"  [{alias}] user={fields.get('user', '')} dsn={fields.get('dsn', '')}")
    if report.get("sql_aliases"):
        print(f"sql dir: {report['sql_dir']}")
        for row in report["sql_aliases"]:
            if row.get("uses_default"):
                print(f"  {row['file']}: (헤더 없음 → default)")
            else:
                print(
                    f"  {row['file']}: -- @db: {row['raw_alias']} "
                    f"(내부 키: {row['alias']})"
                )
    if report.get("notes"):
        print("notes:")
        for note in report["notes"]:
            print(f"  - {note}")
    if report["issues"]:
        print("issues:")
        for issue in report["issues"]:
            print(f"  - {issue}")
    else:
        print("status: OK")


def format_db_config_error(alias: str, buckets: Dict[str, Dict[str, str]]) -> str:
    cfg_path = default_db_config_path()
    diag = diagnose_db_config()
    known = ", ".join(sorted(buckets)) or "(없음)"
    lines = [
        f"DB alias '{_normalize_alias(alias)}' 가 설정에 없습니다.",
        f"등록된 alias: {known}",
        f"설정 파일: {cfg_path} ({'있음' if cfg_path.exists() else '없음'})",
        f"default alias: {diag['default_alias']}",
    ]
    if diag["issues"]:
        lines.append("확인 사항:")
        lines.extend(f"  - {item}" for item in diag["issues"])
    return "\n".join(lines)


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
        raise KeyError(format_db_config_error(key, buckets))
    return DbCredentials(alias=key, **merged)


def _alias_resolvable(alias: str, buckets: Dict[str, Dict[str, str]]) -> bool:
    norm = _normalize_alias(alias)
    return norm in buckets or any(
        anc in buckets for anc in alias_ancestors(norm)
    )


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


def resolve_default_db_alias(
    buckets: Dict[str, Dict[str, str]],
    yaml_default: Optional[str] = None,
    environ: Optional[Dict[str, str]] = None,
) -> Tuple[str, Optional[str]]:
    """default alias 결정. .env 값이 buckets 에 없으면 yaml default 로 대체."""
    env = environ if environ is not None else os.environ
    preferred = default_db_alias(yaml_default, environ)
    if _alias_resolvable(preferred, buckets):
        return preferred, None

    env_raw = env.get("DB_DEFAULT_ALIAS", "").strip()
    if env_raw and yaml_default:
        yaml_norm = _normalize_alias(yaml_default)
        if _alias_resolvable(yaml_norm, buckets):
            return yaml_norm, (
                f".env DB_DEFAULT_ALIAS={env_raw!r} 가 databases.yaml 에 없습니다. "
                f"yaml default '{yaml_norm}' 를 사용합니다."
            )

    return preferred, None


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
        if default_alias is not None:
            self._default_alias = _normalize_alias(default_alias)
            self._default_warn = None
        else:
            resolved, warn = resolve_default_db_alias(buckets, yaml_def)
            self._default_alias = resolved
            self._default_warn = warn
        if self._default_alias not in buckets and not any(
            anc in buckets for anc in alias_ancestors(self._default_alias)
        ):
            diag = diagnose_db_config(yaml_path=yaml_path)
            if diag["issues"]:
                raise KeyError(
                    format_db_config_error(self._default_alias, buckets)
                )
        self._connections: Dict[str, object] = {}

    @property
    def default_alias(self) -> str:
        return self._default_alias

    @property
    def default_warn(self) -> Optional[str]:
        return self._default_warn

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


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="DB alias 설정 진단")
    parser.add_argument("--json", action="store_true", help="JSON 출력")
    args = parser.parse_args(argv)
    report = diagnose_db_config()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_db_config_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
