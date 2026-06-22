"""
data/db_registry.py – SQL 쿼리별 DB alias 및 .env 접속 정보 관리

SQL 파일 상단 주석 (점으로 계층):
    -- @db: main
    -- @db: fab.mes
    -- @db: fab.mes.plan

.env – alias는 ``__``(이중 밑줄)로 계층 구분:
    DB_DEFAULT_ALIAS=fab.mes
    DB_FAB__MES_USER=...
    DB_FAB__MES_PASSWORD=...
    DB_FAB__MES_DSN=hostname:1521/ORCL

    # 하위 alias는 상위 필드를 상속 (DSN만 덮어쓰기 등)
    DB_FAB__MES__PLAN_DSN=plan-host:1521/ORCL
    → @db: fab.mes.plan 은 user/password 를 fab.mes 에서 상속

단일 alias (기존): DB_MAIN_USER → ``main``
평면 alias (단일 _): DB_FAB_MES_USER → ``fab_mes``

하위 호환: ORACLE_USER / ORACLE_PASSWORD / ORACLE_DSN → alias ``main``
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_DB_HEADER_RE = re.compile(
    r"^\s*--\s*@db:\s*([A-Za-z][\w-]*(?:\.[A-Za-z][\w-]*)*)\s*$",
    re.MULTILINE,
)
_ALIAS_ENV_RE = re.compile(r"^DB_([A-Z][A-Z0-9_]*)_(USER|PASSWORD|DSN)$")


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
                f"상위 alias 상속 후에도 USER/PASSWORD/DSN 이 부족합니다."
            )
        return oracledb.connect(
            user=self.user, password=self.password, dsn=self.dsn,
        )


def _normalize_alias(name: str) -> str:
    parts = [p.strip().lower().replace("-", "_") for p in name.strip().split(".")]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError(f"DB alias 가 비어 있습니다: {name!r}")
    return ".".join(parts)


def alias_to_env_prefix(alias: str) -> str:
    """``fab.mes`` → ``FAB__MES`` (환경 변수 키 세그먼트)."""
    norm = _normalize_alias(alias)
    return "__".join(part.upper() for part in norm.split("."))


def env_key_to_alias(env_segment: str) -> str:
    """``FAB__MES`` → ``fab.mes``, ``FAB_MES`` → ``fab_mes``, ``MAIN`` → ``main``."""
    if "__" in env_segment:
        parts = [p.lower().replace("-", "_") for p in env_segment.split("__")]
        return ".".join(p for p in parts if p)
    return env_segment.lower().replace("-", "_")


def alias_ancestors(alias: str) -> List[str]:
    """``fab.mes.plan`` → [``fab``, ``fab.mes``, ``fab.mes.plan``]"""
    norm = _normalize_alias(alias)
    parts = norm.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def parse_sql_db_alias(sql_text: str, default_alias: str = "main") -> str:
    """SQL 본문 상단 ``-- @db: <alias>`` 주석 파싱."""
    head = "\n".join(sql_text.splitlines()[:20])
    match = _DB_HEADER_RE.search(head)
    if match:
        return _normalize_alias(match.group(1))
    return _normalize_alias(default_alias)


def load_db_aliases_from_env(
    environ: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, str]]:
    """환경 변수에서 alias별 raw 필드 로드 (상속 전)."""
    env = environ if environ is not None else os.environ
    buckets: Dict[str, Dict[str, str]] = {}

    for key, value in env.items():
        m = _ALIAS_ENV_RE.match(key)
        if not m:
            continue
        alias = env_key_to_alias(m.group(1))
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


def resolve_db_credentials(
    alias: str,
    buckets: Dict[str, Dict[str, str]],
) -> DbCredentials:
    """계층 alias – 상위 → 하위 순으로 필드 병합."""
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
            f"DB alias '{key}' 가 .env에 정의되지 않았습니다. "
            f"등록된 alias: {known}"
        )
    return DbCredentials(alias=key, **merged)


def default_db_alias(environ: Optional[Dict[str, str]] = None) -> str:
    env = environ if environ is not None else os.environ
    return _normalize_alias(env.get("DB_DEFAULT_ALIAS", "main"))


class DbRegistry:
    """alias → Oracle 연결 (fetch/collector 공용)."""

    def __init__(
        self,
        alias_buckets: Optional[Dict[str, Dict[str, str]]] = None,
        default_alias: Optional[str] = None,
    ):
        self._buckets = (
            alias_buckets if alias_buckets is not None else load_db_aliases_from_env()
        )
        self._default_alias = _normalize_alias(
            default_alias or default_db_alias(),
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
    """SQL 파일 읽기 + alias 반환."""
    text = sql_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"SQL 파일이 비어 있습니다: {sql_path}")
    alias = parse_sql_db_alias(text)
    return text, alias
