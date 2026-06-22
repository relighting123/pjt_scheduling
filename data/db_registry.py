"""
data/db_registry.py – SQL 쿼리별 DB alias 및 .env 접속 정보 관리

SQL 파일 상단 주석:
    -- @db: main

.env (alias별, 대문자):
    DB_DEFAULT_ALIAS=main
    DB_MAIN_USER=...
    DB_MAIN_PASSWORD=...
    DB_MAIN_DSN=hostname:1521/ORCL

하위 호환: ORACLE_USER / ORACLE_PASSWORD / ORACLE_DSN → alias ``main``
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

_DB_HEADER_RE = re.compile(r"^\s*--\s*@db:\s*([A-Za-z][\w-]*)\s*$", re.MULTILINE)
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
                f"DB_{self.alias.upper()}_USER/PASSWORD/DSN 또는 "
                f"ORACLE_USER/PASSWORD/DSN 을 설정하세요."
            )
        return oracledb.connect(
            user=self.user, password=self.password, dsn=self.dsn,
        )


def _normalize_alias(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def parse_sql_db_alias(sql_text: str, default_alias: str = "main") -> str:
    """SQL 본문 상단 ``-- @db: <alias>`` 주석 파싱."""
    head = "\n".join(sql_text.splitlines()[:20])
    match = _DB_HEADER_RE.search(head)
    if match:
        return _normalize_alias(match.group(1))
    return _normalize_alias(default_alias)


def load_db_aliases_from_env(
    environ: Optional[Dict[str, str]] = None,
) -> Dict[str, DbCredentials]:
    """환경 변수에서 DB alias 목록 로드."""
    env = environ if environ is not None else os.environ
    buckets: Dict[str, Dict[str, str]] = {}

    for key, value in env.items():
        m = _ALIAS_ENV_RE.match(key)
        if not m:
            continue
        alias = _normalize_alias(m.group(1))
        field = m.group(2).lower()
        buckets.setdefault(alias, {})[field] = value.strip()

    # 하위 호환: ORACLE_* → main
    legacy_user = env.get("ORACLE_USER", "").strip()
    legacy_password = env.get("ORACLE_PASSWORD", "").strip()
    legacy_dsn = env.get("ORACLE_DSN", "").strip()
    if legacy_user or legacy_password or legacy_dsn:
        main = buckets.setdefault("main", {})
        main.setdefault("user", legacy_user)
        main.setdefault("password", legacy_password)
        main.setdefault("dsn", legacy_dsn)

    return {
        alias: DbCredentials(
            alias=alias,
            user=fields.get("user", ""),
            password=fields.get("password", ""),
            dsn=fields.get("dsn", ""),
        )
        for alias, fields in buckets.items()
    }


def default_db_alias(environ: Optional[Dict[str, str]] = None) -> str:
    env = environ if environ is not None else os.environ
    return _normalize_alias(env.get("DB_DEFAULT_ALIAS", "main"))


class DbRegistry:
    """alias → Oracle 연결 (fetch/collector 공용)."""

    def __init__(
        self,
        aliases: Optional[Dict[str, DbCredentials]] = None,
        default_alias: Optional[str] = None,
    ):
        self._aliases = aliases if aliases is not None else load_db_aliases_from_env()
        self._default_alias = _normalize_alias(
            default_alias or default_db_alias(),
        )
        self._connections: Dict[str, object] = {}

    @property
    def default_alias(self) -> str:
        return self._default_alias

    def get_credentials(self, alias: Optional[str] = None) -> DbCredentials:
        key = _normalize_alias(alias or self._default_alias)
        creds = self._aliases.get(key)
        if creds is None:
            known = ", ".join(sorted(self._aliases)) or "(없음)"
            raise KeyError(
                f"DB alias '{key}' 가 .env에 정의되지 않았습니다. "
                f"등록된 alias: {known}"
            )
        return creds

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
