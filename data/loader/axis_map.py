"""
data/loader/axis_map.py – 축(공정/제품/설비/모델) 인덱스 고정

문제
----
`preprocess()`는 축 목록을 매 회차 **그 회차 데이터로** 재산출한다
(`sorted({...})`). 그래서 앞선 키 하나가 빠지면 뒤의 키가 전부 한 칸씩 밀린다.
정책의 action 버킷은 `flat = oper_idx*P + prod_idx`이고 관측 텐서도 슬롯 위치로
배열되므로, 이 시프트는 정책이 학습한 슬롯 대응을 통째로 어긋나게 만든다.

실측(2026-08, GENERALIZATION_V6, 샘플 시나리오): 물리적으로 **완전히 동일한**
상태에서 제품 축만 한 칸 회전시켰더니 스케줄 시퀀스 일치율이 16.9%로 무너졌고
전환이 20→22로 늘었다. 재공 1건 변화(66~98% 일치)보다 훨씬 큰 불안정 요인이다.

해결
----
FAC별로 축 순서를 파일에 고정해 두고, 매 회차 그 순서를 재사용한다.
  - 저장된 키는 **저장된 슬롯을 그대로 유지**한다(재번호 부여 없음).
  - 이번 회차에 없는 키도 슬롯을 비우지 않고 자리를 지킨다 — 뒤 키가 밀리지
    않게 하는 것이 목적이므로, 데이터가 없는 키는 관측에서 그냥 0으로 남는다.
  - 새 키는 뒤에 정렬 순으로 덧붙인다.
  - 축 상한(config.env.max_*_count)을 넘길 상황이면, 이번 회차에 등장하지 않은
    키부터 `last_seen`이 오래된 순으로 밀어낸다(LRU). 활성 키는 절대 밀려나지
    않는다.

파일이 없으면 아무것도 하지 않는다 — 기존 `sorted()` 동작 그대로다(하위 호환).

경로
----
기본값은 모델과 같은 디렉터리(`models/<FAC_ID>/axis_map.json`)다. 축 순서는
학습된 정책과 짝이 되는 계약이라 모델 옆에 두는 것이 맞다.
환경변수 `AXIS_MAP_CONFIG`로 다른 경로를 직접 지정할 수 있다
(`data/writer/db_load.py`의 `OUTPUT_DB_ROUTING_CONFIG`와 같은 패턴).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from config import CONFIG, model_dir_for

# axis_map.json 의 최상위 키 → env_data 의 축 목록 키
AXIS_KEYS = ("oper_ids", "prod_keys", "eqp_ids", "eqp_models")

_AXIS_MAP_FILE = "axis_map.json"


def axis_map_path(fac_id: Optional[str] = None) -> Path:
    """축 맵 파일 경로. AXIS_MAP_CONFIG 가 있으면 그 값이 우선."""
    raw = os.environ.get("AXIS_MAP_CONFIG", "").strip()
    if raw:
        return Path(raw)
    return model_dir_for(fac_id or CONFIG.path.fac_id) / _AXIS_MAP_FILE


def load_axis_map(path: Optional[Path] = None,
                  fac_id: Optional[str] = None) -> Optional[dict]:
    """축 맵을 읽는다. 파일이 없으면 None(=고정 비활성)."""
    p = path or axis_map_path(fac_id)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"axis_map 최상위는 object 여야 합니다: {p}")
    return raw


def _axis_cap(axis: str) -> Optional[int]:
    """축별 config 상한. 상한이 없는 축은 None."""
    return {
        "oper_ids": CONFIG.env.max_oper_count,
        "prod_keys": CONFIG.env.max_prod_count,
        "eqp_models": CONFIG.env.max_model_count,
    }.get(axis)


def order_axis(current: List[str], stored: Optional[dict], axis: str) -> List[str]:
    """이번 회차 키 목록 `current` 를 저장된 슬롯 순서에 맞춰 재배열한다.

    stored 형식: {"order": [key, ...], "last_seen": {key: "YYYYMMDDHHMMSS"}}
    (하위 호환으로 리스트만 온 경우도 받는다.)
    """
    if not stored:
        return sorted(current)

    entry = stored.get(axis)
    if isinstance(entry, list):
        order, last_seen = list(entry), {}
    elif isinstance(entry, dict):
        order = list(entry.get("order", []))
        last_seen = dict(entry.get("last_seen", {}))
    else:
        return sorted(current)

    cur = set(current)
    # 저장된 순서 유지 + 이번 회차 신규 키는 뒤에 정렬 순으로
    result = [k for k in order]
    for k in sorted(cur - set(order)):
        result.append(k)

    cap = _axis_cap(axis)
    if cap is not None and len(result) > cap:
        # 넘치면 '이번 회차에 없는' 키부터 오래된 순으로 제거. 활성 키는 보존.
        inactive = [k for k in result if k not in cur]
        inactive.sort(key=lambda k: (last_seen.get(k, ""), k))
        drop = set()
        need = len(result) - cap
        for k in inactive:
            if need <= 0:
                break
            drop.add(k)
            need -= 1
        result = [k for k in result if k not in drop]
    return result


def apply_axis_map(env_axes: Dict[str, List[str]],
                   stored: Optional[dict]) -> Dict[str, List[str]]:
    """AXIS_KEYS 전체에 order_axis 를 적용한 새 축 목록."""
    return {
        axis: order_axis(list(env_axes.get(axis, [])), stored, axis)
        for axis in AXIS_KEYS
    }


def positional_index_map(items: List[str]) -> Dict[str, int]:
    """리스트 위치를 그대로 인덱스로 쓴다.

    `utils.helpers.build_index_map()` 은 내부에서 다시 `sorted()` 하므로 고정
    순서를 무너뜨린다. 축 고정 경로에서는 반드시 이 함수를 써야
    `axis_list[i] == key` ⇔ `idx_map[key] == i` 불변식이 유지된다.
    """
    return {v: i for i, v in enumerate(items)}


def build_axis_map(env_data: dict, timestamp: str,
                   previous: Optional[dict] = None) -> dict:
    """현재 env_data 의 축 순서를 저장 형식으로 직렬화."""
    out: dict = {}
    for axis in AXIS_KEYS:
        items = list(env_data.get(axis, []))
        prev_seen = {}
        if previous and isinstance(previous.get(axis), dict):
            prev_seen = dict(previous[axis].get("last_seen", {}))
        prev_seen.update({k: timestamp for k in items})
        out[axis] = {"order": items, "last_seen": prev_seen}
    return out


def save_axis_map(env_data: dict, *, timestamp: str,
                  path: Optional[Path] = None,
                  fac_id: Optional[str] = None) -> Path:
    """축 맵을 파일에 쓴다(기존 파일의 last_seen 은 병합)."""
    p = path or axis_map_path(fac_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    previous = load_axis_map(p) if p.exists() else None
    payload = build_axis_map(env_data, timestamp, previous)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return p
