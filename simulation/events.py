"""
simulation/events.py – DES 시뮬레이션 이벤트 종류 코드

API·UI·event_log와 공통으로 사용하는 대문자 이벤트 코드.

사용자 가시 이벤트 흐름:
  MOVE_OUT → IDLE → JOB_ASSIGNED → [CONV_ASSIGNED] → (가공) → MOVE_OUT …
"""
from typing import Dict, Tuple

# DES 내부 큐 전용 (event_log 미기록)
EVENT_PROCESS_END = "PROCESS_END"
EVENT_IDLE_DECISION = "IDLE_DECISION"
EVENT_CONV_END = "CONV_END"

# event_log에 기록되는 이벤트
EVENT_MOVE_OUT = "MOVE_OUT"
EVENT_IDLE = "IDLE"
EVENT_JOB_ASSIGNED = "JOB_ASSIGNED"
EVENT_CONV_ASSIGNED = "CONV_ASSIGNED"

# 구 event_log / 내부 호환 (신규 실행 시 미기록)
EVENT_TOOL_RELEASE = "TOOL_RELEASE"
EVENT_WIP_INJECT = "WIP_INJECT"
EVENT_TOOL_OCCUPY = "TOOL_OCCUPY"
EVENT_CONV_START = "CONV_START"

ALL_EVENT_KINDS: Tuple[str, ...] = (
    EVENT_MOVE_OUT,
    EVENT_IDLE,
    EVENT_JOB_ASSIGNED,
    EVENT_CONV_ASSIGNED,
)

# 동일 시각 이벤트 처리 순서 (작을수록 먼저)
EVENT_PRIORITY: Dict[str, int] = {
    EVENT_PROCESS_END: 0,
    EVENT_CONV_END:    1,
    EVENT_IDLE_DECISION: 10,
}
