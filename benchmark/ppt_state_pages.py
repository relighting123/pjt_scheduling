"""PPT State 항목별 상세 슬라이드 메타."""

STATE_TERM_PAGES = [
    {
        "key": "global",
        "title": "전역 상태 (Global)",
        "obs_slice": "obs[0:6]",
        "plain": "라인 전체 상황 — 시간·재공·계획·전환·공구를 6개 숫자로 요약합니다.",
        "why": "정책이 \"지금 몇 시쯤인지, 얼마나 급한지\"를 전역적으로 파악하게 합니다.",
        "trace_step": 1,
        "items": [
            {
                "idx": "obs[0]",
                "name": "time_norm",
                "formula": "min(t / sim_end, 1)",
                "meaning": "시뮬 경과 비율 (0=시작, 1=종료)",
                "trace_path": ("obs_global", "time_norm"),
            },
            {
                "idx": "obs[1]",
                "name": "takt_margin",
                "formula": "(soft_cutoff − t) / soft_cutoff",
                "meaning": "남은 horizon 여유 (1→여유 많음)",
                "trace_path": ("obs_global", "takt_margin"),
            },
            {
                "idx": "obs[2]",
                "name": "remaining_lots",
                "formula": "|lot_pool| / 초기 LOT 수",
                "meaning": "미배정 재공 비율 (↓=소진)",
                "trace_path": ("obs_global", "remaining_lots"),
            },
            {
                "idx": "obs[3]",
                "name": "plan_progress",
                "formula": "Σ완료 / Σ계획",
                "meaning": "전체 계획 달성률",
                "trace_path": ("obs_global", "plan_progress"),
            },
            {
                "idx": "obs[4]",
                "name": "conv_idle_ratio",
                "formula": "전환대기 idle 설비 / 전체",
                "meaning": "셋업 바꾸느라 대기 중인 설비 비율",
                "trace_path": ("obs_global", "conv_idle_ratio"),
            },
            {
                "idx": "obs[5]",
                "name": "tool_util",
                "formula": "공구 동시가공 사용률",
                "meaning": "tool cap 포화 정도",
                "trace_path": ("obs_global", "tool_util"),
            },
        ],
    },
    {
        "key": "bucket",
        "title": "버킷 특징 (Bucket)",
        "obs_slice": "obs[6 : 6+O×P×K×16]",
        "plain": "(OPER×PPK×모델) 격자마다 16채널 — \"이 버킷을 지금 잡으면 유효·시급·중복인가\"를 알려줍니다.",
        "why": "보상(페이싱·중복커버) 판단의 근거가 되는 핵심 특징입니다.",
        "trace_step": 1,
        "items": [
            {
                "idx": "ch0", "name": "valid",
                "formula": "1[해당 oper·model 조합에 배정 가능한 설비 존재]",
                "meaning": "현재 EQP·재공·공구 제약과 무관하게 이 (공정×모델)이 유효한가",
                "trace_path": None,
            },
            {
                "idx": "ch1-2", "name": "wip_비중",
                "formula": "wip_q/ΣWIP_전체 ,  wip_q/ΣWIP_동일PPK",
                "meaning": "재공이 이 버킷에 얼마나 몰려있나 (전체 대비 / 같은 제품 대비)",
                "trace_path": None,
            },
            {
                "idx": "ch3", "name": "min_end_time",
                "formula": "min(free_at, 해당 oper·model 설비군) / sim_end",
                "meaning": "이 모델 설비가 가장 빨리 비는 시각(정규화)",
                "trace_path": None,
            },
            {
                "idx": "ch4", "name": "throughput_ratio",
                "formula": "wip_q / max(간트 최대종료시각, min_end_time + st_per_wafer·wf_unit)",
                "meaning": "대기 재공 대비 이 모델의 처리완료 예상 밀도 — 높을수록 적체",
                "trace_path": None,
            },
            {
                "idx": "ch6-7", "name": "prev/post_takt",
                "formula": (
                    "eff_takt(ppk, 이전/다음공정) / max_takt  ;  "
                    "eff_takt = max(수요takt, capa takt)·wf_unit"
                ),
                "meaning": "상류(이전공정)·하류(다음공정) 기준 요구 속도 — 수요 vs 설비capa 중 더 빠듯한 쪽",
                "trace_path": None,
            },
            {
                "idx": "ch12", "name": "needs_conversion",
                "formula": "1[현재 EQP에서 이 LOT_CD/TEMP로 전환 필요]",
                "meaning": "이 버킷 잡으면 셋업 변경?",
                "trace_path": None,
            },
            {
                "idx": "ch13", "name": "tool_can_assign",
                "formula": "1[공구 교체 불필요 또는 공구 슬롯 여유]",
                "meaning": "tool cap 통과 여부",
                "trace_path": None,
            },
            {
                "idx": "ch14", "name": "achievable_ratio",
                "formula": "min(min(계획량, 완료+상류WIP) / 계획량, 1)",
                "meaning": "재공 한도 내 달성 가능한 상한 비율",
                "trace_path": None,
            },
            {
                "idx": "ch15", "name": "projected_cover_ratio",
                "formula": "min(cover / need, 2) / 2  ;  need=달성상한−완료, cover=타EQP 하루 투영생산",
                "meaning": "중복 커버 신호 — 클수록 다른 설비가 이미 커버 중(회피 유도)",
                "trace_path": None,
            },
        ],
    },
    {
        "key": "eqp_local",
        "title": "현재 설비 (EQP local)",
        "obs_slice": "obs[…+0:…+2]",
        "plain": "지금 결정하는 그 설비에만 해당하는 전환 정보 2개.",
        "why": "회피가능 전환 패널티 학습에 직접 쓰입니다.",
        "trace_step": 1,
        "items": [
            {
                "idx": "eqp[0]",
                "name": "needs_conversion",
                "formula": "1[feasible 중 전환 필요 버킷 존재]",
                "meaning": "이 설비가 feasible한 선택 중 셋업 변경이 필요한 게 있나",
                "trace_path": ("obs_eqp_local", "needs_conversion"),
            },
            {
                "idx": "eqp[1]",
                "name": "avoidable_frac",
                "formula": "max α over feasible 전환 버킷",
                "meaning": "전환 시 회피가능 정도 (0~1, 보상 α와 동일 개념)",
                "trace_path": ("obs_eqp_local", "avoidable_frac"),
            },
        ],
    },
    {
        "key": "context",
        "title": "직전 맥락 (Context)",
        "obs_slice": "obs[…+2:…+6]",
        "plain": "직전 스텝에서 누가·무엇을·어느 설비에 배정했는지 정규화 인덱스 4개.",
        "why": "동일 셋업 연속·전환 여부 판단의 단서.",
        "trace_step": 4,
        "items": [
            {"idx": "ctx[0]", "name": "last_ppk", "formula": "encode(PPK)", "meaning": "직전 배정 제품 인덱스", "trace_path": ("obs_context", "last_ppk")},
            {"idx": "ctx[1]", "name": "last_oper", "formula": "encode(OPER)", "meaning": "직전 배정 공정 인덱스", "trace_path": ("obs_context", "last_oper")},
            {"idx": "ctx[2]", "name": "last_eqp", "formula": "encode(EQP)", "meaning": "직전 배정 설비 인덱스", "trace_path": ("obs_context", "last_eqp")},
            {"idx": "ctx[3]", "name": "last_lot_cd", "formula": "encode(LOT_CD)", "meaning": "직전 LOT_CD (셋업 그룹)", "trace_path": ("obs_context", "last_lot_cd")},
        ],
    },
]


def trace_state_value(steps: list, step_no: int, trace_path) -> str:
    if not trace_path:
        return "—"
    for st in steps:
        if st.get("step") != step_no:
            continue
        obj = st.get("state") or {}
        cur = obj
        for key in trace_path:
            cur = (cur or {}).get(key)
        if cur is None:
            return "—"
        return str(cur)
    return "—"


def trace_state_detail(trace: dict, step_no: int, trace_path) -> str:
    """실측 산출식 값 — 산식에 실제 트레이스 수치를 대입한 상세 값."""
    if not trace_path:
        return "—"
    steps = trace.get("steps", [])
    st = next((s for s in steps if s.get("step") == step_no), None)
    if st is None:
        return "—"
    obj = st.get("state") or {}
    cur = obj
    for key in trace_path:
        cur = (cur or {}).get(key)
    if cur is None:
        return "—"
    value = cur
    group, name = trace_path

    if group == "obs_global" and name == "time_norm":
        t = obj.get("time_min", st.get("t"))
        sim_end = trace.get("sim")
        if t is not None and sim_end:
            return f"min({t}/{sim_end}, 1) = {value}"

    if group == "obs_global" and name == "plan_progress":
        produced = obj.get("produced")
        total_plan = trace.get("total_plan")
        if produced is not None and total_plan:
            return f"min({produced}/{total_plan}, 1) = {value}"

    if group == "obs_context" and name in ("last_ppk", "last_oper", "last_eqp"):
        prev = next((s for s in steps if s.get("step") == step_no - 1), None)
        key_map = {"last_ppk": "ppk", "last_oper": "oper", "last_eqp": "eqp"}
        label = prev.get(key_map[name]) if prev else None
        if label:
            return f"직전 배정 {label} → 정규화 {value}"

    return f"= {value}"
