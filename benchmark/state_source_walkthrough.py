"""State 항목별 '소스 그대로 + 대입 계산' 워크스루.

여기 있는 숫자는 실제 SYM_3x3 트레이스가 아니라, State 산식이 어떻게 계산되는지
한 줄씩 눈으로 따라갈 수 있도록 만든 가상 예시 데이터셋(MINI-A)이다.
소스 코드는 simulation/simulator.py 실제 라인을 그대로 인용한다.

"group" 필드는 두 단계로 구성한다: 의미 기준 카테고리(①~⑤) + 이 항목이 실제로
기준으로 삼는 버킷/설비(항목별 정의는 생략하지 않고 그대로 유지).
"""

SOURCE_FILE = "simulation/simulator.py"

MINI_A_DATASET = [
    ("설비", "EQP001: OPER001·OPER002 가능, 세팅 A, busy (free_at=320)"),
    ("", "EQP002: OPER001·OPER002 가능, 세팅 B, idle (free_at=120) — 지금 결정 대상(current_eqp)"),
    ("", "EQP003: OPER002만 가능, 세팅 A, busy (free_at=400)"),
    ("제품·공정", "PPK001: OPER001→OPER002 (세팅 LOT_CD=A, TEMP=없음), PPK002: OPER001만 사용(계획 없음)"),
    ("계획/완료", "(PPK001,OPER002) 목표 60장, 완료 10장  ·  가공시간 OPER001=2분/장, OPER002=5분/장"),
    ("WIP(대기재공)", "(PPK001,OPER001)=140장, (PPK001,OPER002)=15장, (PPK002,OPER001)=20장 → 합계 175장"),
    ("배정 설비수", "OPER001=2대(EQP001,EQP002), OPER002=3대(EQP001,EQP002,EQP003) — 모두 model M1"),
    ("시각", "t=120분, sim_end=soft_cutoff=480분 → T_avail=max(480-120,1)=360분, wf_unit=1"),
    ("직전 배정", "EQP003이 PPK001 · OPER002 · 세팅A를 배정받음 (Context 계산 기준)"),
    ("인덱스맵", "PPK{PPK002:0,PPK001:1}(P=10) · OPER{OPER001:0,OPER002:1}(O=3) · "
                 "EQP{EQP001:0,EQP002:1,EQP003:2} · LOT_CD{B:0,A:1}(n_lc=2)"),
    ("기타", "lot_pool=25/초기40, tool_tracker.utilization()=0.4(가정) — 이 둘은 별도 헬퍼/클래스 내부라 예시값으로 표기"),
]

MINI_A_NOTE = (
    "위 값은 실제 SYM_3x3 트레이스가 아니라 산식 계산 과정을 보여주기 위한 가상 예시 데이터셋(MINI-A)입니다. "
    "이후 슬라이드의 모든 대입 계산은 이 데이터셋을 그대로 사용합니다."
)

# 각 항목: group(의미 카테고리+기준) / title(표시 헤더) / diagram(그림 키) / diagram_desc(그림 설명)
# lines(실제 소스) / calc(대입 계산) / result / note
STATE_WALKTHROUGH = [
    {
        "group": "전역 — ① 시간 진행 (Time Progress)",
        "title": "obs[0] time_norm  ·  obs[1] takt_margin",
        "diagram": "obs01",
        "diagram_desc": (
            "0~sim_end(480분) 축을 t=120 지점에서 둘로 나눈다. 왼쪽 구간(0→t) 길이의 비율이 "
            "time_norm=0.25, 오른쪽 구간(t→soft_cutoff) 길이의 비율이 takt_margin=0.75다. "
            "이 예시처럼 sim_end와 soft_cutoff이 같은 값(480)이면 두 지표가 마치 'takt_margin=1-time_norm'인 "
            "것처럼 보이지만, 실제로는 서로 다른 변수(sim_end vs soft_cutoff)라 분모가 달라지면 이 관계가 깨진다."
        ),
        "lines": [
            "2357  group_global[0] = min(self.current_time / max(self.sim_end, 1), 1.0)",
            "2358  group_global[1] = min(",
            "2359      max(self.soft_cutoff - self.current_time, 0) / max(self.soft_cutoff, 1), 1.0,",
            "2360  )",
        ],
        "calc": [
            "current_time=120, sim_end=480",
            "time_norm = min(120 / 480, 1) = 0.25",
            "",
            "soft_cutoff=480",
            "takt_margin = min( max(480-120,0) / 480, 1 ) = min(360/480, 1) = 0.75",
        ],
        "result": "time_norm = 0.25   /   takt_margin = 0.75",
        "note": "sim_end과 soft_cutoff을 같은 값(480)으로 두면 takt_margin ≈ 1-time_norm 처럼 보이지만, "
                "두 값이 실제로 다르면(예: sim_end=600) 이 관계는 깨진다 — 서로 다른 변수임에 유의.",
    },
    {
        "group": "전역 — ② 재공·계획 진행 (WIP & Plan Progress)",
        "title": "obs[2] remaining_lots  ·  obs[3] plan_progress",
        "diagram": "obs23",
        "diagram_desc": (
            "왼쪽은 LOT 풀(25/40 남음 → remaining_lots=0.625), 오른쪽은 계획 달성(10/60 완료 → "
            "plan_progress=0.1667)을 채움 비율 막대로 나란히 비교한다. 둘 다 '진행률'처럼 보이지만 분모가 "
            "전혀 다르다 — 왼쪽 분모는 '시뮬레이션 시작 시점의 초기 LOT 개수(40)', 오른쪽 분모는 "
            "'이 계획(plan)의 목표 수량(60)'이라 서로 독립적으로 움직인다."
        ),
        "lines": [
            "2331  initial_lot_count = max(len(data[\"lots\"]), 1)",
            "2332  total_plan = max(",
            "2333      sum(p[\"d0_plan_qty\"] for p in data.get(\"plan\", []) if p.get(\"d0_plan_qty\", 0) > 0),",
            "2334      0,",
            "2335  )",
            "  ...",
            "2361  group_global[2] = min(len(self.lot_pool) / initial_lot_count, 1.0)",
            "2362  produced = sum(self.stats[\"completed_qty\"].values())",
            "2363  if total_plan > 0:",
            "2364      group_global[3] = min(produced / total_plan, 1.0)",
            "2365  else:",
            "2366      group_global[3] = min(produced / max(self._initial_wip_total, 1), 1.0)",
        ],
        "calc": [
            "len(lot_pool)=25, initial_lot_count=40",
            "remaining_lots = min(25/40, 1) = 0.625",
            "",
            "produced = Σcompleted_qty = 10  (완료 10장)",
            "total_plan = 60  (d0_plan_qty>0인 계획 합)",
            "plan_progress = min(10/60, 1) = 0.1667",
        ],
        "result": "remaining_lots = 0.625   /   plan_progress = 0.1667",
        "note": "produced는 '이 (ppk,op) 하나'가 아니라 전체 completed_qty 합 — 계획이 하나뿐인 MINI-A에선 결과가 같다.",
    },
    {
        "group": "전역 — ③ 설비·공구 가동 현황 (Equipment & Tool Utilization)",
        "title": "obs[4] conv_idle_ratio  ·  obs[5] tool_util",
        "diagram": "obs45",
        "diagram_desc": (
            "3대 설비의 t=120분 시점 busy/idle 상태를 간트 띠로 펼쳐, conv_idle_ratio(전환대기 idle 설비 "
            "비율)와 tool_util(공구 동시가공 사용률)이 어느 순간을 가리키는지 본다. EQP002는 지금 막(rem=0) "
            "idle이 되었기 때문에 '전환 buffer 중'으로 카운트되지 않고 conv_eqps 집계에서 빠진다 — "
            "코드의 rem>0 조건이 이 배제를 만드는 지점이다."
        ),
        "lines": [
            "2367  conv_eqps = 0",
            "2368  for eqp_id, eqp in self.eqps.items():",
            "2369      rem = max(eqp.free_at - self.current_time, 0)",
            "2370      if rem > 0 and eqp.status == \"idle\" and eqp.prev_lot_cd is not None:",
            "2371          conv_eqps += 1",
            "2372  group_global[4] = conv_eqps / max(len(self.eqps), 1)",
            "2373  group_global[5] = min(self._tool_tracker.utilization(), 1.0)",
        ],
        "calc": [
            "EQP001: busy → status!=idle → 제외",
            "EQP002: idle, free_at=120=t → rem=max(120-120,0)=0 → rem>0 거짓 → 제외",
            "EQP003: busy → status!=idle → 제외",
            "conv_eqps = 0",
            "conv_idle_ratio = 0 / 3 = 0.0",
            "",
            "tool_tracker.utilization() = 0.4  (가정값 — 별도 클래스 내부 계산)",
            "tool_util = min(0.4, 1) = 0.4",
        ],
        "result": "conv_idle_ratio = 0.0   /   tool_util = 0.4",
        "note": "rem>0 조건 때문에 '방금 막 idle된' 설비는 카운트되지 않는다 — 진짜 전환 buffer 중인 설비만 잡음.",
    },
    {
        "group": "버킷 — ① 적합성·가용성 (Feasibility)  ·  (PPK001, OPER002) 기준",
        "title": "ch0 valid",
        "diagram": "ch0",
        "diagram_desc": (
            "OPER001/OPER002 각각을 처리할 수 있는 '설비 모델(model)'이 있는지를 연결선으로 본다. "
            "여기서 '설비'는 개별 EQP가 아니라 모델(model) 그룹 단위다 — MINI-A는 EQP001·EQP002·EQP003을 "
            "모두 model M1 하나로 묶어 단순화했지만, 실제 데이터에는 model이 여러 개 있을 수 있다. "
            "그리고 valid는 PPK와 무관하다: min_end_by_om 사전이 (OPER,model) 두 값만으로 키를 만들기 "
            "때문에, PPK001 버킷이든 다른 PPK의 OPER002 버킷이든 같은 model이 있다면 valid는 항상 같다."
        ),
        "lines": [
            "2261  for mi in range(min(K, len(eqp_models))):",
            "2262      model = eqp_models[mi]",
            "2263      min_end_val = min_end_by_om.get((op, model))",
            "2264      if min_end_val is None:",
            "2265          continue",
            "2267      valid_mis.append(mi)",
            "  ...",
            "2271  if not valid_mis:",
            "2272      continue",
            "  ...",
            "2279  feats[oi, pi, vmis, 0] = 1.0",
        ],
        "calc": [
            "(OPER002, model M1)을 처리 가능한 설비: EQP001·EQP002·EQP003 → min_end_by_om에 값 존재",
            "valid_mis = [0]  (모델 0번 M1이 유효)",
            "→ valid_mis가 비어있지 않으므로 continue 없이 진행 → ch0 = 1.0",
        ],
        "result": "valid (ch0) = 1.0",
        "note": "질문: '특정 장비가 처리 가능하면 1인가?' → 정확히는 '개별 장비'가 아니라 '모델(model) 단위'다. "
                "이 (OPER,model) 조합을 처리할 수 있는 model이 하나도 없으면 valid_mis가 비어 continue → "
                "그 모델 축은 0으로 남는다. PPK는 이 판정에 관여하지 않는다.",
    },
    {
        "group": "버킷 — ② 재공·부하 (WIP & Load)  ·  (PPK001, OPER002) 기준",
        "title": "ch1 wip_ratio_total  ·  ch2 wip_ratio_ppk",
        "diagram": "ch1_2",
        "diagram_desc": (
            "175장짜리 팹 전체 WIP 막대 안에서 이 버킷(15장)의 위치와, PPK001만 따로 뗀 155장 막대 안에서의 "
            "위치를 나란히 비교한다. 분자(wip_q=15)는 같지만 분모가 다르다 — ch1은 '팹 전체 대비', ch2는 "
            "'같은 제품(PPK001) 안에서의 상대적 비중'을 본다. 예를 들어 다른 PPK의 WIP가 갑자기 늘어나면 "
            "ch1만 낮아지고 ch2는 그대로다 — 두 채널이 서로 다른 압력 신호를 정책에 동시에 준다."
        ),
        "lines": [
            "2159  for key, q in self.get_wip_waiting().items():",
            "2160      ppk_w, op_w = key.split(\"|\", 1)",
            "2161      wip_po[(ppk_w, op_w)] = wip_po.get((ppk_w, op_w), 0.0) + q",
            "2162      ppk_wip[ppk_w] = ppk_wip.get(ppk_w, 0.0) + q",
            "2163      total_wip += q",
            "2164  total_wip = max(total_wip, 1.0)",
            "  ...",
            "2221  wip_q = wip_po.get((ppk, op), 0.0)",
            "2222  ppk_total = max(ppk_wip.get(ppk, 0.0), 1.0)",
            "  ...",
            "2280  feats[oi, pi, vmis, 1] = wip_q / total_wip",
            "2281  feats[oi, pi, vmis, 2] = wip_q / ppk_total",
        ],
        "calc": [
            "wip_q = wip_po[(PPK001,OPER002)] = 15",
            "total_wip = 140(PPK001·OPER001) + 15(PPK001·OPER002) + 20(PPK002·OPER001) = 175",
            "ppk_total = ppk_wip[PPK001] = 140+15 = 155",
            "",
            "wip_ratio_total = 15 / 175 = 0.0857",
            "wip_ratio_ppk   = 15 / 155 = 0.0968",
        ],
        "result": "ch1 = 0.0857   /   ch2 = 0.0968",
        "note": "분모가 다르다: ch1은 '팹 전체 WIP', ch2는 '같은 PPK001 안에서의 WIP' — 같은 wip_q라도 두 관점을 동시에 준다.",
    },
    {
        "group": "버킷 — ① 적합성·가용성 (Feasibility)  ·  (PPK001, OPER002) 기준",
        "title": "ch3 min_end_time",
        "diagram": "ch3",
        "diagram_desc": (
            "min_end_by_om은 (OPER, model) 두 값만으로 키를 만드는 사전이라 PPK와는 무관하다 — 그래서 "
            "이 막대그래프가 실제로 비교하는 대상은 'PPK001의 OPER002'가 아니라 'OPER002를 처리하는 "
            "model M1 설비들(EQP001·EQP002·EQP003)의 free_at'이다. 셋 중 최솟값(EQP002=120)이 "
            "min_end_time의 분자가 되며, 이 값은 OPER002를 쓰는 다른 어떤 PPK 버킷에도 그대로 재사용된다."
        ),
        "lines": [
            "2139  min_end_by_om: Dict[tuple, int] = {",
            "2140      key: min(self.eqps[e].free_at for e in eqp_list)",
            "2141      for key, eqp_list in self._eqps_by_om.items()",
            "2142  }",
            "  ...",
            "2263      min_end_val = min_end_by_om.get((op, model))",
            "2268      valid_ends.append(float(min_end_val))",
            "  ...",
            "2297  feats[oi, pi, vmis, 3] = np.minimum(vends / sim_end_norm, 1.0)",
        ],
        "calc": [
            "(OPER002, model M1)의 free_at들: EQP001=320, EQP002=120, EQP003=400",
            "min_end_by_om[(OPER002,M1)] = min(320,120,400) = 120",
            "sim_end_norm = max(sim_end,1) = 480",
            "min_end_time = min(120/480, 1) = 0.25",
        ],
        "result": "min_end_time (ch3) = 0.25",
        "note": "질문: 'PPK,OPER002,M1' 아닌가? → 아니다. min_end_by_om의 키는 (OPER,model) 뿐이라 PPK는 "
                "관여하지 않는다. EQP002 자신이 idle(free_at=120)이라 min이 EQP002로 잡힌다 — "
                "'이 모델은 지금 당장도 비어있다'는 신호.",
    },
    {
        "group": "버킷 — ② 재공·부하 (WIP & Load)  ·  (PPK001, OPER002) 기준",
        "title": "ch4 throughput_ratio",
        "diagram": "ch4",
        "diagram_desc": (
            "분자는 이 버킷의 대기 재공량 wip_q(=15)다. 분모는 두 후보 중 더 큰 값을 쓴다 — "
            "①이 모델이 당장 비는 시각(min_end_time=120)에 처리 1단위 시간(st×wf_unit=5)을 더한 125, "
            "②팹 전체에서 이미 스케줄된 가장 늦은 종료시각 max_gantt_end(=320). 이 예시는 ②가 더 커서 "
            "320이 분모로 채택된다. 즉 throughput_ratio는 '대기 물량 대비, 이 라인을 실제로 언제 다시 "
            "쓸 수 있는가'를 정규화한 지표다 — 재공이 많아도 팹이 이미 꽉 차 있으면 값이 낮아지고, 재공이 "
            "적어도 라인이 한가하면 상대적으로 커진다. 정책이 '지금 투입하면 스루풋을 뽑아낼 여지가 있는 "
            "버킷'을 가려내도록 돕기 위해 필요하다."
        ),
        "lines": [
            "2166  max_gantt_end = max((r[\"END_TM\"] for r in self.schedule), default=0)",
            "2167  T_avail = max(self.soft_cutoff - self.current_time, 1)",
            "2168  max_takt = max(T_avail * wf_unit, 1.0)",
            "  ...",
            "2298      feats[oi, pi, vmis, 8] = vsts / max_arrange_st",
            "2299      proc_full_arr = vsts * wf_unit",
            "2300      denom_arr = np.maximum(",
            "2301          np.maximum(float(max_gantt_end), vends + proc_full_arr), 1.0",
            "2302      )",
            "2303      feats[oi, pi, vmis, 4] = max(wip_q, 0.0) / denom_arr",
        ],
        "calc": [
            "max_gantt_end = 320  (EQP001이 지금 처리 중인 bar의 종료시각, 스케줄 중 최대)",
            "vends(min_end_time, raw) = 120, st_per_wafer=5, wf_unit=1 → proc_full=5",
            "denom = max(320, 120+5) = max(320,125) = 320",
            "wip_q = 15  (분자 = 이 버킷의 대기 재공량, ch1/ch2와 동일한 wip_q)",
            "throughput_ratio = max(15,0) / 320 = 0.0469",
        ],
        "result": "throughput_ratio (ch4) = 0.0469",
        "note": "질문: 분자가 뭐고 왜 필요한가 → 분자는 wip_q(대기 재공)다. 분모가 '이미 스케줄된 최대 "
                "종료시각(320)'에 눌려있어, 이 버킷만 보면 125인데 실제론 다른 바가 더 길어서 320이 쓰인다 — "
                "재공 대비 '언제 다시 손댈 수 있는 라인인가'를 알려주는 신호.",
    },
    {
        "group": "버킷 — ④ 연속성·전환 (Continuity & Conversion)  ·  (PPK001, OPER002) 기준",
        "title": "ch5 same_ppk",
        "diagram": "ch5",
        "diagram_desc": (
            "직전 배정 PPK(PPK001)와 이 버킷의 PPK(PPK001)를 카드 두 장으로 맞대어 본다. 같은 제품이면 "
            "same=1.0(일치), 다른 제품이었다면 0.0(불일치)이 됐을 것이다. 이 채널은 Reward의 "
            "same_setup(+1.0) 판정과 완전히 같은 재료(직전 PPK 일치 여부)를 미리 State로 노출해, 정책이 "
            "'전환 없이 이어갈 수 있는 버킷'을 사전에 알아채도록 돕는다."
        ),
        "lines": [
            "2170  last_ppk = (",
            "2171      self._last_assigned.get(\"plan_prod_key\") if self._last_assigned else None",
            "2172  )",
            "  ...",
            "2229  same = 1.0 if ppk == last_ppk else 0.0",
            "  ...",
            "2282  feats[oi, pi, vmis, 5] = same",
        ],
        "calc": [
            "_last_assigned.plan_prod_key = \"PPK001\"  (직전 배정)",
            "이 버킷의 ppk = \"PPK001\"",
            "same = 1[PPK001 == PPK001] = 1.0",
        ],
        "result": "same_ppk (ch5) = 1.0",
        "note": "이 채널이 곧 Reward의 same_setup(+1.0) 판정과 같은 재료(PPK 일치 여부)를 미리 State로 보여주는 것.",
    },
    {
        "group": "버킷 — ③ 타이밍·여유 (Pacing & Takt)  ·  OPER001 vs OPER002 버킷 비교",
        "title": "ch6 prev_takt  ·  ch7 post_takt",
        "diagram": "ch6_7",
        "diagram_desc": (
            "eff_takt = 설비능력(cap_takt)과 계획수요(demand_takt) 중 더 여유로운 값. OPER002는 demand_takt(7.2)가 "
            "더 커서 그쪽을 채택 — 이 차이가 두 버킷의 prev/post_takt(max_takt=360 정규화)에 반대로 나타난다."
        ),
        "diagram_tall": True,
        "lines": [
            "2185  def eff_takt(ppk, op):",
            "2189      lst = arranges_by.get((ppk, op))",
            "2190      spw = (sum(s for _, s in lst) / len(lst)) if lst else None",
            "2191      n = max(n_eqp_per_oper.get(op, 0), 1)",
            "2192      cap_takt = (spw / n) if spw is not None else 0.0",
            "2193      pm = plan_meta.get((ppk, op))",
            "2194      if pm and pm.get(\"d0_plan_qty\", 0) > 0:",
            "2195          q_plan = max(pm[\"d0_plan_qty\"] - completed.get((ppk, op), 0), 1)",
            "2196          demand_takt = T_avail / q_plan",
            "2197          return max(demand_takt, cap_takt) * wf_unit",
            "2198      return cap_takt * wf_unit",
            "  ...",
            "2232  ppk_fp = flow_prev.get(ppk, {})",
            "2233  ppk_fpo = flow_post.get(ppk, {})",
            "2234  prev_takt = eff_takt(ppk, ppk_fp.get(op)) / max_takt",
            "2235  post_takt = eff_takt(ppk, ppk_fpo.get(op)) / max_takt",
        ],
        "calc": [
            "eff_takt(PPK001,OPER001): spw=2, n=2 → cap_takt=1 ; 계획없음(중간공정) → eff_takt=1*1=1",
            "eff_takt(PPK001,OPER002): spw=5, n=3 → cap_takt=1.67 ; q_plan=max(60-10,1)=50,",
            "  demand_takt=360/50=7.2 → eff_takt=max(7.2,1.67)*1=7.2",
            "max_takt = T_avail*wf_unit = 360",
            "",
            "[OPER001 버킷] prev_takt=eff_takt(flow_prev=없음)/360=0/360=0.0",
            "[OPER001 버킷] post_takt=eff_takt(OPER002)/360=7.2/360=0.02",
            "[OPER002 버킷] prev_takt=eff_takt(OPER001)/360=1/360=0.0028",
            "[OPER002 버킷] post_takt=eff_takt(flow_post=없음)/360=0/360=0.0",
        ],
        "result": "OPER001버킷: prev=0.0, post=0.02   /   OPER002버킷: prev=0.0028, post=0.0",
        "note": "eff_takt는 '계획을 맞추려면 허용되는 로트당 간격' — 클수록 여유(천천히 만들어도 됨). "
                "OPER002(=7.2)가 OPER001(=1)보다 훨씬 여유롭다는 게 OPER001버킷의 post_takt(0.02)로 드러난다. "
                "flow_prev/flow_post가 없는 끝단 공정은 eff_takt(None)=0.0으로 처리된다.",
    },
    {
        "group": "버킷 — ④ 연속성·전환 (Continuity & Conversion)  ·  (PPK001, OPER002) 기준",
        "title": "ch8 self_st",
        "diagram": "ch8",
        "diagram_desc": (
            "OPER001(2분/장)과 OPER002(5분/장)의 장당 가공시간을 막대로 비교한다. self_st는 이 "
            "(ppk,op,model)의 가공시간을 전체 (ppk,op,model) 중 최댓값(max_arrange_st)으로 정규화한 값 — "
            "값이 클수록 이 공정이 상대적으로 느리다는 뜻이다. OPER002가 이 예시의 최댓값이라 self_st=1.0 "
            "(가장 느린 공정)이 된다."
        ),
        "lines": [
            "2126  max_arrange_st = max(data.get(\"max_arrange_st\", 1), 1)",
            "  ...",
            "2266      st = st_per_wafer(ppk, op, model)",
            "  ...",
            "2269      valid_sts.append(float(st) if st is not None else 0.0)",
            "  ...",
            "2298  feats[oi, pi, vmis, 8] = vsts / max_arrange_st",
        ],
        "calc": [
            "st_per_wafer(PPK001,OPER002,M1) = 5분/장",
            "max_arrange_st = 전체 (ppk,op,model) 중 최댓값 = 5  (OPER002가 가장 느림)",
            "self_st = 5 / 5 = 1.0",
        ],
        "result": "self_st (ch8) = 1.0",
        "note": "이 예시에서 OPER002가 가장 느린 공정이라 정규화 최댓값과 같아져 1.0 — 극단값 예시.",
    },
    {
        "group": "버킷 — ③ 타이밍·여유 (Pacing & Takt)  ·  (PPK001, OPER002) 기준",
        "title": "ch9 plan_urgency",
        "diagram": "ch9",
        "diagram_desc": (
            "같은 gap(=plan_qty-completed=50)이라도 남은 시간(T_avail)이 360일 때와 30일 때 urgency "
            "게이지가 얼마나 달라지는지 두 게이지로 비교한다. urgency = min( (gap/T_avail)/plan_qty, 1 )로, "
            "분모에 T_avail이 들어가기 때문에 시간이 줄면 값이 반비례로 커진다 — 이 예시에서는 12배 뛴다."
        ),
        "lines": [
            "2224  pm = plan_meta.get((ppk, op))",
            "2225  if pm and pm.get(\"d0_plan_qty\", 0) > 0:",
            "2226      plan_qty = max(pm[\"d0_plan_qty\"], 1)",
            "2227      gap = max(plan_qty - completed.get((ppk, op), 0), 0)",
            "2228      urgency = min(gap / T_avail / plan_qty, 1.0)",
            "  ...",
            "2285  feats[oi, pi, vmis, 9] = urgency",
        ],
        "calc": [
            "plan_qty=60, completed=10 → gap=max(60-10,0)=50",
            "T_avail=360",
            "plan_urgency = min( (50/360)/60, 1 ) = min(0.00231, 1) = 0.0023",
        ],
        "result": "plan_urgency (ch9) = 0.0023",
        "note": "T_avail이 분모라서, 시간이 줄면(예: T_avail=30) 같은 gap=50이어도 urgency=min((50/30)/60,1)=0.0278로 12배 뜀.",
    },
    {
        "group": "버킷 — ④ 연속성·전환 (Continuity & Conversion)  ·  (PPK001, OPER002) 기준",
        "title": "ch10 wip_lot_cd  ·  ch11 wip_temp",
        "diagram": "ch10_11",
        "diagram_desc": (
            "LOT_CD 인덱스({B:0,A:1})와 TEMP 인덱스(이 예시는 미사용) 위에 이 버킷의 값(LOT_CD=A, "
            "TEMP=없음)을 점으로 찍어 encode_normalized가 0~1 스케일 위 어디로 보내는지 본다. "
            "encode_normalized(값, idx맵, n)은 idx맵에서 값의 순번을 찾아 (n-1)로 나눈다 — 값이 None이거나 "
            "idx맵에 없으면 0.0으로 떨어진다."
        ),
        "lines": [
            "2129  lot_cd_idx = data.get(\"lot_cd_idx\", {})",
            "2130  temp_idx = data.get(\"temp_idx\", {})",
            "2131  n_lc = max(len(lot_cd_idx), 1)",
            "2132  n_tp = max(len(temp_idx), 1)",
            "  ...",
            "2253  lc, tp = self._bucket_lot_cd_temp(ppk, op)",
            "2254  lc_enc = encode_normalized(lc or None, lot_cd_idx, n_lc)",
            "2255  tp_enc = encode_normalized(tp or None, temp_idx, n_tp)",
            "  ...",
            "2286  feats[oi, pi, vmis, 10] = lc_enc",
            "2287  feats[oi, pi, vmis, 11] = tp_enc",
        ],
        "calc": [
            "이 버킷의 lc=\"A\", tp=None",
            "lot_cd_idx={\"B\":0,\"A\":1}, n_lc=2 → lc_enc = encode(\"A\") = 1/(2-1) = 1.0",
            "temp_idx={} (TEMP 미사용), n_tp=1 → tp_enc = encode(None) = 0.0  (값이 None이라 기본값 0)",
        ],
        "result": "wip_lot_cd (ch10) = 1.0   /   wip_temp (ch11) = 0.0",
        "note": "ch11의 0.0은 'TEMP 조건이 아예 없다'는 뜻이지, 오류나 결측이 아니다 — encode_normalized(None,...)의 정상 동작.",
    },
    {
        "group": "버킷 — ④ 연속성·전환 (Continuity & Conversion)  ·  current_eqp=EQP002 기준",
        "title": "ch12 needs_conversion  ·  ch13 tool_can_assign",
        "diagram": "ch12_13",
        "diagram_desc": (
            "EQP002의 현재 세팅(B)과 이 버킷이 요구하는 세팅(A)을 맞대어 불일치를 보여준다(needs_conversion"
            "=1.0). tool_can_assign은 OR 조건이다 — '애초에 공구 교체가 필요 없다' 또는 '교체가 필요해도 "
            "공구 트래커가 지금 배정 가능하다고 판단' 둘 중 하나만 참이어도 1.0이 된다. 이 예시는 공구 교체가 "
            "필요 없다고 가정해 바로 1.0이 된다."
        ),
        "lines": [
            "2144  current_eqp_model = eqp_model_map.get(current_eqp) if current_eqp else None",
            "  ...",
            "2205  curr_eqp_in_om: bool = (",
            "2206      current_eqp is not None",
            "2207      and current_eqp_model is not None",
            "2208      and current_eqp in self._eqps_by_om.get((op, current_eqp_model), [])",
            "2209  )",
            "  ...",
            "2306  if curr_eqp_in_om and current_mi >= 0 and current_mi in valid_mis and lc:",
            "2307      feats[oi, pi, current_mi, 12] = (",
            "2308          1.0 if self._would_need_conversion(current_eqp, lc, tp) else 0.0",
            "2309      )",
            "2310      feats[oi, pi, current_mi, 13] = (",
            "2311          1.0 if not self._needs_tool_swap(current_eqp, lc, tp)",
            "2312          or self._tool_tracker.can_assign(lc, current_eqp) else 0.0",
            "2313      )",
        ],
        "calc": [
            "current_eqp=EQP002, EQP002는 (OPER002,M1)에 속함 → curr_eqp_in_om=True",
            "EQP002.prev_lot_cd=\"B\", 이 버킷 lc=\"A\" → would_need_conversion(EQP002,\"A\",None)=True",
            "needs_conversion = 1.0",
            "",
            "needs_tool_swap(EQP002,\"A\",None) = False (가정: 공구 교체 불필요)",
            "→ \"not False\" = True → OR 조건 만족(뒤 항 평가 불필요)",
            "tool_can_assign = 1.0",
        ],
        "result": "needs_conversion (ch12) = 1.0   /   tool_can_assign (ch13) = 1.0",
        "note": "이 두 채널은 current_eqp가 속한 (op,model) 한 칸에만 기록되고, 나머지 칸은 0으로 남는다.",
    },
    {
        "group": "버킷 — ⑤ 달성·커버리지 (Achievability & Coverage)  ·  (PPK001, OPER002) 기준",
        "title": "ch14 achievable_ratio",
        "diagram": "ch14",
        "diagram_desc": (
            "완료(10) + 자기 WIP(15) + 상류 OPER001 WIP(140)를 순서대로 쌓아가는 누적 막대가 plan_qty=60 "
            "기준선을 훌쩍 넘는 모습을 보여준다. reachable은 이 공정부터 flow_prev 체인을 거슬러 올라가며 "
            "wip_po를 더한 값 — 상류 재공이 충분하면 achievable_ratio가 1.0으로 캡핑되어 '계획을 못 채울 "
            "걱정은 없다'는 신호를 준다."
        ),
        "lines": [
            "2239  achievable_ratio = 1.0",
            "2240  if pm and pm.get(\"d0_plan_qty\", 0) > 0:",
            "2241      plan_qty = max(pm[\"d0_plan_qty\"], 1)",
            "2242      done = completed.get((ppk, op), 0)",
            "2243      reachable = wip_po.get((ppk, op), 0.0)",
            "2244      seen = {op}",
            "2245      prev_op = ppk_fp.get(op)",
            "2246      while prev_op and prev_op not in seen:",
            "2247          seen.add(prev_op)",
            "2248          reachable += wip_po.get((ppk, prev_op), 0.0)",
            "2249          prev_op = ppk_fp.get(prev_op)",
            "2250      achievable_ratio = min(min(plan_qty, done + reachable) / plan_qty, 1.0)",
            "  ...",
            "2288  feats[oi, pi, vmis, 14] = achievable_ratio",
        ],
        "calc": [
            "plan_qty=60, done=10",
            "reachable = wip_po[(PPK001,OPER002)]=15  (자기 자신)",
            "→ 상류(OPER001) 순회: reachable += wip_po[(PPK001,OPER001)]=140  → reachable=155",
            "achievable_ratio = min( min(60, 10+155) / 60, 1 ) = min(60/60, 1) = 1.0",
        ],
        "result": "achievable_ratio (ch14) = 1.0",
        "note": "상류 재공(140장)이 충분해서 achievable이 plan_qty(60)에 그대로 캡핑됨 — 재공 부족 걱정이 없는 상태.",
    },
    {
        "group": "버킷 — ⑤ 달성·커버리지 (Achievability & Coverage)  ·  (PPK001, OPER002) 기준",
        "title": "ch15 projected_cover_ratio",
        "diagram": "ch15",
        "diagram_desc": (
            "current_eqp(EQP002)를 뺀 나머지 설비 중 '직전에 이 버킷과 같은 흐름(prev_prod=PPK001, "
            "prev_oper=OPER002)'이던 설비만 커버량에 반영된다 — EQP001은 직전 공정이 OPER001이라 제외되고, "
            "EQP003만 포함되어 remaining(=soft_cutoff-t=360)/st(=5)=72를 기여한다. 이 커버량(72)을 남은 "
            "필요량(need=50)과 비교해 0.72라는 비율을 얻는다."
        ),
        "lines": [
            "2289  # 채널 15: 투영 커버 비율 — 다른 장비(current_eqp 제외)가 이 버킷을",
            "2290  # 하루 끝까지 덮는 양 / 남은 필요량. 1에 가까울수록 '이미 덮임' → 회피 유도.",
            "2291  done15 = completed.get((ppk, op), 0)",
            "2292  need15 = max(self._achievable_qty(ppk, op) - done15, 1.0)",
            "2293  cov15 = self._bucket_projected_cover(ppk, op, exclude_eqp=current_eqp)",
            "2294  feats[oi, pi, vmis, 15] = min(cov15 / need15, 2.0) / 2.0",
        ],
        "calc": [
            "_achievable_qty(PPK001,OPER002) = min(60, 10+155) = 60  → need15 = max(60-10,1) = 50",
            "",
            "_bucket_projected_cover: EQP002(본인) 제외, prev_prod==PPK001 & prev_oper==OPER002인 '다른' 설비만",
            "  EQP001: prev_oper=OPER001 → 불일치, 제외",
            "  EQP003: prev_oper=OPER002, prev_prod=PPK001 → 일치! (busy 여부는 이 함수에서 안 봄)",
            "    remaining=max(soft_cutoff-t,0)=360, st=5 → 기여 = 360/5 = 72",
            "cov15 = 72",
            "",
            "projected_cover_ratio = min(72/50, 2) / 2 = min(1.44,2)/2 = 0.72",
        ],
        "result": "projected_cover_ratio (ch15) = 0.72",
        "note": "_bucket_projected_cover는 설비의 '지금 busy 여부'를 빼지 않는다(반면 avoidable_frac의 alt_cap 계산은 뺀다) "
                "— 같은 '다른 설비 capa' 개념이라도 두 함수가 정밀도가 다르다는 점이 이 코드베이스의 흥미로운 디테일.",
    },
    {
        "group": "현재 설비 — 전환 필요성과 회피가능성 (Conversion Need & Avoidability)  ·  current_eqp=EQP002",
        "title": "eqp[0] needs_conversion  ·  eqp[1] avoidable_frac",
        "diagram": "eqp",
        "diagram_desc": (
            "세팅A를 필요로 하는 총수요 155장(=OPER001 WIP 140 + OPER002 WIP 15)을 다른 설비가 얼마나 "
            "대신 커버할 수 있는지 막대로 쪼갠다. alt_cap은 EQP001·EQP003 각각의 '남은 가용시간 / "
            "best_st(가장 빠른 처리시간)'을 합산한 값(80+16=96) — 이 몫(96/155=0.619)이 곧 이 전환을 "
            "감행했을 때 '사실 다른 설비가 대신할 수 있었던' avoidable_frac이 된다."
        ),
        "lines": [
            "2340  eqp_local = np.zeros(2, dtype=np.float32)",
            "2341  current_eqp_id = self._current_eqp",
            "2342  if current_eqp_id and current_eqp_id in self.eqps:",
            "2343      max_avoidable = 0.0",
            "2344      for flat in self.get_feasible_ppk_oper(current_eqp_id):",
            "2345          ppk_f, oper_f = self.ppk_oper_from_flat(flat)",
            "2346          lc, tp = self._bucket_lot_cd_temp(ppk_f, oper_f)",
            "2347          if self._would_need_conversion(current_eqp_id, lc, tp):",
            "2348              eqp_local[0] = 1.0",
            "2349              av = self._conversion_avoidable_fraction(",
            "2350                  current_eqp_id, ppk_f, oper_f, lc, tp,",
            "2351              )",
            "2352              if av > max_avoidable:",
            "2353                  max_avoidable = av",
            "2354      eqp_local[1] = max_avoidable",
        ],
        "calc": [
            "EQP002가 고를 수 있는 feasible 버킷 중 (PPK001,OPER002)는 세팅 A 필요(EQP002는 B) → 전환 필요 → eqp[0]=1.0",
            "",
            "α 계산 (_conversion_avoidable_fraction):",
            "demand = 세팅A 공유 ready WIP 합 = 140(OPER001)+15(OPER002) = 155",
            "alt_cap: EQP001 남은가용=max(360-max(320-120,0),0)=160, best_st=2(OPER001) → 160/2=80",
            "         EQP003 남은가용=max(360-max(400-120,0),0)=80,  best_st=5(OPER002) → 80/5=16",
            "         alt_cap = 80+16 = 96",
            "coverage_frac = min(96/155, 1) = 0.619",
            "residual = max(155-96,0)=59, my_st=5, my_cap=360/5=72",
            "my_run_min = min(72,59)*5 = 295, need=1.0×60=60",
            "short_run_frac = max(0, 1-295/60) = 0  (이미 넉넉히 돌 수 있어 '짧아서 아깝다'는 근거 없음)",
            "α = max(0.619, 0) = 0.619",
        ],
        "result": "needs_conversion (eqp[0]) = 1.0   /   avoidable_frac (eqp[1]) = 0.619",
        "note": "α=0.619 → 이 전환의 62%는 다른 설비가 대신 커버 가능했다는 뜻 "
                "→ 실제로 전환을 감행하면 avoidable_conversion = -8.0×0.619 ≈ -4.95 페널티.",
    },
    {
        "group": "직전 맥락 — 마지막 배정 기록 (Last Assignment Context)  ·  EQP003→PPK001/OPER002/세팅A",
        "title": "ctx[0~3] last_ppk · last_oper · last_eqp · last_lot_cd",
        "diagram": "ctx",
        "diagram_desc": (
            "직전 배정 레코드(PPK001·OPER002·EQP003·세팅A) 카드 4장이 각자의 인덱스맵을 거쳐 0~1 스케일 "
            "위 어느 지점으로 인코딩되는지 화살표로 연결해 본다. last_oper만 예외적으로 lot_id를 통해 "
            "data['lots'] 테이블을 역추적해서 oper_id를 알아낸다 — 나머지 세 값은 _last_assigned 레코드에서 "
            "바로 꺼낸다."
        ),
        "lines": [
            "2375  context = np.zeros(4, dtype=np.float32)",
            "2376  if self._last_assigned:",
            "2377      la = self._last_assigned",
            "2378      context[0] = encode_normalized(la.get(\"plan_prod_key\"), prod_idx, P)",
            "2381      oper_guess = None",
            "2382      for ld in data.get(\"lots\", []):",
            "2383          if ld[\"lot_id\"] == la.get(\"lot_id\"):",
            "2384              oper_guess = ld.get(\"oper_id\")",
            "2385              break",
            "2386      context[1] = encode_normalized(oper_guess, oper_idx, O)",
            "2387      context[2] = encode_normalized(la.get(\"eqp_id\"), eqp_idx, len(data[\"eqp_ids\"]))",
            "2388      context[3] = encode_normalized(la.get(\"lot_cd\"), lot_cd_idx, max(len(lot_cd_idx), 1))",
        ],
        "calc": [
            "la = {plan_prod_key:PPK001, lot_id:LOT099, eqp_id:EQP003, lot_cd:A}",
            "prod_idx={PPK002:0,PPK001:1}, P=10 → last_ppk = 1/(10-1) = 0.111",
            "",
            "lot_id=LOT099 → data['lots']에서 역추적 → oper_guess=OPER002",
            "oper_idx={OPER001:0,OPER002:1}, O=3 → last_oper = 1/(3-1) = 0.5",
            "",
            "eqp_idx={EQP001:0,EQP002:1,EQP003:2} → last_eqp = 2/(3-1) = 1.0",
            "lot_cd_idx={B:0,A:1} → last_lot_cd = 1/(2-1) = 1.0",
        ],
        "result": "last_ppk=0.111 / last_oper=0.5 / last_eqp=1.0 / last_lot_cd=1.0",
        "note": "oper_guess는 lot_id로 원본 lots 테이블을 뒤져서 찾는다 — 못 찾으면 None → encode 결과 0.0으로 빠진다.",
    },
]
