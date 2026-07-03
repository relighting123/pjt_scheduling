"""
TOOL_CHANGE_BENCH 데이터셋 생성기 — "tool 교체(전환)" 대표 스케줄링 문제 10종
=========================================================================
반도체 라인에서 실제로 마주치는 3가지 대표 상황을 각각 여러 난이도로 구성한다.

  A. 단일 제품 · 공정별 재공 충분 (SINGLE)
     제품 1개, 여러 공정이 각각 전담 장비 + 충분한 초기 재공을 보유.
     전환이 아예 필요 없는 대조군 — 알고리즘이 '쉬운 문제'에서 100%를
     달성하는지 확인하는 기준선.

  B. 다품종 · 공정별 재공 충분, 장비 공유 (MULTI)
     제품 여러 개가 공정 단계별로(OPER001→OPER002) 장비를 공유.
     제품 수 > 장비 수(불가피한 전환), 물량 편중, 전환시간≫처리시간 등
     BENCH_SUITE/CONV_BENCH의 단일 공정 아이디어를 2단계 flow로 확장.

  C. 재공 편중 + 안전재공(safety WIP) 확보 후 전환 (SAFETY)
     한 제품이 OPER001(재공 다량) → OPER002(재공 0, OPER001 완료로만 유입)로
     흐르고, 두 공정을 겸용 가능한 장비가 있다. 단순 takt/진행률 기준(예:
     min-progress)은 OPER002에 재공이 한 장만 생겨도 즉시 전환해버려 '전환
     쏠림(chattering)'이 발생한다. 최적 정책은 OPER001을 안전재공이 쌓일
     때까지 진행하다가 '한 번만' 전환해 OPER002를 끝까지 처리하는 것이다.

각 데이터셋은 최적해를 '수식으로 우겨넣은 값'이 아니라, 실제로 달성 가능한
구성적(constructive) 스케줄을 손으로 설계해 도출한 값이다 — 자세한 도출 방법은
`data/dataset/tool_change_bench_meta.json`의 `optimal.derivation`에 기록한다.
（검증: `benchmark/tool_change_bench.py`가 오라클 정책으로 실제 시뮬레이터를
구동해 이 값이 진짜로 달성되는지 재확인한다.）
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
SUITE_ROOT = ROOT / "data/dataset"
TIMEKEY = "20260703000000"
TEMP = "T600"
TOOL_MAX = 99


def _write(bench_id: str, files: dict) -> Path:
    out = SUITE_ROOT / bench_id / "train" / TIMEKEY / "input"
    out.mkdir(parents=True, exist_ok=True)
    for fn, data in files.items():
        with open(out / fn, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return out


def _carrier_rows(ppk, oper, st, n, eqp_cycle, lot_prefix, model):
    """n개의 WF_QTY=1 carrier(=lot) 행. EQP_ID는 eqp_cycle을 순환 배정(홈 배정)."""
    rows = []
    for i in range(n):
        eqp = eqp_cycle[i % len(eqp_cycle)]
        rows.append({
            "EQP_ID": eqp, "LOT_ID": f"{lot_prefix}{i + 1:03d}",
            "PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper, "ST": st,
            "EQP_MODEL_CD": model, "WF_QTY": 1, "SEQ": 1,
            "CARRIER_ID": f"{lot_prefix}C{i + 1:03d}",
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════
# A. 단일 제품 · 공정 전담 · 재공 충분 (대조군)
# ═══════════════════════════════════════════════════════════════════════

def gen_single_dedicated(bench_id, desc, tests, opers):
    """opers: [{"oper":..,"st":..,"n_eqp":..,"wip":..,"plan":..}, ...] 순서대로 flow."""
    ppk = "PPK001"
    flow, plan, abstract, discrete, lot_master, batch, tool_cd = [], [], [], [], [], [], []
    eqp_counter = 0
    for seq, o in enumerate(opers, start=1):
        model = f"M{seq}"
        flow.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": seq, "OPER_ID": o["oper"]})
        plan.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": o["oper"],
                      "D0_PLAN_QTY": o["plan"], "D1_PLAN_QTY": o["plan"], "PLAN_PRIORITY": 1})
        abstract.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": o["oper"],
                          "EQP_MODEL_CD": model, "ST": o["st"]})
        eqps = []
        for _ in range(o["n_eqp"]):
            eqp_counter += 1
            eqps.append(f"EQP{eqp_counter:03d}")
        lot_cd = f"LC_{o['oper']}"
        batch.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": o["oper"], "LOT_CD": lot_cd, "TEMP": TEMP})
        tool_cd.append({"LOT_CD": lot_cd, "TEMP": TEMP, "MAX_TOOL": TOOL_MAX})
        rows = _carrier_rows(ppk, o["oper"], o["st"], o["wip"], eqps, f"L{seq}_", model)
        discrete.extend(rows)
        lot_master.extend({"LOT_ID": r["LOT_ID"], "LOT_CD": lot_cd, "TEMP": TEMP} for r in rows)

    split = [{"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": o["oper"], "EQP_MODEL_CD": f"M{seq}", "SPLIT_QTY": 1}
             for seq, o in enumerate(opers, start=1)]
    files = {
        "abstract_arrange.json": abstract, "discrete_arrange.json": discrete,
        "lot_master.json": lot_master, "plan.json": plan, "flow.json": flow,
        "split.json": split, "batch_info.json": batch, "tool_capacity.json": tool_cd,
    }
    _write(bench_id, files)
    return dict(files=files, n_eqp=eqp_counter)


# ═══════════════════════════════════════════════════════════════════════
# B. 다품종 · 2단계 flow · 장비 공유 (전환 불가피/편중/전환과중)
# ═══════════════════════════════════════════════════════════════════════

def gen_multi_stage(bench_id, stage1, stage2):
    """stage1/stage2: dict(oper, n_eqp, ppks=[{ppk,carriers,st,group?}], conv, scramble)
    각 stage는 독립된 EQP 풀(모델도 stage별로 분리)을 가진다.

    `group`(0-based EQP index, 선택)을 지정하면 그 제품은 해당 EQP만 처리 가능한
    전용 모델을 쓴다(같은 group끼리는 서로 공유). 미지정이면 전 EQP 공유 모델
    (완전 유연 장비, TCB04/06처럼 '스스로 전담을 찾아내는지'를 테스트)."""
    flow, plan, abstract, discrete, lot_master, batch, tool_cd = [], [], [], [], [], [], []
    split = []
    eqp_counter = 0
    stage_dims = []
    for si, stage in enumerate((stage1, stage2), start=1):
        oper = stage["oper"]
        n_eqp = stage["n_eqp"]
        eqps = []
        for _ in range(n_eqp):
            eqp_counter += 1
            eqps.append(f"EQP{eqp_counter:03d}")
        for pi, p in enumerate(stage["ppks"]):
            ppk = p["ppk"]
            group = p.get("group")
            model = f"S{si}" if group is None else f"S{si}_E{group}"
            lot_cd = f"LC_{oper}_{ppk}"
            if si == 1:
                flow.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": stage1["oper"]})
                flow.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 2, "OPER_ID": stage2["oper"]})
            plan.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper,
                         "D0_PLAN_QTY": p["carriers"], "D1_PLAN_QTY": p["carriers"], "PLAN_PRIORITY": 1})
            abstract.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper,
                              "EQP_MODEL_CD": model, "ST": p["st"]})
            batch.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper, "LOT_CD": lot_cd, "TEMP": TEMP})
            tool_cd.append({"LOT_CD": lot_cd, "TEMP": TEMP, "MAX_TOOL": TOOL_MAX})
            split.append({"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": oper, "EQP_MODEL_CD": model, "SPLIT_QTY": 1})
            if group is not None:
                # 전용 그룹: 그 EQP 1대만 홈 배정(모델 자체가 그 EQP 전용이라 스크램블 불필요)
                eqp_cycle = [eqps[group]]
            elif stage.get("scramble"):
                # 홈 배정: (pi + carrier_idx) % n_eqp로 섞어 배정
                eqp_cycle = [eqps[(pi + i) % n_eqp] for i in range(p["carriers"])]
            else:
                eqp_cycle = [eqps[pi % n_eqp]]
            rows = _carrier_rows(ppk, oper, p["st"], p["carriers"], eqp_cycle, f"S{si}_{ppk}_", model)
            discrete.extend(rows)
            lot_master.extend({"LOT_ID": r["LOT_ID"], "LOT_CD": lot_cd, "TEMP": TEMP} for r in rows)
        stage_dims.append(dict(oper=oper, n_eqp=n_eqp, conv=stage["conv"]))

    files = {
        "abstract_arrange.json": abstract, "discrete_arrange.json": discrete,
        "lot_master.json": lot_master, "plan.json": plan, "flow.json": flow,
        "split.json": split, "batch_info.json": batch, "tool_capacity.json": tool_cd,
    }
    _write(bench_id, files)
    return stage_dims


# ═══════════════════════════════════════════════════════════════════════
# C. 재공 편중 + 안전재공 확보 후 전환 (동일 제품, 2공정 겸용 장비)
# ═══════════════════════════════════════════════════════════════════════

def gen_safety_switch(bench_id, n_eqp, up, down, conv):
    """up/down: dict(oper, st, wip, plan). 동일 PPK, 겸용 장비 n_eqp대."""
    ppk = "PPK001"
    model = "A"
    eqps = [f"EQP{i + 1:03d}" for i in range(n_eqp)]
    flow = [
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 1, "OPER_ID": up["oper"]},
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_SEQ": 2, "OPER_ID": down["oper"]},
    ]
    plan = [
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": up["oper"],
         "D0_PLAN_QTY": up["plan"], "D1_PLAN_QTY": up["plan"], "PLAN_PRIORITY": 1},
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": down["oper"],
         "D0_PLAN_QTY": down["plan"], "D1_PLAN_QTY": down["plan"], "PLAN_PRIORITY": 1},
    ]
    abstract = [
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": up["oper"], "EQP_MODEL_CD": model, "ST": up["st"]},
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": down["oper"], "EQP_MODEL_CD": model, "ST": down["st"]},
    ]
    lot_cd_up, lot_cd_down = f"LC_{up['oper']}", f"LC_{down['oper']}"
    batch = [
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": up["oper"], "LOT_CD": lot_cd_up, "TEMP": TEMP},
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": down["oper"], "LOT_CD": lot_cd_down, "TEMP": TEMP},
    ]
    tool_cd = [
        {"LOT_CD": lot_cd_up, "TEMP": TEMP, "MAX_TOOL": TOOL_MAX},
        {"LOT_CD": lot_cd_down, "TEMP": TEMP, "MAX_TOOL": TOOL_MAX},
    ]
    split = [
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": up["oper"], "EQP_MODEL_CD": model, "SPLIT_QTY": 1},
        {"PLAN_PROD_ATTR_VAL": ppk, "OPER_ID": down["oper"], "EQP_MODEL_CD": model, "SPLIT_QTY": 1},
    ]
    # OPER002(하류)는 초기 재공 0 — discrete_arrange에 아예 등록하지 않는다.
    # (등록해도 eligibility는 abstract_arrange만으로 충분 — _eqp_can_process 참고)
    discrete = _carrier_rows(ppk, up["oper"], up["st"], up["wip"], eqps, "UP_", model)
    lot_master = [{"LOT_ID": r["LOT_ID"], "LOT_CD": lot_cd_up, "TEMP": TEMP} for r in discrete]

    files = {
        "abstract_arrange.json": abstract, "discrete_arrange.json": discrete,
        "lot_master.json": lot_master, "plan.json": plan, "flow.json": flow,
        "split.json": split, "batch_info.json": batch, "tool_capacity.json": tool_cd,
    }
    _write(bench_id, files)
    return eqps


# ═══════════════════════════════════════════════════════════════════════
# 메타(정답지) — 각 데이터셋의 sim_end / conversion_minutes / 최적해
# ═══════════════════════════════════════════════════════════════════════

META = []


def _add_meta(**kw):
    META.append(kw)


def build_all():
    # ---- A1: 2공정 균형 ----
    gen_single_dedicated(
        "TCB01_SINGLE_2OP", "1제품·2공정 전담, 재공 충분(대조군)", "쉬운 문제에서 100% 달성 확인",
        opers=[dict(oper="OPER001", st=20, n_eqp=1, wip=30, plan=24),
               dict(oper="OPER002", st=20, n_eqp=1, wip=30, plan=24)],
    )
    _add_meta(
        id="TCB01_SINGLE_2OP", category="A. 단일제품 재공충분", n_ppk=1,
        sim_end_minutes=480, conversion_minutes=60,
        desc="PPK001 단일, OPER001→OPER002 각 전담 1대, 재공 30장(계획 24장 상회).",
        test_focus="전환 유인이 전혀 없는 문제에서 알고리즘이 100% 생산을 달성하는지(대조군).",
        optimal=dict(production=48, conversions=0,
                     by_oper={"OPER001": 24, "OPER002": 24},
                     derivation="전담 장비 1:1이므로 전환은 항상 회피 가능. "
                                "장비당 처리량 상한 = floor(480/20)=24개. "
                                "재공(30)>계획(24)이므로 상한이 곧 최적. 0전환은 자명."),
    )

    # ---- A2: 3공정 병목(장비 2대) ----
    gen_single_dedicated(
        "TCB02_SINGLE_3OP_BOTTLENECK", "1제품·3공정 전담, 병목 공정 2대 보강", "다중 장비 병목 공정 완전 활용 확인",
        opers=[dict(oper="OPER001", st=15, n_eqp=1, wip=40, plan=32),
               dict(oper="OPER002", st=15, n_eqp=1, wip=40, plan=32),
               dict(oper="OPER003", st=30, n_eqp=2, wip=40, plan=32)],
    )
    _add_meta(
        id="TCB02_SINGLE_3OP_BOTTLENECK", category="A. 단일제품 재공충분", n_ppk=1,
        sim_end_minutes=480, conversion_minutes=60,
        desc="OPER001(ST15,1대)→OPER002(ST15,1대)→OPER003(ST30,2대, 병목 보강). 재공 40장/공정.",
        test_focus="병목 공정에 배치된 장비 2대를 모두 100% 활용해 병목을 해소하는지.",
        optimal=dict(production=96, conversions=0,
                     by_oper={"OPER001": 32, "OPER002": 32, "OPER003": 32},
                     derivation="세 공정 모두 장비수×floor(480/ST)=32로 균형 설계(1×32, 1×32, 2×16×... "
                                "정확히는 OPER003=2대×floor(480/30)=2×16=32). 전담이므로 0전환. "
                                "재공(40)>계획(32) → 상한=32/공정이 최적."),
    )

    # ---- A3: 4공정 체인 ----
    gen_single_dedicated(
        "TCB03_SINGLE_4OP_CHAIN", "1제품·4공정 체인 전담, 재공 충분", "긴 체인에서 페이싱 유지 확인",
        opers=[dict(oper="OPER001", st=12, n_eqp=1, wip=60, plan=50),
               dict(oper="OPER002", st=12, n_eqp=1, wip=60, plan=50),
               dict(oper="OPER003", st=24, n_eqp=2, wip=60, plan=50),
               dict(oper="OPER004", st=12, n_eqp=1, wip=60, plan=50)],
    )
    _add_meta(
        id="TCB03_SINGLE_4OP_CHAIN", category="A. 단일제품 재공충분", n_ppk=1,
        sim_end_minutes=600, conversion_minutes=60,
        desc="OPER001→002→003(2대)→004, ST=12/12/24/12. 재공 60장/공정.",
        test_focus="4단계 체인 전체에서 병목 없이 균등 페이싱 유지.",
        optimal=dict(production=200, conversions=0,
                     by_oper={"OPER001": 50, "OPER002": 50, "OPER003": 50, "OPER004": 50},
                     derivation="장비수×floor(600/ST): 1×50, 1×50, 2×25=50, 1×50 → 모두 50으로 균형. "
                                "전담·재공충분이므로 0전환·상한 100% 달성이 최적."),
    )

    # ---- B1: 대칭 대조군(2단계, 홈배정 스크램블) ----
    d1 = gen_multi_stage(
        "TCB04_MULTI_SYM_2STAGE",
        stage1=dict(oper="OPER001", n_eqp=3, conv=40, scramble=True,
                    ppks=[dict(ppk=f"PPK00{i+1}", carriers=8, st=40) for i in range(3)]),
        stage2=dict(oper="OPER002", n_eqp=3, conv=40, scramble=True,
                    ppks=[dict(ppk=f"PPK00{i+1}", carriers=8, st=40) for i in range(3)]),
    )
    _add_meta(
        id="TCB04_MULTI_SYM_2STAGE", category="B. 다품종 재공충분", n_ppk=3,
        sim_end_minutes=320, conversion_minutes=40,
        desc="3제품×2공정(OPER001/002 각 독립 장비 3대), 홈 lot는 의도적으로 3제품 섞어 배정.",
        test_focus="장비=제품 수가 같아 완전 전담이 최적인데도 섞인 홈배정에 낚여 불필요한 전환을 내는지.",
        optimal=dict(production=48, conversions=0,
                     by_stage={"OPER001": {"production": 24, "conversions": 0},
                               "OPER002": {"production": 24, "conversions": 0}},
                     derivation="각 단계 3대×3제품, 제품당 8carrier×40min=320min=sim_end 정확히 일치. "
                                "장비당 1제품 전담(비홈 lot도 abstract 경로로 선택 가능)이면 0전환으로 "
                                "정확히 24개/단계 완결 — 이보다 많은 생산은 재공 총량(24) 초과라 불가능."),
    )

    # ---- B2: 제품 과잉(2단계) ----
    _over_ppks = [dict(ppk=f"PPK00{i+1}", carriers=4, st=45, group=[0, 0, 1, 1, 2][i]) for i in range(5)]
    gen_multi_stage(
        "TCB05_MULTI_OVER_2STAGE",
        stage1=dict(oper="OPER001", n_eqp=3, conv=45, ppks=_over_ppks),
        stage2=dict(oper="OPER002", n_eqp=3, conv=45, ppks=_over_ppks),
    )
    _add_meta(
        id="TCB05_MULTI_OVER_2STAGE", category="B. 다품종 재공충분", n_ppk=5,
        sim_end_minutes=405, conversion_minutes=45,
        desc="5제품×2공정, 각 공정 장비 3대뿐(EQP1=제품A/B, EQP2=제품C/D, EQP3=제품E 전용 그룹) "
             "→ 제품 과잉으로 전환 불가피(비둘기집 하한=2/공정).",
        test_focus="불가피한 전환을 최소 횟수(2/공정, 장비당 최대 1회)로 묶어 손실을 최소화하는지 — "
                    "장비 내에서 A→B→A처럼 왔다갔다 하면 하한을 초과해 손실이 커진다.",
        optimal=dict(production=40, conversions=4,
                     by_stage={"OPER001": {"production": 20, "conversions": 2},
                               "OPER002": {"production": 20, "conversions": 2}},
                     derivation="구성적 증명(단계별 동일): 3대에 5제품을 2/2/1로 배분 "
                                "→ 전환 (2-1)+(2-1)+(1-1)=2회(하한: 5제품-3대=2, 달성). "
                                "M1=A+B(4+4carrier×45=360min+conv45=405), M2=C+D(405), M3=E(180min). "
                                "makespan=405=sim_end. 전량(20/20) 생산 + 최소전환 동시 달성 → 최적. "
                                "장비별 담당 제품 그룹을 전용 장비모델로 고정해(EQP1↔{A,B}, EQP2↔{C,D}, "
                                "EQP3↔{E}) 의도치 않은 교차 배정을 원천 차단했고, 실제 시뮬레이터 오라클 "
                                "정책으로 이 스케줄이 그대로 달성됨을 검증함(benchmark/tool_change_bench.py "
                                "--verify)."),
    )

    # ---- B3: 물량 편중(2단계) ----
    gen_multi_stage(
        "TCB06_MULTI_LOADSKEW_2STAGE",
        stage1=dict(oper="OPER001", n_eqp=4, conv=30, scramble=True,
                    ppks=[dict(ppk="PPK001", carriers=12, st=30), dict(ppk="PPK002", carriers=8, st=30),
                          dict(ppk="PPK003", carriers=6, st=30), dict(ppk="PPK004", carriers=4, st=30)]),
        stage2=dict(oper="OPER002", n_eqp=4, conv=30, scramble=True,
                    ppks=[dict(ppk="PPK001", carriers=12, st=30), dict(ppk="PPK002", carriers=8, st=30),
                          dict(ppk="PPK003", carriers=6, st=30), dict(ppk="PPK004", carriers=4, st=30)]),
    )
    _add_meta(
        id="TCB06_MULTI_LOADSKEW_2STAGE", category="B. 다품종 재공충분", n_ppk=4,
        sim_end_minutes=360, conversion_minutes=30,
        desc="4제품×2공정, 장비=제품 수(4)지만 물량 12/8/6/4로 편중. 장비는 4제품 모두 처리 가능(공유).",
        test_focus="장비수=제품수라도 '균등 분배'가 아니라 '물량 비례 배분'이 최적임을 확인 "
                    "(섞인 홈배정을 무시하고 제품별 전담 시간을 다르게 가져가야 함).",
        optimal=dict(production=60, conversions=0,
                     by_stage={"OPER001": {"production": 30, "conversions": 0},
                               "OPER002": {"production": 30, "conversions": 0}},
                     derivation="장비 1대를 제품 1개에 전담 배정(물량이 큰 제품일수록 그 장비를 오래 점유): "
                                "12×30=360, 8×30=240, 6×30=180, 4×30=120(모두 sim_end=360 이내). "
                                "전담이므로 0전환, 재공 전량(30/단계) 소진이 상한 — 동시 달성. 물량이 적은 "
                                "제품 담당 장비는 자기 몫을 일찍 끝내고 남은 시간 유휴 상태가 되는데, 이는 "
                                "비용이 전혀 없으므로(전환 0 유지) 그대로 최적 — 실제 오라클 정책을 "
                                "max_conversions_per_eqp=0으로 강제해(유휴 장비가 남의 제품을 대신 돕지 "
                                "못하게 막음) 이 0전환·전량생산이 동시에 달성 가능함을 시뮬레이터로 "
                                "검증함(--verify)."),
    )

    # ---- B4: 전환 과중(2단계) ----
    _convheavy_ppks = [dict(ppk=f"PPK00{i+1}", carriers=4, st=40, group=[0, 0, 1, 1, 2][i]) for i in range(5)]
    gen_multi_stage(
        "TCB07_MULTI_CONVHEAVY_2STAGE",
        stage1=dict(oper="OPER001", n_eqp=3, conv=120, ppks=_convheavy_ppks),
        stage2=dict(oper="OPER002", n_eqp=3, conv=120, ppks=_convheavy_ppks),
    )
    _add_meta(
        id="TCB07_MULTI_CONVHEAVY_2STAGE", category="B. 다품종 재공충분", n_ppk=5,
        sim_end_minutes=440, conversion_minutes=120,
        desc="5제품×2공정, 장비 3대(EQP1={A,B}, EQP2={C,D}, EQP3={E} 전용 그룹). "
             "전환시간(120)=처리시간(40)의 3배 — 불필요한 전환의 대가가 매우 큼.",
        test_focus="전환 1회=carrier 3개 손실에 해당하는 상황에서 얼마나 전환을 절제하는지(가장 어려운 시나리오).",
        optimal=dict(production=40, conversions=4,
                     by_stage={"OPER001": {"production": 20, "conversions": 2},
                               "OPER002": {"production": 20, "conversions": 2}},
                     derivation="B2와 동일 구성(2/2/1 분배, 하한 전환=2/단계). "
                                "M1=A+B(160+160+120=440), M2=C+D(440), M3=E(160). "
                                "makespan=440=sim_end. 전량(20/20)+최소전환 동시 달성. 전용 장비그룹으로 "
                                "교차 배정을 차단해 실제 시뮬레이터 오라클 정책으로 이 스케줄이 그대로 "
                                "달성됨을 검증함(--verify)."),
    )

    # ---- C1: 1EQP 겸용, 안전재공 확보 후 1회 전환 ----
    gen_safety_switch(
        "TCB08_SAFETY_1EQP", n_eqp=1, conv=60,
        up=dict(oper="OPER001", st=20, wip=20, plan=20),
        down=dict(oper="OPER002", st=20, wip=0, plan=20),
    )
    _add_meta(
        id="TCB08_SAFETY_1EQP", category="C. 재공편중·안전재공", n_ppk=1,
        sim_end_minutes=860, conversion_minutes=60,
        desc="PPK001, 겸용 장비 1대. OPER001(재공20)→OPER002(재공0, OPER001 완료로만 유입). ST=20 동일.",
        test_focus="장비가 1대뿐이라 병렬 이득이 전혀 없는데도, 트리클 유입마다 전환을 반복하는지 "
                    "(min-progress형 진행률 균형 로직의 대표적 실패 케이스).",
        optimal=dict(production=40, conversions=1,
                     by_oper={"OPER001": 20, "OPER002": 20},
                     derivation="증명: (1)장비가 1대뿐이므로 두 공정을 동시에 처리 불가 → 전환을 여러 번 해도 "
                                "병렬 이득이 없고 전환시간만 순손실. 따라서 전환 횟수를 늘리는 어떤 정책도 "
                                "1회 전환 정책보다 나을 수 없다. (2)OPER002는 초기 재공 0이라 첫 배정은 "
                                "반드시 OPER001 → 전환 0회는 불가능(OPER002 계획 미달성). "
                                "→ 최소 전환=1이 하한이자 달성 가능: OPER001 20개(400min) 전부 처리 후 "
                                "전환(60min) → OPER002 20개(400min). 총 860min. 오라클 정책으로 시뮬레이터 "
                                "직접 검증(benchmark/tool_change_bench.py)."),
    )

    # ---- C2: 2EQP 겸용, 안전재공 확보 후 1대만 전환 ----
    eqps2 = gen_safety_switch(
        "TCB09_SAFETY_2EQP_BUFFER", n_eqp=2, conv=60,
        up=dict(oper="OPER001", st=15, wip=60, plan=60),
        down=dict(oper="OPER002", st=15, wip=0, plan=40),
    )
    _add_meta(
        id="TCB09_SAFETY_2EQP_BUFFER", category="C. 재공편중·안전재공", n_ppk=1,
        sim_end_minutes=780, conversion_minutes=60,
        desc="PPK001, 겸용 장비 2대. OPER001(재공60,계획60) → OPER002(재공0,계획40). ST=15 동일.",
        test_focus="OPER001 물량이 훨씬 많은(재공 편중) 상황에서, 두 장비를 모두 OPER001에 투입해 "
                    "안전재공을 빠르게 쌓은 뒤 '한 대만' OPER002로 넘기는 정책 vs. "
                    "트리클마다 전환하는 정책의 격차.",
        optimal=dict(production=100, conversions=1,
                     by_oper={"OPER001": 60, "OPER002": 40},
                     derivation="구성적 균형점(crossing-point) 설계: 두 장비가 t=0~120min 함께 OPER001 진행"
                                "(16개 완료, 안전재공 16 확보) → 1대가 전환(120~180min, 전환 중에도 나머지 "
                                "1대가 OPER001 계속 진행해 재공 +4 → 20) → 그 장비는 이후 OPER002 전담"
                                "(공급률=잔여 1대의 OPER001 산출률과 정확히 같아 재공이 마르지 않음, 40개 "
                                "처리에 600min, t=780 완료). 남은 1대는 계속 OPER001 진행해 나머지 44개 "
                                "(660min)를 처리, t=780 완료. 두 장비가 t=780에 동시 완결 → makespan 최소화"
                                "(crossing-point 표준 논증). 이론 하한(총 작업량/2대, 전환 무시)=750min에 "
                                "불과 30min(전환 1회의 절반) 위 — 사실상 최적."),
    )

    # ---- C3: 3EQP 겸용, 더 강한 편중 ----
    gen_safety_switch(
        "TCB10_SAFETY_3EQP_SKEWED", n_eqp=3, conv=40,
        up=dict(oper="OPER001", st=10, wip=90, plan=90),
        down=dict(oper="OPER002", st=10, wip=0, plan=30),
    )
    _add_meta(
        id="TCB10_SAFETY_3EQP_SKEWED", category="C. 재공편중·안전재공", n_ppk=1,
        sim_end_minutes=420, conversion_minutes=40,
        desc="PPK001, 겸용 장비 3대. OPER001(재공90,계획90) → OPER002(재공0,계획30). ST=10 동일. "
             "OPER001 편중이 C2보다 더 심함(재공 90 vs 계획 30).",
        test_focus="장비 3대 중 몇 대를, 언제 OPER002로 넘길지 스스로 판단해야 하는 가장 복잡한 케이스. "
                    "무계획 전환은 3대 모두가 번갈아 전환하며 전환비용을 크게 낭비하게 된다.",
        optimal=dict(production=120, conversions=1,
                     by_oper={"OPER001": 90, "OPER002": 30},
                     derivation="구성적 설계: 3대 모두 t=0~80min 함께 OPER001(24개 완료, 안전재공 확보) "
                                "→ 1대가 전환(80~120min, 나머지 2대는 OPER001 계속 진행해 재공 +8 → 32) "
                                "→ 전환 장비는 OPER002 전담(잔여 2대 공급률 > 자신의 소비율이라 재공이 계속 "
                                "쌓이기만 함, 절대 마르지 않음. 30개 처리 300min, t=420 완료). 남은 2대는 "
                                "OPER001 잔여 66개(330min)를 t=80+330=410에 완결. makespan=max(410,420)=420. "
                                "전환 1회로 두 계획(90+30) 모두 100% 달성 — crossing-point 근사 최적"
                                "(이론 하한 대비 여유 <2%)."),
    )


def main():
    build_all()
    with open(SUITE_ROOT / "tool_change_bench_meta.json", "w", encoding="utf-8") as f:
        json.dump(META, f, ensure_ascii=False, indent=2)
    print(f"TOOL_CHANGE_BENCH {len(META)}종 생성 완료 → {SUITE_ROOT}")
    for m in META:
        opt = m["optimal"]
        print(f"  {m['id']:<32} [{m['category']:<14}] "
              f"sim={m['sim_end_minutes']:<5} 최적생산={opt['production']:<4} 최적전환={opt['conversions']}")


if __name__ == "__main__":
    main()
