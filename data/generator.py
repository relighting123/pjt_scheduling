"""
data/generator.py – 샘플 JSON 데이터 생성 (Oracle 없이 개발·테스트용)

출력 경로:
  data/dataset/{FAC_ID}/train/{RULE_TIMEKEY}/input|output
  data/dataset/{FAC_ID}/test/{RULE_TIMEKEY}/input|output
  data/dataset/{FAC_ID}/infer/input|output
"""
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import (
    CONFIG,
    DATASET_DIR,
    DATASET_SPLITS,
    PERIOD_SPLITS,
    RULE_TIMEKEY_FMT,
    iter_rule_timekeys,
    normalize_rule_timekey,
    period_today,
    resolve_dataset_path,
    rule_timekey_now,
    rule_timekey_today,
    set_input_folder,
    validate_path_segment,
)


@dataclass
class GeneratorConfig:
    """랜덤 샘플 생성 파라미터"""
    n_products: int = 3
    n_eqps: int = 3
    n_opers: int = 2
    lots_per_oper: int = 3
    wf_qty: int = 25
    st_min: float = 60.0
    st_max: float = 180.0
    st_std: float = 20.0
    eligibility: float = 0.7          # 0=전용, 1=전체 가능
    plan_qty_min: int = 25
    plan_qty_max: int = 150
    plan_priority: int = 1
    train_period_count: int = 3         # train RULE_TIMEKEY 폴더 수
    test_period_count: int = 1          # test RULE_TIMEKEY 폴더 수
    split_qty: int = 3                  # PPK×MODEL wafer split (장)
    seed: Optional[int] = None

    def validate(self) -> None:
        if self.st_max < self.st_min:
            raise ValueError(f"st_max({self.st_max}) >= st_min({self.st_min}) 이어야 합니다.")
        if self.plan_qty_max < self.plan_qty_min:
            raise ValueError(
                f"plan_qty_max({self.plan_qty_max}) >= plan_qty_min({self.plan_qty_min}) 이어야 합니다."
            )
        if not 0.0 <= self.eligibility <= 1.0:
            raise ValueError(f"eligibility는 0~1 사이: {self.eligibility}")


DEFAULT_GENERATOR_CONFIG = GeneratorConfig()


def generator_config_from_dict(data: Optional[Dict[str, Any]]) -> GeneratorConfig:
    if not data:
        return GeneratorConfig()
    known = {f.name for f in GeneratorConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known and v is not None}
    cfg = GeneratorConfig(**filtered)
    cfg.validate()
    return cfg


def generator_config_to_dict(cfg: Optional[GeneratorConfig] = None) -> dict:
    return asdict(cfg or DEFAULT_GENERATOR_CONFIG)


def list_period_keys(count: int, start_key: Optional[str] = None) -> List[str]:
    """RULE_TIMEKEY count개 (일별, 시작 키 시·분·초 유지)"""
    if count < 1:
        raise ValueError("count는 1 이상")
    start = normalize_rule_timekey(start_key or rule_timekey_today())
    dt = datetime.strptime(start, RULE_TIMEKEY_FMT)
    h, m, s = dt.hour, dt.minute, dt.second
    keys = [start]
    for i in range(1, count):
        nxt = dt + timedelta(days=i)
        keys.append(nxt.replace(hour=h, minute=m, second=s, microsecond=0).strftime(RULE_TIMEKEY_FMT))
    return keys


def _discrete_row(
    eqp_id: str,
    lot_id: str,
    ppk: str,
    oper_id: str,
    proc_time: int,
    wf_qty: int = 25,
    eqp_model: str = "A",
    seq: Optional[int] = None,
    carrier_id: Optional[str] = None,
) -> dict:
    row = {
        "EQP_ID": eqp_id,
        "LOT_ID": lot_id,
        "PLAN_PROD_KEY": ppk,
        "OPER_ID": oper_id,
        "ST": proc_time,
        "EQP_MODEL": eqp_model,
        "WF_QTY": wf_qty,
    }
    if seq is not None:
        row["SEQ"] = seq
    if carrier_id:
        row["CARRIER_ID"] = carrier_id
    return row


def _abstract_row(ppk: str, oper_id: str, eqp_model: str, st: int) -> dict:
    return {
        "PLAN_PROD_KEY": ppk,
        "OPER_ID": oper_id,
        "EQP_MODEL": eqp_model,
        "ST": st,
    }


def build_abstract_arrange(
    discrete_arrange: List[dict],
    flow: Optional[List[dict]] = None,
) -> List[dict]:
    """discrete_arrange → PPK×OPER×MODEL feasible routes (OPER_ID·ST 집계)."""
    del flow
    route_st: Dict[Tuple[str, str, str], List[int]] = {}
    for r in discrete_arrange:
        oper = r.get("OPER_ID")
        if not oper:
            continue
        ppk = r["PLAN_PROD_KEY"]
        model = str(r.get("EQP_MODEL") or "A")
        st = int(r.get("ST") or 60)
        route_st.setdefault((ppk, oper, model), []).append(st)

    return [
        _abstract_row(ppk, oper, model, int(sum(sts) / len(sts)))
        for (ppk, oper, model), sts in sorted(route_st.items())
    ]


def _split_row(ppk: str, oper_id: str, eqp_model: str, split_qty: int) -> dict:
    return {
        "PLAN_PROD_KEY": ppk,
        "OPER_ID": oper_id,
        "EQP_MODEL": eqp_model,
        "SPLIT_QTY": split_qty,
    }


def build_split_rules(
    flow: List[dict],
    split_qty: int = 3,
    models: Tuple[str, ...] = ("A", "B", "C", "D", "E"),
) -> List[dict]:
    """flow의 PPK × OPER × EQP MODEL 별 SPLIT_QTY 규칙 생성"""
    ppk_opers = sorted({(r["PLAN_PROD_KEY"], r["OPER_ID"]) for r in flow})
    return [
        _split_row(ppk, oper_id, model, split_qty)
        for ppk, oper_id in ppk_opers
        for model in models
    ]


def build_lot_master_from_discrete(discrete_arrange: List[dict]) -> List[dict]:
    rows = []
    seen = set()
    for r in discrete_arrange:
        lid = r["LOT_ID"]
        if lid in seen:
            continue
        seen.add(lid)
        ppk = r["PLAN_PROD_KEY"]
        rows.append({
            "LOT_ID": lid,
            "LOT_CD": f"LC{ppk[-3:]}",
            "TEMP": "T650" if int(ppk[-3:]) % 2 else "T700",
        })
    return rows


def build_batch_info_from_discrete(discrete_arrange: List[dict]) -> List[dict]:
    """(PPK, OPER)별 LOT_CD/TEMP — conversion 레시피."""
    pairs: set = set()
    for r in discrete_arrange:
        oper_id = r.get("OPER_ID")
        if not oper_id:
            continue
        pairs.add((r["PLAN_PROD_KEY"], oper_id))
    rows = []
    for ppk, oper_id in sorted(pairs):
        suffix = int(ppk[-3:]) if ppk[-3:].isdigit() else sum(ord(c) for c in ppk)
        rows.append({
            "PLAN_PROD_KEY": ppk,
            "OPER_ID": oper_id,
            "LOT_CD": f"LC{ppk[-3:]}",
            "TEMP": "T650" if suffix % 2 == 0 else "T700",
        })
    return rows


def build_tool_capacity_from_lots(
    lot_master: List[dict],
    models: Tuple[str, ...] = ("A", "B", "C", "D", "E"),
    max_tool: int = 2,
) -> List[dict]:
    lot_cds = sorted({r["LOT_CD"] for r in lot_master})
    return [
        {"LOT_CD": lc, "EQP_MODEL": model, "MAX_TOOL": max_tool}
        for lc in lot_cds
        for model in models
    ]


def write_json_bundle(
    output_dir: Path,
    discrete_arrange: List[dict],
    plan: List[dict],
    flow: List[dict],
    split: Optional[List[dict]] = None,
    lot_master: Optional[List[dict]] = None,
    batch_info: Optional[List[dict]] = None,
    tool_capacity: Optional[List[dict]] = None,
    abstract_arrange: Optional[List[dict]] = None,
    eqp_initial_state: Optional[List[dict]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_data = split if split is not None else build_split_rules(flow)
    abstract_data = abstract_arrange if abstract_arrange is not None else build_abstract_arrange(
        discrete_arrange,
    )
    lot_master_data = lot_master if lot_master is not None else build_lot_master_from_discrete(
        discrete_arrange,
    )
    batch_info_data = batch_info if batch_info is not None else build_batch_info_from_discrete(
        discrete_arrange,
    )
    tool_capacity_data = tool_capacity if tool_capacity is not None else build_tool_capacity_from_lots(
        lot_master_data,
    )
    files = [
        (CONFIG.path.discrete_arrange_file, discrete_arrange),
        (CONFIG.path.abstract_arrange_file, abstract_data),
        (CONFIG.path.plan_file, plan),
        (CONFIG.path.flow_file, flow),
        (CONFIG.path.split_file, split_data),
        (CONFIG.path.lot_master_file, lot_master_data),
        (CONFIG.path.batch_info_file, batch_info_data),
        (CONFIG.path.tool_capacity_file, tool_capacity_data),
    ]
    if eqp_initial_state is not None:
        files.append((CONFIG.path.eqp_initial_state_file, eqp_initial_state))
    for filename, data in files:
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_split_dirs(fac_id: str, split: str, snapshot: Optional[str] = None) -> Tuple[Path, Path]:
    """input/output 디렉터리 생성 후 경로 반환"""
    inp, out = resolve_dataset_path(fac_id, split, snapshot)
    inp.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    return inp, out


def _build_default_sample() -> Tuple[List[dict], List[dict], List[dict]]:
    oper1, oper2 = "OPER001", "OPER002"
    discrete: List[dict] = []
    lot_specs = [
        ("LOT001", "CAR001", "PPK001", oper1, 1, 120, ("EQP001",)),
        ("LOT002", "CAR002", "PPK002", oper1, 1, 90, ("EQP001",)),
        ("LOT003", "CAR003", "PPK001", oper1, 1, 120, ("EQP001",)),
        ("LOT004", "CAR004", "PPK003", oper1, 1, 105, ("EQP001", "EQP002")),
        ("LOT005", "CAR005", "PPK001", oper1, 1, 120, ("EQP002",)),
        ("LOT006", "CAR006", "PPK002", oper1, 1, 120, ("EQP002",)),
        ("LOT007", "CAR007", "PPK002", oper2, 2, 180, ("EQP003",)),
        ("LOT008", "CAR008", "PPK003", oper2, 2, 120, ("EQP003",)),
        ("LOT009", "CAR009", "PPK001", oper2, 2, 180, ("EQP003",)),
    ]
    for lot_id, carrier, ppk, oper_id, seq, st, eqps in lot_specs:
        for eqp_id in eqps:
            discrete.append(_discrete_row(
                eqp_id, lot_id, ppk, oper_id, st,
                carrier_id=carrier, seq=seq,
            ))
    plan = [
        {"PLAN_PROD_KEY": "PPK001", "OPER_ID": "OPER001",
         "D0_PLAN_QTY": 75, "D1_PLAN_QTY": 100, "PLAN_PRIORITY": 1},
        {"PLAN_PROD_KEY": "PPK002", "OPER_ID": "OPER001",
         "D0_PLAN_QTY": 50, "D1_PLAN_QTY": 75, "PLAN_PRIORITY": 1},
        {"PLAN_PROD_KEY": "PPK003", "OPER_ID": "OPER001",
         "D0_PLAN_QTY": 25, "D1_PLAN_QTY": 50, "PLAN_PRIORITY": 1},
        {"PLAN_PROD_KEY": "PPK001", "OPER_ID": "OPER002",
         "D0_PLAN_QTY": 50, "D1_PLAN_QTY": 75, "PLAN_PRIORITY": 1},
        {"PLAN_PROD_KEY": "PPK002", "OPER_ID": "OPER002",
         "D0_PLAN_QTY": 25, "D1_PLAN_QTY": 50, "PLAN_PRIORITY": 1},
        {"PLAN_PROD_KEY": "PPK003", "OPER_ID": "OPER002",
         "D0_PLAN_QTY": 25, "D1_PLAN_QTY": 50, "PLAN_PRIORITY": 1},
    ]
    flow = [
        {"PLAN_PROD_KEY": "PPK001", "SEQ_ID": 1, "OPER_ID": "OPER001"},
        {"PLAN_PROD_KEY": "PPK001", "SEQ_ID": 2, "OPER_ID": "OPER002"},
        {"PLAN_PROD_KEY": "PPK002", "SEQ_ID": 1, "OPER_ID": "OPER001"},
        {"PLAN_PROD_KEY": "PPK002", "SEQ_ID": 2, "OPER_ID": "OPER002"},
        {"PLAN_PROD_KEY": "PPK003", "SEQ_ID": 1, "OPER_ID": "OPER001"},
        {"PLAN_PROD_KEY": "PPK003", "SEQ_ID": 2, "OPER_ID": "OPER002"},
    ]
    return discrete, plan, flow


def _build_single_heavy_wip_sample() -> Tuple[List[dict], List[dict], List[dict]]:
    ppk = "PPK001"
    wf_qty = 25
    n_lots = 8
    proc1 = 120
    proc2 = 90
    st_step1 = 120
    st_step2 = 60
    oper1_eqps = ("EQP001", "EQP002")
    all_eqps = ("EQP001", "EQP002", "EQP003")

    discrete: List[dict] = []

    for i in range(n_lots):
        lot_id = f"LOT{i + 1:03d}"
        eqp = "EQP001" if i < n_lots // 2 else "EQP002"
        for eqp_id in oper1_eqps:
            discrete.append(_discrete_row(
                eqp_id, lot_id, ppk, "OPER001", proc1, wf_qty,
                carrier_id=f"CAR{i + 1:03d}", seq=1,
            ))

    for i in range(n_lots):
        lot_id = f"LOT{101 + i}"
        for eqp_id in all_eqps:
            discrete.append(_discrete_row(
                eqp_id, lot_id, ppk, "OPER002", proc2, wf_qty,
                carrier_id=f"CAR{101 + i}", seq=2,
            ))

    total = n_lots * wf_qty
    plan = [
        {"PLAN_PROD_KEY": ppk, "OPER_ID": "OPER001",
         "D0_PLAN_QTY": total, "D1_PLAN_QTY": total + 50, "PLAN_PRIORITY": 1},
        {"PLAN_PROD_KEY": ppk, "OPER_ID": "OPER002",
         "D0_PLAN_QTY": total, "D1_PLAN_QTY": total + 50, "PLAN_PRIORITY": 1},
    ]
    flow = [
        {"PLAN_PROD_KEY": ppk, "SEQ_ID": 1, "OPER_ID": "OPER001"},
        {"PLAN_PROD_KEY": ppk, "SEQ_ID": 2, "OPER_ID": "OPER002"},
    ]
    return discrete, plan, flow


def _sample_st(rng: random.Random, cfg: GeneratorConfig) -> int:
    if cfg.st_std <= 0:
        return int(round((cfg.st_min + cfg.st_max) / 2))
    mean = (cfg.st_min + cfg.st_max) / 2
    val = rng.gauss(mean, cfg.st_std)
    return int(max(cfg.st_min, min(cfg.st_max, round(val))))


def _is_eqp_eligible(
    rng: random.Random,
    prod_idx: int,
    eqp_idx: int,
    n_eqps: int,
    eligibility: float,
) -> bool:
    """
    제품×설비 eligibility
    1.0 → 모든 조합, 0.0 → 제품별 전용 1대, 그 외 → 전용 + 확률 eligibility
    """
    if eligibility >= 1.0:
        return True
    dedicated = prod_idx % n_eqps
    if eligibility <= 0.0:
        return eqp_idx == dedicated
    if eqp_idx == dedicated:
        return True
    return rng.random() < eligibility


def _build_random_sample(
    cfg: Optional[GeneratorConfig] = None,
) -> Tuple[List[dict], List[dict], List[dict]]:
    cfg = cfg or DEFAULT_GENERATOR_CONFIG
    cfg.validate()
    rng = random.Random(cfg.seed)

    products = [f"PPK{i + 1:03d}" for i in range(cfg.n_products)]
    eqps = [f"EQP{i + 1:03d}" for i in range(cfg.n_eqps)]
    opers = [f"OPER{i + 1:03d}" for i in range(cfg.n_opers)]
    eqp_models = ["A", "B", "C", "D", "E"]

    flow: List[dict] = []
    for ppk in products:
        for seq in range(1, cfg.n_opers + 1):
            flow.append({
                "PLAN_PROD_KEY": ppk,
                "SEQ_ID": seq,
                "OPER_ID": opers[seq - 1],
            })

    plan: List[dict] = []
    for ppk in products:
        for oper in opers:
            d0 = rng.randint(cfg.plan_qty_min, cfg.plan_qty_max)
            d1 = rng.randint(cfg.plan_qty_min, cfg.plan_qty_max)
            if d1 < d0:
                d0, d1 = d1, d0
            plan.append({
                "PLAN_PROD_KEY": ppk,
                "OPER_ID": oper,
                "D0_PLAN_QTY": d0,
                "D1_PLAN_QTY": d1,
                "PLAN_PRIORITY": cfg.plan_priority,
            })

    discrete: List[dict] = []
    lot_counter = 0

    for p_idx, ppk in enumerate(products):
        for o_idx in range(cfg.n_opers):
            seq = o_idx + 1
            oper_id = opers[o_idx]
            for _ in range(cfg.lots_per_oper):
                lot_counter += 1
                lot_id = f"LOT{lot_counter:03d}"
                carrier_id = f"CAR{lot_counter:03d}"

                eligible_eqps = [
                    eqps[e_idx] for e_idx in range(cfg.n_eqps)
                    if _is_eqp_eligible(rng, p_idx, e_idx, cfg.n_eqps, cfg.eligibility)
                ]
                if not eligible_eqps:
                    eligible_eqps = [eqps[p_idx % cfg.n_eqps]]

                eqp_model = eqp_models[p_idx % len(eqp_models)]
                for e_idx, eqp in enumerate(eqps):
                    if _is_eqp_eligible(rng, p_idx, e_idx, cfg.n_eqps, cfg.eligibility):
                        st = _sample_st(rng, cfg)
                        discrete.append(_discrete_row(
                            eqp, lot_id, ppk, oper_id, st, cfg.wf_qty, eqp_model,
                            seq=seq, carrier_id=carrier_id,
                        ))

    return discrete, plan, flow


def _build_pacing_steady_sample() -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Pacing·PPK 전환 검증용 미니 문제.

    - PPK001/PPK002, 2공정(OPER001→OPER002), EQP 2대(EQP001/002)
    - 계획: 각 PPK×OPER 100매 (soft_cutoff 1320분 → 이상 기울기 ≈ 100/1320 매/분)
    - PPK001: OPER001 재공 4 LOT (100매) — OPER002는 유입만 가능
    - PPK002: OPER001 재공 2 LOT (50매) — OPER1 부족 → 전환·앞공정 우선 학습
    - OPER001만 초기 schedule; OPER002는 abstract 유입 → 계단/공백이 잘 보임
    """
    wf_qty = 25
    proc = 90
    plan_qty = 100
    ppk_a, ppk_b = "PPK001", "PPK002"

    flow = [
        {"PLAN_PROD_KEY": ppk_a, "SEQ_ID": 1, "OPER_ID": "OPER001"},
        {"PLAN_PROD_KEY": ppk_a, "SEQ_ID": 2, "OPER_ID": "OPER002"},
        {"PLAN_PROD_KEY": ppk_b, "SEQ_ID": 1, "OPER_ID": "OPER001"},
        {"PLAN_PROD_KEY": ppk_b, "SEQ_ID": 2, "OPER_ID": "OPER002"},
    ]
    plan = [
        {"PLAN_PROD_KEY": ppk_a, "OPER_ID": "OPER001",
         "D0_PLAN_QTY": plan_qty, "D1_PLAN_QTY": plan_qty, "PLAN_PRIORITY": 1},
        {"PLAN_PROD_KEY": ppk_a, "OPER_ID": "OPER002",
         "D0_PLAN_QTY": plan_qty, "D1_PLAN_QTY": plan_qty, "PLAN_PRIORITY": 1},
        {"PLAN_PROD_KEY": ppk_b, "OPER_ID": "OPER001",
         "D0_PLAN_QTY": plan_qty, "D1_PLAN_QTY": plan_qty, "PLAN_PRIORITY": 1},
        {"PLAN_PROD_KEY": ppk_b, "OPER_ID": "OPER002",
         "D0_PLAN_QTY": plan_qty, "D1_PLAN_QTY": plan_qty, "PLAN_PRIORITY": 1},
    ]

    discrete: List[dict] = []

    for i in range(4):
        lot_id = f"LOT{i + 1:03d}"
        for eqp_id in ("EQP001", "EQP002"):
            discrete.append(_discrete_row(
                eqp_id, lot_id, ppk_a, "OPER001", proc, wf_qty,
                carrier_id=f"CAR{i + 1:03d}", seq=1,
            ))

    for i in range(2):
        lot_id = f"LOT{101 + i}"
        for eqp_id in ("EQP001", "EQP002"):
            discrete.append(_discrete_row(
                eqp_id, lot_id, ppk_b, "OPER001", proc, wf_qty,
                carrier_id=f"CAR{101 + i}", seq=1,
            ))

    return discrete, plan, flow


def build_pacing_steady_abstract_arrange() -> List[dict]:
    """OPER002 유입 라우트 — 초기 discrete에 OPER2 LOT 없어도 abstract 유입 가능."""
    st = 90
    rows = []
    for ppk in ("PPK001", "PPK002"):
        for oper in ("OPER001", "OPER002"):
            rows.append(_abstract_row(ppk, oper, "A", st))
    return rows


SampleBuilder = Callable[[], Tuple[List[dict], List[dict], List[dict]]]

SAMPLE_SCENARIOS: Dict[str, dict] = {
    "default": {
        "name": "기본 (3제품)",
        "description": "PPK 3종, LOT 9개, 혼합 priority",
        "build": _build_default_sample,
        "configurable": False,
    },
    "single_heavy_wip": {
        "name": "단일제품 ST½ 재공다량",
        "description": "PPK001 단일 제품, OPER002 ST 90분·전 설비, OPER002 START 간격 OPER001의 1/2",
        "build": _build_single_heavy_wip_sample,
        "configurable": False,
    },
    "pacing_steady": {
        "name": "Pacing 검증 (2제품)",
        "description": (
            "PPK001/002 2공정·계획 100매. PPK002 OPER1 재공 50매 부족. "
            "누적 생산 점선 대비 기울기·PPK 전환 테스트용."
        ),
        "build": _build_pacing_steady_sample,
        "configurable": False,
        "abstract_arrange": build_pacing_steady_abstract_arrange,
    },
    "random": {
        "name": "랜덤 (파라미터)",
        "description": "ST·eligibility·계획량·규모·train/test 크기를 UI/설정으로 생성",
        "build": lambda: _build_random_sample(DEFAULT_GENERATOR_CONFIG),
        "configurable": True,
    },
}

try:
    from data.pacing_scenarios import TAKT_SCENARIOS  # noqa: E402
    SAMPLE_SCENARIOS.update(TAKT_SCENARIOS)
except ImportError:
    pass

try:
    from data.conversion_scenarios import CONV_SCENARIOS  # noqa: E402
    SAMPLE_SCENARIOS.update(CONV_SCENARIOS)
except ImportError:
    pass


def _build_dataset_bundle(
    scenario: str,
    gen_config: Optional[GeneratorConfig] = None,
) -> Tuple[List[dict], List[dict], List[dict]]:
    if scenario not in SAMPLE_SCENARIOS:
        raise ValueError(f"알 수 없는 시나리오: {scenario}")
    if scenario == "random":
        return _build_random_sample(gen_config or DEFAULT_GENERATOR_CONFIG)
    return SAMPLE_SCENARIOS[scenario]["build"]()


def list_sample_scenarios() -> List[dict]:
    return [
        {
            "id": sid,
            "name": meta["name"],
            "description": meta["description"],
            "configurable": meta.get("configurable", False),
        }
        for sid, meta in SAMPLE_SCENARIOS.items()
    ]


def generate_sample_data(
    scenario: str = "default",
    fac_id: str = "FAC001",
    split: str = "train",
    snapshot: Optional[str] = None,
    period: Optional[str] = None,
    output_dir: Optional[Path] = None,
    gen_config: Optional[GeneratorConfig] = None,
    *,
    period_seed_offset: int = 0,
) -> Path:
    """
    시나리오별 샘플 JSON을 dataset 경로에 생성
    train/test → {RULE_TIMEKEY}/input (기본: train=현재시각, test=오늘 07:00:00)
    infer → infer/input (기간 하위 폴더 없음)
    """
    if scenario not in SAMPLE_SCENARIOS:
        raise ValueError(
            f"알 수 없는 시나리오: {scenario}. "
            f"사용 가능: {', '.join(SAMPLE_SCENARIOS)}"
        )
    fac_id = validate_path_segment(fac_id, "FAC_ID")
    split = validate_path_segment(split, "split")
    if split not in DATASET_SPLITS:
        raise ValueError(f"split은 {DATASET_SPLITS} 중 하나: {split!r}")

    per = period or snapshot
    if output_dir is None:
        if split in PERIOD_SPLITS and not per:
            per = rule_timekey_today() if split == "test" else rule_timekey_now()
        inp, _ = ensure_split_dirs(fac_id, split, per)
        output_dir = inp
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    cfg = gen_config
    if scenario == "random" and cfg and cfg.seed is not None and period_seed_offset:
        cfg = GeneratorConfig(**{
            **asdict(cfg),
            "seed": cfg.seed + period_seed_offset,
        })

    discrete, plan, flow = _build_dataset_bundle(scenario, cfg)
    split_rules = build_split_rules(
        flow,
        split_qty=(cfg.split_qty if cfg and scenario == "random" else 3),
    )
    meta = SAMPLE_SCENARIOS.get(scenario, {})
    abstract_fn = meta.get("abstract_arrange")
    abstract_data = abstract_fn() if callable(abstract_fn) else None
    bundle_kwargs: dict = {"abstract_arrange": abstract_data}
    full_build = meta.get("full_build")
    if callable(full_build):
        _d, _p, _f, lot_master, abstract_full, eqp_init, tool_cap = full_build()
        bundle_kwargs.update({
            "lot_master": lot_master,
            "abstract_arrange": abstract_full,
            "eqp_initial_state": eqp_init,
            "tool_capacity": tool_cap,
        })
    write_json_bundle(
        output_dir, discrete, plan, flow, split_rules,
        **bundle_kwargs,
    )
    print(f"[generator] 샘플 생성 ({scenario}) → {output_dir}")
    return output_dir


def generate_sample_period_range(
    scenario: str,
    fac_id: str,
    from_timekey: Optional[str] = None,
    to_timekey: Optional[str] = None,
    split: str = "train",
    gen_config: Optional[GeneratorConfig] = None,
    *,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    period_count: Optional[int] = None,
) -> List[Path]:
    """RULE_TIMEKEY 구간 또는 count 기준 train|test 샘플 JSON 생성"""
    cfg = gen_config or DEFAULT_GENERATOR_CONFIG
    paths: List[Path] = []

    if period_count is not None and period_count > 0:
        keys = list_period_keys(period_count)
    else:
        start_key = from_timekey or from_date
        end_key = to_timekey or to_date
        if not start_key or not end_key:
            raise ValueError(
                "from_timekey/to_timekey, from_date/to_date, 또는 period_count를 지정하세요."
            )
        keys = list(iter_rule_timekeys(start_key, end_key))

    for idx, per in enumerate(keys):
        paths.append(generate_sample_data(
            scenario=scenario,
            fac_id=fac_id,
            split=split,
            period=per,
            gen_config=cfg,
            period_seed_offset=idx,
        ))
    print(f"[generator] {split} → {len(paths)}개 RULE_TIMEKEY 폴더 생성")
    return paths


def bootstrap_facility_datasets(
    fac_id: str = "FAC001",
    scenario: str = "default",
    gen_config: Optional[GeneratorConfig] = None,
) -> dict:
    """
    FAC_ID 기준 train/test(period_count) + infer 골격 및 샘플 JSON 생성
    """
    cfg = gen_config or DEFAULT_GENERATOR_CONFIG
    cfg.validate()
    fac_id = validate_path_segment(fac_id, "FAC_ID")

    train_keys = list_period_keys(cfg.train_period_count)
    test_keys = list_period_keys(cfg.test_period_count)

    paths = {}
    split_specs = [
        ("train", train_keys),
        ("test", test_keys),
        ("infer", [None]),
    ]
    for split, period_list in split_specs:
        split_paths = []
        for idx, per in enumerate(period_list):
            if split in PERIOD_SPLITS:
                inp, out = ensure_split_dirs(fac_id, split, per)
            else:
                inp, out = ensure_split_dirs(fac_id, split, None)
            generate_sample_data(
                scenario=scenario,
                fac_id=fac_id,
                split=split,
                period=per,
                output_dir=inp,
                gen_config=cfg,
                period_seed_offset=idx,
            )
            split_paths.append({"input": str(inp), "output": str(out), "period": per})
        paths[split] = split_paths if len(split_paths) > 1 else split_paths[0]

    return {
        "fac_id": fac_id,
        "train_periods": train_keys,
        "test_periods": test_keys,
        "train_snapshot": train_keys[-1] if train_keys else rule_timekey_now(),
        "test_period": test_keys[-1] if test_keys else rule_timekey_today(),
        "paths": paths,
        "generator_config": generator_config_to_dict(cfg),
    }


def _last_split_input(paths: dict, split: str) -> Path:
    entry = paths[split]
    if isinstance(entry, list):
        return Path(entry[-1]["input"])
    return Path(entry["input"])


def _apply_sample_input_folder(fac_id: str, split: str, path: Path) -> None:
    if split in PERIOD_SPLITS:
        set_input_folder(f"{fac_id}/{split}/{path.parent.name}")
    else:
        set_input_folder(f"{fac_id}/{split}")


def generate_sample(
    scenario: str = "default",
    fac_id: str = "FAC001",
    split: str = "train",
    *,
    bootstrap: bool = False,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    use_period_count: bool = False,
    gen_config: Optional[GeneratorConfig] = None,
    update_input_folder: bool = True,
    verbose: bool = False,
) -> dict:
    """
    샘플 JSON 생성 통합 진입점 (CLI · API 공용).

    Returns:
        path: 마지막 생성 input 디렉터리
        bootstrap_info: bootstrap 시 bootstrap_facility_datasets 반환값
        generator_config: 적용된 GeneratorConfig dict
    """
    if scenario not in SAMPLE_SCENARIOS:
        raise ValueError(
            f"알 수 없는 시나리오: {scenario}. "
            f"사용 가능: {', '.join(SAMPLE_SCENARIOS)}"
        )

    cfg = gen_config or DEFAULT_GENERATOR_CONFIG
    if verbose:
        print("=" * 60)
        print(f"[generator] 샘플 데이터 생성 (시나리오: {scenario}, FAC: {fac_id})")

    if bootstrap:
        info = bootstrap_facility_datasets(fac_id=fac_id, scenario=scenario, gen_config=cfg)
        snap = info["train_snapshot"]
        path = _last_split_input(info["paths"], "train")
        if update_input_folder:
            set_input_folder(f"{fac_id}/train/{snap}")
        if verbose:
            print(f"  train periods: {info.get('train_periods', [snap])}")
            print(f"  test periods:  {info.get('test_periods', [info.get('test_period')])}")
        return {
            "path": path,
            "bootstrap_info": info,
            "generator_config": info.get("generator_config"),
        }

    if use_period_count:
        count = (
            cfg.train_period_count if split == "train"
            else cfg.test_period_count if split == "test"
            else 1
        )
        paths = generate_sample_period_range(
            scenario=scenario,
            fac_id=fac_id,
            split=split,
            gen_config=cfg,
            period_count=count,
        )
        path = paths[-1]
        if update_input_folder:
            _apply_sample_input_folder(fac_id, split, path)
        return {
            "path": path,
            "bootstrap_info": None,
            "generator_config": generator_config_to_dict(cfg),
        }

    if from_date and to_date:
        paths = generate_sample_period_range(
            scenario=scenario,
            fac_id=fac_id,
            from_date=from_date,
            to_date=to_date,
            split=split,
            gen_config=cfg,
        )
        path = paths[-1]
        if update_input_folder:
            _apply_sample_input_folder(fac_id, split, path)
        return {
            "path": path,
            "bootstrap_info": None,
            "generator_config": generator_config_to_dict(cfg),
        }

    if from_date or to_date:
        raise ValueError("from_date와 to_date를 함께 지정하세요.")

    path = generate_sample_data(
        scenario=scenario,
        fac_id=fac_id,
        split=split,
        gen_config=cfg,
    )
    if update_input_folder:
        _apply_sample_input_folder(fac_id, split, path)
    return {
        "path": path,
        "bootstrap_info": None,
        "generator_config": generator_config_to_dict(cfg),
    }
