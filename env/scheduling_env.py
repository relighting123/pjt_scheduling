"""

env/scheduling_env.py – Gymnasium 커스텀 환경

SchedulingSimulator를 감싸 RL 표준 인터페이스(reset/step)를 제공합니다.

StableBaselines3 MaskablePPO와 호환됩니다.

"""

from typing import List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import CONFIG
from simulation.simulator import SchedulingSimulator
from simulation.decision_log import build_step_decision_entry

_OBS_GLOBAL_DIM = 5
_OBS_EQP_LOCAL_DIM = 2
_OBS_FIXED_DIM = _OBS_GLOBAL_DIM + _OBS_EQP_LOCAL_DIM


def obs_dim_components() -> dict:
    """obs_dim 구성 요소 (config 기준)."""
    O = CONFIG.env.max_oper_count
    P = CONFIG.env.max_prod_count
    K = CONFIG.env.max_model_count
    F = SchedulingSimulator.BUCKET_FEATURES
    bucket = O * P * K * F
    return {
        "O": O,
        "P": P,
        "K": K,
        "F": F,
        "global": _OBS_GLOBAL_DIM,
        "bucket": bucket,
        "eqp_local": _OBS_EQP_LOCAL_DIM,
        "total": _OBS_FIXED_DIM + bucket,
    }


def compute_obs_dim() -> int:
    """Global(5) + Bucket(O×P×K×F) + current EQP(2)"""
    return obs_dim_components()["total"]


def _opk_product_from_obs_dim(dim: int) -> Optional[int]:
    """obs_dim에서 O×P×K 곱을 역산. 고정 7차원(Global+EQP) 제외."""
    inner = dim - _OBS_FIXED_DIM
    F = SchedulingSimulator.BUCKET_FEATURES
    if inner < 0 or inner % F != 0:
        return None
    return inner // F


def _factor_opk_triples(product: int, *, limit: int = 6) -> List[Tuple[int, int, int]]:
    """O×P×K=product 인 (O,P,K) 후보 (O≤P≤K)."""
    triples: List[Tuple[int, int, int]] = []
    for o in range(1, product + 1):
        if product % o != 0:
            continue
        rest = product // o
        for p in range(o, rest + 1):
            if rest % p != 0:
                continue
            k = rest // p
            if k >= p:
                triples.append((o, p, k))
    return triples[:limit]


def _format_id_list(items: list, *, label: str, max_items: int = 8) -> str:
    if not items:
        return f"  {label}: (없음)"
    shown = list(items[:max_items])
    suffix = f" 외 {len(items) - max_items}개" if len(items) > max_items else ""
    return f"  {label} {len(items)}개{suffix}: {shown}"


def _describe_obs_dim_side(
    dim: int,
    *,
    title: str,
    use_config_axes: bool = False,
) -> List[str]:
    """단일 obs_dim 측면 설명."""
    F = SchedulingSimulator.BUCKET_FEATURES
    lines = [f"[{title}]", f"  obs_dim = {dim}"]

    opk = _opk_product_from_obs_dim(dim)
    if opk is None:
        lines.append(
            f"  ※ {dim}은 표준 공식({_OBS_GLOBAL_DIM}+O×P×K×{F}+{_OBS_EQP_LOCAL_DIM})과 맞지 않습니다."
        )
        return lines

    bucket = opk * F
    if use_config_axes:
        comp = obs_dim_components()
        O, P, K = comp["O"], comp["P"], comp["K"]
        lines.append(
            f"  구성: Global({_OBS_GLOBAL_DIM}) + Bucket({bucket}={O}×{P}×{K}×{F}) "
            f"+ EQP({_OBS_EQP_LOCAL_DIM})"
        )
        lines.append(
            f"  config: max_oper_count(O)={O}, max_prod_count(P)={P}, "
            f"max_model_count(K)={K}, bucket_features(F)={F}"
        )
    else:
        lines.append(
            f"  구성: Global({_OBS_GLOBAL_DIM}) + Bucket({bucket}=O×P×K×{F}) "
            f"+ EQP({_OBS_EQP_LOCAL_DIM})"
        )
        triples = _factor_opk_triples(opk)
        if triples:
            hints = ", ".join(f"O={o},P={p},K={k}" for o, p, k in triples)
            lines.append(f"  추정 config (O×P×K={opk}): {hints}")
        else:
            lines.append(f"  추정 O×P×K = {opk}")
    return lines


def env_obs_data_context(env_data: Optional[dict]) -> List[str]:
    """입력 데이터의 공정·제품·모델 실측 (config와 비교용)."""
    if not env_data:
        return []

    comp = obs_dim_components()
    O_cfg, P_cfg, K_cfg = comp["O"], comp["P"], comp["K"]
    oper_ids = list(env_data.get("oper_ids", []))
    prod_keys = list(env_data.get("prod_keys", []))
    eqp_models = list(env_data.get("eqp_models", []))
    eqp_ids = list(env_data.get("eqp_ids", []))

    lines = ["[입력 데이터 실측]"]
    lines.append(_format_id_list(oper_ids, label="OPER"))
    lines.append(_format_id_list(prod_keys, label="제품(PPK)"))
    lines.append(_format_id_list(eqp_models, label="장비모델"))
    lines.append(_format_id_list(eqp_ids, label="호기(EQP)"))

    warnings: List[str] = []
    if len(oper_ids) > O_cfg:
        warnings.append(
            f"OPER {len(oper_ids)}개 > config O={O_cfg} → 상위 {O_cfg}개만 obs/action에 반영"
        )
    if len(prod_keys) > P_cfg:
        warnings.append(
            f"제품 {len(prod_keys)}개 > config P={P_cfg} → 상위 {P_cfg}개만 반영"
        )
    if len(eqp_models) > K_cfg:
        warnings.append(
            f"장비모델 {len(eqp_models)}개 > config K={K_cfg} → 상위 {K_cfg}개만 반영"
        )
    if warnings:
        lines.append("  ※ 데이터 초과 (config 상한):")
        for w in warnings:
            lines.append(f"    - {w}")
    return lines


def format_obs_dim_mismatch(
    expected_dim: int,
    actual_dim: int,
    *,
    env_data: Optional[dict] = None,
    source: str = "",
    model_files: Optional[List[str]] = None,
) -> str:
    """obs_dim 불일치 시 config·데이터·모델 정보를 포함한 진단 메시지."""
    header = "관측 차원(obs_dim) 불일치"
    if source:
        header += f" ({source})"

    lines = [header, ""]
    lines.extend(
        _describe_obs_dim_side(expected_dim, title="현재 환경 (config 기준)", use_config_axes=True)
    )
    lines.append("")
    lines.extend(
        _describe_obs_dim_side(actual_dim, title="불일치 측 (모델/실측)", use_config_axes=False)
    )
    if env_data:
        lines.append("")
        lines.extend(env_obs_data_context(env_data))
    if model_files:
        lines.append("")
        lines.append("[확인된 모델 파일]")
        for name in model_files:
            lines.append(f"  - {name}")
    lines.append("")
    lines.append("[조치]")
    lines.append(
        "  1. config.py의 max_oper_count(O), max_prod_count(P), max_model_count(K)를 "
        "학습 시와 동일하게 맞추거나"
    )
    lines.append("  2. 현재 config로 python main.py train (또는 UI 학습) 재실행")
    lines.append(
        f"  ※ 공식: obs_dim = {_OBS_GLOBAL_DIM} + O×P×K×{SchedulingSimulator.BUCKET_FEATURES} "
        f"+ {_OBS_EQP_LOCAL_DIM}"
    )
    return "\n".join(lines)


def validate_obs_shape(
    obs: np.ndarray,
    expected_dim: Optional[int] = None,
    *,
    env_data: Optional[dict] = None,
    source: str = "추론",
) -> None:
    """obs 벡터 길이 검증. 불일치 시 상세 ValueError."""
    if expected_dim is None:
        expected_dim = compute_obs_dim()
    actual_dim = int(np.asarray(obs).shape[-1]) if np.asarray(obs).size else 0
    if actual_dim == expected_dim:
        return
    raise ValueError(
        format_obs_dim_mismatch(
            expected_dim,
            actual_dim,
            env_data=env_data,
            source=source,
        )
    )


class SchedulingEnv(gym.Env):
    """
    Scheduling 강화학습 환경

    - 관측: 전역 + PPK/OPER/MODEL bucket + 현재 idle EQP 상태
    - 행동: Discrete(O×P) – (PPK/OPER bucket). EQP·LOT은 규칙 자동 배정
  """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_data: dict,
        render_mode: Optional[str] = None,
        record_history: bool = True,
        record_decision_log: bool = False,
        record_event_log: bool = True,
        max_episode_steps: Optional[int] = None,
        truncate_on_time: bool = True,
    ):
        super().__init__()
        self._env_data = env_data
        self._record_history = record_history
        self._record_decision_log = record_decision_log
        self._record_event_log = record_event_log
        self._decision_log: List[dict] = []
        self._max_episode_steps_override = max_episode_steps
        self._max_episode_steps = 0
        self._episode_steps = 0
        self._truncate_on_time = truncate_on_time

        env_cfg = CONFIG.env
        O = env_cfg.max_oper_count
        P = env_cfg.max_prod_count

        obs_dim = compute_obs_dim()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(O * P)

        self.sim: Optional[SchedulingSimulator] = None
        self.render_mode = render_mode
        self._total_reward = 0.0
        self._O = O
        self._P = P

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.sim = SchedulingSimulator(
            self._env_data, CONFIG.reward,
            record_history=self._record_history,
            record_event_log=self._record_event_log,
        )
        self._total_reward = 0.0
        sim_end = int(self._env_data.get("sim_end_minutes", CONFIG.env.hard_horizon_minutes))
        self._max_episode_steps = (
            self._max_episode_steps_override
            if self._max_episode_steps_override is not None
            else sim_end + 500
        )
        self._episode_steps = 0
        self._decision_log = []
        obs = self.sim.get_observation()
        return obs, {}

    def _ensure_decision_eqp(self) -> Optional[str]:
        """idle 결정 호기가 없으면 시간 전진 또는 동시 idle 탐색."""
        if self.sim.current_idle_eqp() is not None:
            return self.sim.current_idle_eqp()

        while self.sim.get_idle_eqps() and self.sim.current_idle_eqp() is None:
            self.sim._select_same_time_next_eqp()
            if self.sim.current_idle_eqp() is not None:
                return self.sim.current_idle_eqp()

        if self.sim._has_pending_processing():
            self.sim._advance_to_next_decision()
            return self.sim.current_idle_eqp()

        return None

    def _resolve_ppk_oper(self, ppk_oper_idx: int, feasible: List[int]) -> Optional[int]:
        """invalid bucket → 현재 EQP feasible 중 보정."""
        if not feasible:
            return None
        flat = int(ppk_oper_idx) % (self._O * self._P)
        if flat in feasible:
            return flat
        return feasible[0]

    def step(self, action: Union[int, np.ndarray, list]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action_arr = np.asarray(action, dtype=np.int64).flatten()
        ppk_oper_idx = int(action_arr[0]) if len(action_arr) > 0 else 0

        time_at_step_start = self.sim.current_time
        eqp_id = self._ensure_decision_eqp()
        time_advanced = self.sim.current_time != time_at_step_start
        schedule_before = len(self.sim.schedule)

        if self._record_history:
            arrange_actual_before = self.sim.get_remaining_arrange_actual()
            arrange_abstract_before = self.sim.get_abstract_arrange()
            wip_waiting_before = self.sim.get_wip_waiting()
        else:
            arrange_actual_before = None
            arrange_abstract_before = None
            wip_waiting_before = None

        reward = 0.0
        resolved_flat: Optional[int] = None
        if eqp_id is not None:
            if self.sim._eqp_selection == "min_st":
                if self.sim.eqps[eqp_id].status == "idle":
                    reward = self.sim.assign_earliest_st_pending(eqp_id)
                    assignment = self.sim._last_decision_assignment
                    if assignment and assignment.get("eqp_id") == eqp_id:
                        resolved_flat = self.sim.ppk_oper_flat_index(
                            assignment["oper_id"], assignment["plan_prod_key"],
                        )
                else:
                    reward = -0.5
            else:
                feasible = self.sim.get_feasible_ppk_oper(eqp_id)
                resolved_flat = self._resolve_ppk_oper(ppk_oper_idx, feasible)
                if resolved_flat is not None and self.sim.eqps[eqp_id].status == "idle":
                    ppk, oper_id = self.sim.ppk_oper_from_flat(resolved_flat)
                    reward = self.sim.assign_ppk_oper(eqp_id, ppk, oper_id)
                elif feasible:
                    reward = -0.5
                else:
                    # tool cap 등으로 idle이지만 배정 불가 → 시간 전진
                    self.sim._current_eqp = None
                    if self.sim._has_pending_processing():
                        self.sim._advance_to_next_decision()
                        time_advanced = self.sim.current_time != time_at_step_start
        elif not self.sim.is_done():
            if self.sim._has_pending_processing() or self.sim.get_idle_eqps():
                self.sim._advance_to_next_decision()
                time_advanced = self.sim.current_time != time_at_step_start

        self._episode_steps += 1
        terminated = self.sim.is_done()
        truncated = (not terminated) and (
            (self._truncate_on_time and self.sim.current_time >= self.sim.sim_end)
            or self._episode_steps >= self._max_episode_steps
        )

        if self._record_decision_log:
            entry = build_step_decision_entry(
                step=self._episode_steps,
                sim_time=time_at_step_start,
                sim_time_after=self.sim.current_time,
                eqp_id=eqp_id,
                action_flat=ppk_oper_idx,
                resolved_flat=resolved_flat,
                reward=reward,
                sim=self.sim,
                terminated=terminated,
            )
            if len(self.sim.schedule) > schedule_before:
                entry["assigned_lot_id"] = self.sim.schedule[-1].get("LOT_ID")
                if entry["status"] not in ("assigned", "action_corrected"):
                    entry["status"] = "assigned"
            self._decision_log.append(entry)

        if self._record_history:
            wip_for_history = (
                wip_waiting_before
                if time_advanced
                else self.sim.get_wip_waiting()
            )
            self.sim.save_history_step(
                arrange_snapshot=arrange_actual_before,
                arrange_abstract_snapshot=arrange_abstract_before,
                wip_waiting_snapshot=wip_for_history,
            )
        else:
            self.sim.clear_step_assignment()

        # Step A: step reward clip (PPO advantage 안정화, idle/conversion 폭주 방지)
        clip = CONFIG.reward.reward_clip
        if clip and clip > 0:
            reward = float(np.clip(reward, -clip, clip))
        self._total_reward += reward

        obs = self.sim.get_observation()
        info = {
            "total_reward":  self._total_reward,
            "oper_switches": self.sim.stats["oper_switches"],
            "prod_switches": self.sim.stats["prod_switches"],
            "conversions":   self.sim.stats.get("conversions", 0),
            "idle_total":    self.sim.stats["idle_total"],
            "completed_qty": dict(self.sim.stats["completed_qty"]),
            "current_eqp":   eqp_id,
        }
        return obs, reward, terminated, truncated, info

    def get_schedule(self) -> list:
        return self.sim.schedule if self.sim else []

    def get_history(self) -> list:
        return self.sim.history if self.sim else []

    def get_decision_log(self) -> list:
        return list(self._decision_log)

    def action_masks(self) -> np.ndarray:
        """MaskablePPO용 – 현재 idle EQP에서 feasible한 ppk_oper mask (O×P)."""
        n_ppk = self._O * self._P
        ppk_mask = np.zeros(n_ppk, dtype=bool)

        if self.sim is None:
            ppk_mask[0] = True
            return ppk_mask

        eqp_id = self.sim.current_idle_eqp()
        if eqp_id is None:
            ppk_mask[0] = True
            return ppk_mask

        for flat in self.sim.get_feasible_ppk_oper(eqp_id):
            if 0 <= flat < n_ppk:
                ppk_mask[flat] = True

        if not ppk_mask.any():
            ppk_mask[0] = True

        return ppk_mask
