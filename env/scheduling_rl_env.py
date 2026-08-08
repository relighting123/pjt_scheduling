"""
env/scheduling_rl_env.py – 벌크 점유(Bulk-Fill) MDP 환경 (Phase 1 스캐폴딩)

기존 SchedulingEnv가 idle EQP마다 1 carrier를 배정하는 것과 달리,
정책이 (PPK/OPER 버킷, 블록 크기 레벨)을 함께 선택하면 해당 장비가
같은 setup으로 N carrier를 연속(블록) 처리하도록 '커밋'한다.

시뮬레이터 무결성(이산 사건: assign 후 즉시 busy)을 해치지 않기 위해
블록은 env 레이어에서 'masked replay'로 실현한다:
  - 블록 시작 결정 시 N을 산출하고 1 carrier 배정 + remaining=N-1 저장
  - 같은 장비의 다음 idle 결정들은 같은 버킷으로 강제(마스크) 재생
  - WIP/계획/tool 잔여 소진 또는 remaining=0이면 블록 종료

블록 크기 상한은 simulator.bulk_block_size()가
min(takt 예산, 가용 WIP, 잔여 계획, tool '잔여'(추가 가용))로 묶는다.

StableBaselines3-Contrib MaskablePPO(MultiDiscrete) 호환:
action_masks()는 [버킷 마스크(O*P) | 크기 마스크(L)] 평탄 연결을 반환한다.
"""
from typing import List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import CONFIG
from simulation.simulator import SchedulingSimulator
from simulation.decision_log import build_step_decision_entry
from env.scheduling_env import compute_obs_dim, validate_axis_capacity

# 블록 크기 레벨 수 (action 두 번째 차원). level→takt 예산 분율.
BULK_SIZE_LEVELS = 4

# ── RL 전용 결정 컨텍스트 관측 ───────────────────────────────────────────────
# simulator.get_observation()에 없는 '이번 결정의 맥락'을 6채널로 덧붙인다.
# 모두 [0,1] 정규화이고, **장비 고유 식별자(호기)는 넣지 않는다** — 아래 참고.
#
#   0 결정 장비 유무(1=idle 결정 중)        결정 스텝/시간 진행 스텝 구분
#   1 결정 장비 범용성(model_breadth) 비율   전용/범용 구분(단일 모델이면 상수)
#   2 첫 셋업 여부(prev_lot_cd 없음)         전환 없이 아무거나 잡아도 되는 시점
#   3 블록 진행 중 여부                      ↓
#   4 블록 잔여 / 블록 총량                  ↓
#   5 idle 장비 수 / 전체 장비 수            지금 몰릴 여유가 있는지
#
# **채널 3·4(블록 상태)가 이 블록의 핵심 이유다.** self._block(어느 버킷을 몇
# carrier 남기고 커밋 중인지)은 env 레이어에만 있고 시뮬레이터에는 없다.
# 그래서 "새 블록을 시작하는 스텝"과 "블록이 3장 남은 스텝"의 관측이 같았는데,
# 행동 의미는 전혀 다르다(블록 중이면 마스크가 같은 버킷을 강제하고
# size_level은 무시된다). 크리틱 입장에서는 남은 커밋 길이에 따라 기대 return이
# 크게 다른데 구분할 수가 없어 value 추정이 흐려지고 advantage도 같이 흐려진다.
#
# ⚠️ **장비 순번(호기 인덱스)은 일부러 넣지 않는다.**
# 한때 "같은 모델 장비가 여러 대면 대칭이라 구분이 안 된다"는 이유로
# `(장비 순번)/(장비 수)` 채널을 넣었는데, 근거가 틀렸다. 결정은 **순차적**이라
# 앞 장비가 배정하면 WIP·cover가 바뀌어 다음 장비는 이미 다른 상태를 본다 —
# SYM_5x5 실제 롤아웃 30개 결정에서 '다른 장비인데 관측이 동일한' 쌍은 0개였다
# (동일하게 보였던 건 t=0 상태를 고정한 채 _current_eqp만 바꿔치기한 인위적
# 테스트에서였다). 반면 순번 채널의 해악은 분명하다:
#   · 호기 종속 — "0.6번 장비는 제품 3"을 외운다. 장비가 다운되거나 추가되면
#     인덱스가 통째로 밀려 학습한 매핑이 깨진다.
#   · 명목형(nominal) ID를 순서형 축에 얹어, 없는 순서 관계를 주입한다.
#     장비 수가 바뀌면 같은 호기의 값도 달라진다.
# 장비를 구분해야 한다면 식별자가 아니라 **성질**(범용성, 셋업 상태, 블록 상태
# 처럼 장비 집합이 바뀌어도 의미가 유지되는 값)로 구분한다.
RL_EXTRA_OBS_DIM = 6


def rl_obs_dim() -> int:
    """SchedulingRLEnv가 실제로 내보내는 관측 차원 (기본 관측 + 결정 컨텍스트)."""
    return compute_obs_dim() + RL_EXTRA_OBS_DIM


def build_env(env_cls: type, data: dict, **kwargs):
    """학습·BC 시연·KPI 평가가 모두 같은 조건의 env를 쓰도록 하는 팩토리.

    특히 `truncate_on_time`: 추론(`inference/runner.py`)은 `False`로 돌려서
    sim_end를 지나서도 남은 재공을 계속 배정한다. 반면 학습 env는 기본값
    `True`라 sim_end에서 끊겼다 — 그래서 정책은 "컷오프 이후 구간"을 한 번도
    겪어보지 못한 채, 벤치마크에서는 그 구간의 전환까지 전부 점수에
    반영당했다(실측: 같은 모델인데 학습 KPI 전환 63회 ↔ 벤치마크 전환 83회).
    `CONFIG.rl.train_truncate_on_time`으로 이를 추론과 맞춘다.
    """
    params = {
        "record_history": False,
        "record_event_log": False,
        "truncate_on_time": bool(getattr(CONFIG.rl, "train_truncate_on_time", True)),
    }
    params.update(kwargs)
    return env_cls(data, **params)


class SchedulingRLEnv(gym.Env):
    """벌크 점유 강화학습 환경 — action=MultiDiscrete([O*P, L])."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_data: dict,
        render_mode: Optional[str] = None,
        record_history: bool = True,
        record_event_log: bool = True,
        truncate_on_time: bool = True,
        size_levels: int = BULK_SIZE_LEVELS,
        record_decision_log: bool = False,
    ):
        super().__init__()
        self._env_data = env_data
        self._record_history = record_history
        self._record_event_log = record_event_log
        self._record_decision_log = record_decision_log
        self._decision_log: list = []
        self._truncate_on_time = truncate_on_time
        self._L = max(int(size_levels), 1)

        env_cfg = CONFIG.env
        self._O = env_cfg.max_oper_count
        self._P = env_cfg.max_prod_count
        self._n_bucket = self._O * self._P
        # 데이터 공정·제품·모델 수가 축 상한을 넘으면 그 버킷은 action 공간에
        # 자리가 없어 정책이 영원히 못 고른다 — 조용히 빠진 스케줄 대신 즉시 실패.
        validate_axis_capacity(env_data, source="SchedulingRLEnv")

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(rl_obs_dim(),), dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([self._n_bucket, self._L])

        self.sim: Optional[SchedulingSimulator] = None
        self.render_mode = render_mode
        self._total_reward = 0.0
        self._episode_steps = 0
        self._max_episode_steps = 0
        # 블록 상태: eqp_id -> [flat_bucket, remaining, total]
        self._block: dict = {}
        self._terminal_reward: float = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.sim = SchedulingSimulator(
            self._env_data, CONFIG.reward,
            record_history=self._record_history,
            record_event_log=self._record_event_log,
        )
        self._total_reward = 0.0
        self._episode_steps = 0
        self._decision_log = []
        sim_end = int(self._env_data.get("sim_end_minutes", CONFIG.env.hard_horizon_minutes))
        self._max_episode_steps = sim_end + 500
        self._block = {}
        self._terminal_reward = 0.0
        return self._observe(None), {}

    # ── 관측 ──────────────────────────────────────────────────────────────────

    def _decision_context_features(self, eqp_id: Optional[str]) -> np.ndarray:
        """결정 컨텍스트 6채널 (RL_EXTRA_OBS_DIM 주석 참고).

        장비 고유 식별자(호기 인덱스)는 넣지 않는다 — 성질만 넣는다.
        """
        feats = np.zeros(RL_EXTRA_OBS_DIM, dtype=np.float32)
        sim = self.sim
        if sim is None:
            return feats

        eqp_ids = self._env_data.get("eqp_ids", [])
        n_eqp = max(len(eqp_ids), 1)

        if eqp_id is not None:
            feats[0] = 1.0
            breadth = sim.model_breadth(eqp_id)
            max_breadth = max((sim.model_breadth(e) for e in eqp_ids), default=1)
            feats[1] = min(breadth / max(max_breadth, 1), 1.0)
            eqp = sim.eqps.get(eqp_id)
            feats[2] = 1.0 if (eqp is None or eqp.prev_lot_cd is None) else 0.0
            blk = self._block.get(eqp_id)
            if blk and blk[1] > 0:
                feats[3] = 1.0
                total = blk[2] if len(blk) > 2 else max(blk[1], 1)
                feats[4] = min(blk[1] / max(total, 1), 1.0)

        idle = sum(1 for e in sim.eqps.values() if e.status == "idle")
        feats[5] = min(idle / n_eqp, 1.0)
        return feats

    def _observe(self, eqp_id: Optional[str]) -> np.ndarray:
        return np.concatenate([
            self.sim.get_observation(),
            self._decision_context_features(eqp_id),
        ]).astype(np.float32)

    # ── 종단 KPI 보상 ─────────────────────────────────────────────────────────

    def _compute_terminal_reward(self) -> float:
        """에피소드 종료 시 벤치마크 KPI를 그대로 보상으로 환산.

        벤치마크가 재는 값과 동일한 정의를 쓴다:
          생산량 = sim_end 안에 END_TM이 들어오는 schedule 행 수(carrier)
          전환수 = stats["conversions"]
        step reward들은 전부 대리지표라 "보상 최고 = KPI 최고"가 보장되지
        않았는데, 이 항이 그 간극을 직접 메운다.
        """
        cfg = CONFIG.reward
        w_thr = getattr(cfg, "w_terminal_throughput", 0.0)
        w_conv = getattr(cfg, "w_terminal_conversion", 0.0)
        if not w_thr and not w_conv:
            return 0.0

        sim = self.sim
        sim_end = sim.sim_end
        produced = sum(
            1 for r in sim.schedule if r.get("END_TM", 0) <= sim_end
        )
        capacity = self.producible_carriers()
        term = 0.0
        if w_thr:
            term += w_thr * min(produced / max(capacity, 1), 1.0)
        if w_conv:
            term += w_conv * sim.stats.get("conversions", 0)

        clip = getattr(cfg, "terminal_reward_clip", 0.0)
        if clip and clip > 0:
            term = float(np.clip(term, -clip, clip))
        return float(term)

    def producible_carriers(self) -> int:
        """이 시나리오에서 최대로 생산 가능한 carrier 수(종단 보상 분모).

        계획(d0_plan_qty)이 있으면 계획 총량, 없으면 초기 재공 총량.
        wf 단위를 carrier 단위로 환산해 schedule 행 수와 같은 척도로 맞춘다.
        """
        data = self._env_data
        wf_unit = max(data.get("max_wf_qty", 1), 1)
        plan_total = sum(
            p.get("d0_plan_qty", 0) for p in data.get("plan", [])
            if p.get("d0_plan_qty", 0) > 0
        )
        if plan_total > 0:
            return max(int(np.ceil(plan_total / wf_unit)), 1)
        wip_total = getattr(self.sim, "_initial_wip_total", 0) if self.sim else 0
        return max(int(np.ceil(max(wip_total, 1) / wf_unit)), 1)

    def _ensure_decision_eqp(self) -> Optional[str]:
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

    # ── 블록 상태 ─────────────────────────────────────────────────────────────

    def _active_block_flat(self, eqp_id: str) -> Optional[int]:
        """진행 중 블록의 버킷 flat. 더 이상 feasible하지 않으면 종료하고 None."""
        blk = self._block.get(eqp_id)
        if not blk or blk[1] <= 0:
            if blk is not None:
                self._block.pop(eqp_id, None)
            return None
        flat = blk[0]
        if flat in self.sim.get_feasible_ppk_oper(eqp_id):
            return flat
        self._block.pop(eqp_id, None)
        return None

    # ── masks ─────────────────────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """MaskablePPO(MultiDiscrete)용 평탄 마스크 [버킷(O*P) | 크기(L)]."""
        bucket_mask = np.zeros(self._n_bucket, dtype=bool)
        size_mask = np.zeros(self._L, dtype=bool)

        if self.sim is None:
            bucket_mask[0] = True
            size_mask[0] = True
            return np.concatenate([bucket_mask, size_mask])

        # step()과 동일하게 지연 해석을 강제한다 — current_idle_eqp()가 아직
        # None인 시점(직전 스텝에서 conversion 시작 등으로 리셋된 직후)에 그대로
        # 읽으면 실제 결정 EQP가 아닌 stale 상태로 마스크가 degenerate(bucket 0만
        # 허용)되고, step()에서야 다른 EQP로 해석되며 액션이 엉뚱한 버킷으로
        # 보정(feasible[0] 폴백)되는 버그가 있었다.
        eqp_id = self._ensure_decision_eqp()
        if eqp_id is None:
            bucket_mask[0] = True
            size_mask[0] = True
            return np.concatenate([bucket_mask, size_mask])

        active = self._active_block_flat(eqp_id)
        if active is not None and 0 <= active < self._n_bucket:
            # 블록 진행 중 → 같은 버킷 강제, 크기 레벨 0 고정
            bucket_mask[active] = True
            size_mask[0] = True
        else:
            if active is not None:
                # config O×P 밖 버킷은 블록 종료 후 일반 마스크로 폴백
                self._block.pop(eqp_id, None)
            for flat in self.sim.get_feasible_ppk_oper(eqp_id):
                if 0 <= flat < self._n_bucket:
                    bucket_mask[flat] = True
            if not bucket_mask.any():
                bucket_mask[0] = True
            size_mask[:] = True  # 모든 크기 레벨 허용 (clamp가 실제 유효성 보정)

        return np.concatenate([bucket_mask, size_mask])

    # ── step ──────────────────────────────────────────────────────────────────

    def step(
        self, action: Union[int, np.ndarray, list],
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action_arr = np.asarray(action, dtype=np.int64).flatten()
        bucket_idx = int(action_arr[0]) if action_arr.size > 0 else 0
        size_level = int(action_arr[1]) if action_arr.size > 1 else 0

        time_at_step_start = self.sim.current_time
        schedule_before = len(self.sim.schedule)
        resolved_flat: Optional[int] = None
        block_size_dbg: Optional[int] = None
        is_block_start = False
        block_size_calc: Optional[dict] = None
        block_progress: Optional[dict] = None

        eqp_id = self._ensure_decision_eqp()
        reward = 0.0

        if eqp_id is not None:
            feasible = self.sim.get_feasible_ppk_oper(eqp_id)
            active = self._active_block_flat(eqp_id)

            if active is not None:
                flat: Optional[int] = active
            else:
                flat = bucket_idx % self._n_bucket
                if flat not in feasible:
                    # feasible는 시뮬레이터가 주는 flat 목록이라 축 상한을 넘는
                    # 값이 섞일 수 있다(공정 수 > max_oper_count). 그런 버킷은
                    # 마스크에 표시조차 못 하므로 폴백 대상에서 제외한다.
                    in_range = [f for f in feasible if 0 <= f < self._n_bucket]
                    flat = in_range[0] if in_range else None
            resolved_flat = flat

            if flat is not None and self.sim.eqps[eqp_id].status == "idle":
                ppk, oper_id = self.sim.ppk_oper_from_flat(flat)
                reward = self.sim.assign_ppk_oper(eqp_id, ppk, oper_id)
                if active is None:
                    # 새 블록 시작 — 배정 성공(-1.0이 아님)이면 shaping 적용
                    if reward != -1.0:
                        is_block_start = True
                        block_size_calc = self.sim.bulk_block_size_breakdown(
                            eqp_id, ppk, oper_id, size_level, self._L,
                        )
                        n = block_size_calc["block_size"]
                        block_size_dbg = n
                        reward += self.sim.bulk_decision_shaping(
                            eqp_id, ppk, oper_id, n,
                        )
                        if n > 1:
                            self._block[eqp_id] = [flat, n - 1, n]
                        block_progress = {
                            "done": 1, "total": max(n, 1),
                            "remaining": max(n - 1, 0),
                        }
                else:
                    # 블록 진행 — remaining 감소
                    blk = self._block.get(eqp_id)
                    if blk:
                        blk[1] -= 1
                        total = blk[2] if len(blk) > 2 else None
                        remaining = max(blk[1], 0)
                        block_progress = {
                            "done": (total - remaining) if total is not None else None,
                            "total": total,
                            "remaining": remaining,
                        }
                        if blk[1] <= 0:
                            self._block.pop(eqp_id, None)
                    if reward == -1.0:
                        self._block.pop(eqp_id, None)
                        if block_progress is not None:
                            block_progress["aborted"] = True
            elif feasible:
                reward = -0.5
            else:
                self.sim._current_eqp = None
                if self.sim._has_pending_processing():
                    self.sim._advance_to_next_decision()
        elif not self.sim.is_done():
            if self.sim._has_pending_processing() or self.sim.get_idle_eqps():
                self.sim._advance_to_next_decision()

        self._episode_steps += 1
        terminated = self.sim.is_done()
        truncated = (not terminated) and (
            (self._truncate_on_time and self.sim.current_time >= self.sim.sim_end)
            or self._episode_steps >= self._max_episode_steps
        )

        # decision_log — history/clear 이전에 빌드(_last_decision_assignment 유효)
        if self._record_decision_log:
            entry = build_step_decision_entry(
                step=self._episode_steps,
                sim_time=time_at_step_start,
                sim_time_after=self.sim.current_time,
                eqp_id=eqp_id,
                action_flat=bucket_idx,
                resolved_flat=resolved_flat,
                reward=reward,
                sim=self.sim,
                terminated=terminated,
            )
            entry["block_start"] = is_block_start
            entry["block_size"] = block_size_dbg
            entry["size_level"] = size_level
            if block_size_calc is not None:
                entry["block_size_calc"] = block_size_calc
            if block_progress is not None:
                entry["block_progress"] = block_progress
            if len(self.sim.schedule) > schedule_before:
                entry["assigned_lot_id"] = self.sim.schedule[-1].get("LOT_ID")
                if entry["status"] not in ("assigned", "action_corrected"):
                    entry["status"] = "assigned"
            self._decision_log.append(entry)

        if self._record_history:
            self.sim.save_history_step()
        else:
            self.sim.clear_step_assignment()

        clip = CONFIG.reward.reward_clip
        if clip and clip > 0:
            reward = float(np.clip(reward, -clip, clip))

        # 종단 KPI 보상은 step clip '이후'에 더한다 — clip 안에 넣으면 ±20에
        # 눌려 생산량 차이가 보상에 반영되지 않는다.
        if terminated or truncated:
            self._terminal_reward = self._compute_terminal_reward()
            reward += self._terminal_reward

        self._total_reward += reward

        obs = self._observe(eqp_id)
        info = {
            "total_reward":  self._total_reward,
            "oper_switches": self.sim.stats["oper_switches"],
            "prod_switches": self.sim.stats["prod_switches"],
            "conversions":   self.sim.stats.get("conversions", 0),
            "idle_total":    self.sim.stats["idle_total"],
            "completed_qty": dict(self.sim.stats["completed_qty"]),
            "current_eqp":   eqp_id,
        }
        if terminated or truncated:
            info["terminal_reward"] = self._terminal_reward
            info["produced_carriers"] = sum(
                1 for r in self.sim.schedule if r.get("END_TM", 0) <= self.sim.sim_end
            )
            info["producible_carriers"] = self.producible_carriers()
        return obs, reward, terminated, truncated, info

    # ── accessors ──────────────────────────────────────────────────────────────

    def get_schedule(self) -> list:
        return self.sim.schedule if self.sim else []

    def get_history(self) -> list:
        return self.sim.history if self.sim else []

    def get_decision_log(self) -> list:
        return list(self._decision_log)


class RandomizedSchedulingRLEnv(SchedulingRLEnv):
    """에피소드마다 데이터셋 풀에서 하나를 무작위로 골라 도는 env (도메인 랜덤화).

    왜 필요한가: 데이터셋 하나당 env 하나를 만들어 co-train하면, 정책이
    "전담이라는 규칙"이 아니라 "이 8개 시나리오의 정답 순열"을 외워버린다
    (실측: 학습에 쓴 BENCH_SUITE 8종에서는 dedication을 +5점 상회하는데,
    학습에 쓰지 않은 HOLDOUT 6종에서는 −7점으로 뒤집혔다 — 대칭
    케이스 H_SYM_6x6에서조차 전환 4회).

    같은 관측/행동 공간을 쓰는 시나리오들을 한 env가 번갈아 돌면
      · 정책이 시나리오 고유 패턴을 외울 수 없어 일반화되고,
      · env 수를 늘리지 않아도 되므로 롤아웃 크기가 커지지 않아
        같은 예산에서 PPO 업데이트 횟수가 유지된다.

    관측/행동 공간은 CONFIG.env의 고정 축(O, P, K)으로 정해지므로 풀 안의
    시나리오끼리 규모가 달라도 공간은 동일하다.
    """

    def __init__(self, datasets: List[dict], *, pool_seed: Optional[int] = None, **kwargs):
        pool = list(datasets)
        if not pool:
            raise ValueError("데이터셋 풀이 비어 있습니다.")
        super().__init__(pool[0], **kwargs)
        self._pool = pool
        self._pool_rng = np.random.default_rng(pool_seed)
        self.current_dataset_index: int = 0

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        idx = int(self._pool_rng.integers(len(self._pool)))
        self.current_dataset_index = idx
        self._env_data = self._pool[idx]
        return super().reset(seed=seed, options=options)
