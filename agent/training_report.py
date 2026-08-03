"""학습 후 수렴 여부를 확인할 수 있는 차트 + 데이터를 model_dir에 남긴다.

- evaluations.npz (SB3 EvalCallback 표준 출력, model_dir/logs/) : eval 보상 곡선
- progress_series (UI 학습의 TrainProgressState.series, 있을 때만) : 학습 중 ep_rew_mean·loss 곡선
둘을 합쳐 JSON(원자료)과 PNG(차트, matplotlib 있을 때만)를 저장한다.
matplotlib이 없거나 evaluations.npz가 없어도 예외를 던지지 않고 가능한 만큼만 남긴다.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_eval_npz(model_dir: Path) -> Optional[dict]:
    """SB3 EvalCallback이 남긴 model_dir/logs/evaluations.npz를 읽는다."""
    npz_path = model_dir / "logs" / "evaluations.npz"
    if not npz_path.exists():
        return None
    try:
        import numpy as np
        data = np.load(npz_path)
        results = data["results"]  # shape (n_evals, n_eval_episodes)
        return {
            "timesteps": data["timesteps"].tolist(),
            "mean_reward": results.mean(axis=1).tolist(),
            "std_reward": results.std(axis=1).tolist(),
            "ep_lengths_mean": data["ep_lengths"].mean(axis=1).tolist()
            if "ep_lengths" in data else [],
        }
    except Exception:
        return None


def _read_bc_history(model_dir: Path) -> Optional[dict]:
    """behavior_clone()이 남긴 model_dir/logs/bc_history.json.

    PPO(eval/kpi 곡선)는 timestep=0부터 시작하는데, 그 시점 정책은 이미
    BC 워밍스타트를 거친 상태다 — BC 구간을 안 보여주면 "왜 eval 점수가
    안 오르지?"에 "BC가 이미 다 올려놨다"는 흔한 답을 확인할 방법이 없다.
    """
    path = model_dir / "logs" / "bc_history.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    history = data.get("history") or []
    if not history:
        return None
    return {
        "epoch": [h["epoch"] for h in history],
        "train_nll": [h["train_nll"] for h in history],
        "val_nll": [h["val_nll"] for h in history],
        "entropy": [h["entropy"] for h in history],
        "value_loss": [h["value_loss"] for h in history],
    }


def _read_kpi_history(model_dir: Path) -> Optional[dict]:
    """KPIEvalCallback이 남긴 model_dir/logs/kpi_evaluations.json.

    KPI 평가(`CONFIG.rl.kpi_eval_enabled=True`)를 쓰면 EvalCallback이 돌지
    않아 evaluations.npz가 없다. 대신 이 파일이 "생산량·전환수·점수" 곡선을
    담고 있고, 이쪽이 shaping 보상 곡선보다 훨씬 해석하기 쉽다
    (보상 곡선이 평평해 보이는 것과 무관하게 KPI는 계단식으로 움직인다).
    """
    path = model_dir / "logs" / "kpi_evaluations.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    history = data.get("history") or []
    if not history:
        return None
    return {
        "timesteps": [h["timestep"] for h in history],
        "score": [h["score"] for h in history],
        "produced": [h["produced"] for h in history],
        "producible": [h.get("producible") for h in history],
        "conversions": [h["conversions"] for h in history],
        "baseline_score": data.get("baseline", {}).get("score")
        if isinstance(data.get("baseline"), dict) else None,
        "conv_weight": data.get("conv_weight", 1.0),
    }


def _kpi_verdict(kpi_curve: dict) -> dict:
    """KPI 점수 곡선 기준 판정 — dedication 기준선 대비 위치까지 함께 본다."""
    scores = kpi_curve["score"]
    baseline = kpi_curve.get("baseline_score")
    best = max(scores)
    first, last = scores[0], scores[-1]
    n = len(scores)
    k = max(1, n // 5)
    tail_spread = max(scores[-k:]) - min(scores[-k:])

    parts = [
        f"KPI 점수(생산−전환): 시작 {first:.1f} → 최고 {best:.1f} → 종료 {last:.1f}."
    ]
    if baseline is not None:
        gap = best - baseline
        parts.append(
            f"Dedication 기준선 {baseline:.1f} 대비 최고 {gap:+.1f}점"
            f"({'상회' if gap > 0 else '미달'})."
        )

    if best > first + 1e-9 and tail_spread <= max(abs(best) * 0.1, 1.0):
        verdict = "converged"
    elif best > first + 1e-9:
        verdict = "improving"
        parts.append(f"최근 구간 변동폭 {tail_spread:.1f}로 아직 흔들립니다.")
    elif n < 5:
        verdict = "unknown"
        parts.append("eval 포인트가 5개 미만이라 추세 판단이 이릅니다.")
    else:
        verdict = "plateau"
        parts.append("워밍스타트 이후 KPI 개선이 없습니다 — 학습 예산·탐색·보상 설계를 점검하세요.")

    return {
        "verdict": verdict,
        "note": " ".join(parts),
        "kpi_first": first,
        "kpi_best": best,
        "kpi_last": last,
        "kpi_baseline": baseline,
        "tail_spread": tail_spread,
        "basis": "kpi",
    }


def _convergence_verdict(eval_curve: Optional[dict]) -> dict:
    """eval 보상 곡선의 앞/뒤 20% 평균을 비교하는 단순 휴리스틱.

    엄밀한 수렴 검정이 아니라 '개선 추세 + 최근 변동폭'만 보는 참고 지표다.
    """
    if not eval_curve or len(eval_curve.get("mean_reward", [])) < 5:
        return {
            "verdict": "unknown",
            "note": "eval 포인트가 5개 미만이라 추세를 판단하기 어렵습니다. "
                    "학습을 더 진행하거나 eval_freq를 낮춰보세요.",
        }
    rewards = eval_curve["mean_reward"]
    n = len(rewards)
    k = max(1, n // 5)
    head = sum(rewards[:k]) / k
    tail = sum(rewards[-k:]) / k
    tail_spread = max(rewards[-k:]) - min(rewards[-k:])
    improvement = tail - head
    denom = max(abs(head), abs(tail), 1e-6)
    rel_improvement = improvement / denom
    stable = tail_spread <= max(abs(tail) * 0.15, 0.5)

    if rel_improvement > 0.05 and stable:
        verdict = "converged"
        note = (
            f"초반 평균({head:.2f}) → 최근 평균({tail:.2f})으로 개선되었고, "
            f"최근 구간 변동폭({tail_spread:.2f})도 작아 수렴한 것으로 보입니다."
        )
    elif rel_improvement > 0.05:
        verdict = "improving"
        note = (
            f"초반 평균({head:.2f}) → 최근 평균({tail:.2f})으로 개선 중이지만, "
            f"최근 구간 변동폭({tail_spread:.2f})이 커서 아직 수렴했다고 보긴 이릅니다."
        )
    elif rel_improvement > -0.05:
        verdict = "plateau"
        note = (
            f"초반 평균({head:.2f})과 최근 평균({tail:.2f})이 비슷합니다. "
            "더 개선되지 않는 정체 구간이거나, 이미 초반에 수렴했을 수 있습니다."
        )
    else:
        verdict = "diverging_or_unstable"
        note = (
            f"초반 평균({head:.2f})보다 최근 평균({tail:.2f})이 더 낮습니다. "
            "학습률이 너무 높거나 보상 설계 문제일 수 있으니 점검이 필요합니다."
        )
    return {
        "verdict": verdict, "note": note,
        "head_mean_reward": head, "tail_mean_reward": tail,
        "tail_spread": tail_spread,
    }


def _plot_png(
    png_path: Path,
    eval_curve: Optional[dict],
    progress_series: Optional[dict],
    kpi_curve: Optional[dict] = None,
    bc_curve: Optional[dict] = None,
) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    panels = []
    # bc를 맨 앞에 둔다 — PPO(kpi/eval)는 timestep=0부터 시작하지만 그 시점
    # 정책은 이미 BC 워밍스타트를 거친 뒤라, BC 구간을 먼저 보여줘야
    # "eval이 왜 안 오르나"가 "BC가 이미 다 올려놨다"인지 한눈에 보인다.
    if bc_curve and bc_curve.get("train_nll"):
        panels.append("bc")
    if kpi_curve and kpi_curve.get("score"):
        panels.append("kpi")
    if eval_curve and eval_curve.get("mean_reward"):
        panels.append("eval")
    if progress_series and any(progress_series.get("ep_rew_mean") or []):
        panels.append("train_reward")
    if progress_series and any(progress_series.get("policy_loss") or []):
        panels.append("loss")
    if not panels:
        return False

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 3.2 * len(panels)), squeeze=False)
    axes = [a[0] for a in axes]

    for ax, kind in zip(axes, panels):
        if kind == "bc":
            ep = bc_curve["epoch"]
            ax.plot(ep, bc_curve["train_nll"], color="#1B3257", label="train NLL")
            ax.plot(ep, bc_curve["val_nll"], color="#B33A3A", linestyle="--", label="val NLL")
            ax.set_title("BC warmstart (before PPO — epoch, not timestep)")
            ax.set_xlabel("BC epoch")
            ax.set_ylabel("NLL loss")
            ax.grid(alpha=0.25)

            ax2 = ax.twinx()
            ax2.plot(ep, bc_curve["entropy"], color="#2E7D4F", alpha=0.55,
                     linestyle=":", label="entropy")
            ax2.set_ylabel("policy entropy")

            handles, labels = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(handles + h2, labels + l2, fontsize=8, loc="best")
        elif kind == "kpi":
            # 점수와 생산량(수백 규모)을 한 축에 그리면 축 범위가 생산량에
            # 끌려가 점수 곡선이 '평평해 보인다' — 정확히 이 리포트가 답해야
            # 할 질문(학습이 되고 있나?)을 가려버린다. 점수/기준선은 주축에
            # 붙여 자기 범위로 스케일하고, 생산량·전환수는 보조축으로 뺀다.
            ts = kpi_curve["timesteps"]
            ax.plot(ts, kpi_curve["score"], color="#1B3257", marker="o",
                    markersize=4, linewidth=2,
                    label="KPI score (produced - conversions)")
            baseline = kpi_curve.get("baseline_score")
            if baseline is not None:
                ax.axhline(baseline, color="#B33A3A", linestyle="--",
                           label=f"Dedication baseline ({baseline:.1f})")
            ax.set_ylabel("KPI score")
            ax.set_xlabel("timesteps")
            ax.set_title("Benchmark KPI vs Dedication baseline")
            ax.grid(alpha=0.25)

            ax2 = ax.twinx()
            ax2.plot(ts, kpi_curve["produced"], color="#2E7D4F", alpha=0.55,
                     linestyle=":", label="produced carriers")
            ax2.plot(ts, kpi_curve["conversions"], color="#C8861E", alpha=0.55,
                     linestyle=":", label="conversions")
            ax2.set_ylabel("carriers / conversions")

            handles, labels = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(handles + h2, labels + l2, fontsize=8, loc="best")
        elif kind == "eval":
            ts = eval_curve["timesteps"]
            mean = eval_curve["mean_reward"]
            std = eval_curve["std_reward"]
            ax.plot(ts, mean, color="#1B3257", label="eval mean reward")
            lo = [m - s for m, s in zip(mean, std)]
            hi = [m + s for m, s in zip(mean, std)]
            ax.fill_between(ts, lo, hi, color="#2E6FB0", alpha=0.2, label="+/-1 std")
            ax.set_title("Eval reward (EvalCallback) - convergence check")
            ax.set_xlabel("timesteps")
            ax.set_ylabel("mean episode reward")
            ax.legend()
        elif kind == "train_reward":
            ts = progress_series.get("timesteps", [])
            ax.plot(ts, progress_series.get("ep_rew_mean", []), color="#2E7D4F")
            ax.set_title("Training ep_rew_mean (rollout moving average)")
            ax.set_xlabel("timesteps")
            ax.set_ylabel("ep_rew_mean")
        elif kind == "loss":
            ts = progress_series.get("timesteps", [])
            ax.plot(ts, progress_series.get("policy_loss", []), color="#C8861E", label="policy_loss")
            ax.plot(ts, progress_series.get("value_loss", []), color="#B33A3A", label="value_loss")
            ax.set_title("PPO loss")
            ax.set_xlabel("timesteps")
            ax.legend()

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    return True


def save_training_convergence_report(
    model_dir: Path,
    algorithm: str = "scheduling_rl",
    progress_series: Optional[dict[str, list]] = None,
    eval_metrics: Optional[dict[str, Any]] = None,
) -> dict:
    """학습 종료 직후 호출. JSON(+가능하면 PNG)을 model_dir에 남기고 요약을 반환한다."""
    model_dir = Path(model_dir)
    eval_curve = _read_eval_npz(model_dir)
    kpi_curve = _read_kpi_history(model_dir)
    bc_curve = _read_bc_history(model_dir)
    # KPI 곡선이 있으면 그쪽을 판정 근거로 쓴다 — shaping 보상보다 실제
    # 벤치마크 목표에 직접 대응하기 때문.
    verdict = _kpi_verdict(kpi_curve) if kpi_curve else _convergence_verdict(eval_curve)

    report = {
        "generated_at": _now_iso(),
        "algorithm": algorithm,
        "bc_curve": bc_curve,
        "eval_curve": eval_curve,
        "kpi_curve": kpi_curve,
        "train_series": progress_series,
        "final_eval_metrics": eval_metrics,
        "convergence": verdict,
    }

    json_path = model_dir / f"training_history_{algorithm}.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )

    png_path = model_dir / f"training_convergence_{algorithm}.png"
    has_png = _plot_png(png_path, eval_curve, progress_series, kpi_curve, bc_curve)

    return {
        "json_path": str(json_path),
        "png_path": str(png_path) if has_png else None,
        "verdict": verdict["verdict"],
        "note": verdict["note"],
    }
