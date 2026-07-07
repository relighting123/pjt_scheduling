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
) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    panels = []
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
        if kind == "eval":
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
    verdict = _convergence_verdict(eval_curve)

    report = {
        "generated_at": _now_iso(),
        "algorithm": algorithm,
        "eval_curve": eval_curve,
        "train_series": progress_series,
        "final_eval_metrics": eval_metrics,
        "convergence": verdict,
    }

    json_path = model_dir / f"training_history_{algorithm}.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )

    png_path = model_dir / f"training_convergence_{algorithm}.png"
    has_png = _plot_png(png_path, eval_curve, progress_series)

    return {
        "json_path": str(json_path),
        "png_path": str(png_path) if has_png else None,
        "verdict": verdict["verdict"],
        "note": verdict["note"],
    }
