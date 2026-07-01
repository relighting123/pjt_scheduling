"""trace_steps.json 스텝별 누적 간트 PNG 생성 (PPT 삽입용).

Bulk-Fill 특징 시각화:
  - 블록 커밋 구간(연한 해칭): N캐리어 연속 점유 약속
  - 실선 막대: 이미 스케줄된 가공(동일 셋업 LOT 연속 병합)
  - 점선 구간: 커밋했으나 아직 배정 전인 잔여 블록
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).parent.parent
TRACE_PATH = ROOT / "docs/trace_steps.json"
OUT_DIR = ROOT / "docs/gantt/trace"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PPK_COLORS = ["#2E6FB0", "#2E7D4F", "#C8861E", "#7A4FA0", "#0E8AA8"]
NAVY = "#1B3255"
GRAY = "#5C6670"
BLOCK_HATCH = "////"


def _setup_font():
    import matplotlib.font_manager as fm

    for fp in [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]:
        if Path(fp).exists():
            fm.fontManager.addfont(fp)
            matplotlib.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def _merge_segments(schedule: List[dict], eqp_id: str, ppk: str) -> List[dict]:
    """동일 EQP·PPK에서 START=이전 END 인 막대를 하나의 블록 세그먼트로 병합."""
    bars = sorted(
        [b for b in schedule if b["EQP_ID"] == eqp_id and b["PPK"] == ppk],
        key=lambda x: (x["START_TM"], x["END_TM"]),
    )
    merged: List[dict] = []
    for bar in bars:
        if (
            merged
            and merged[-1]["END_TM"] == bar["START_TM"]
            and merged[-1]["PPK"] == bar["PPK"]
        ):
            merged[-1]["END_TM"] = bar["END_TM"]
            merged[-1]["lots"].append(bar.get("LOT_ID", ""))
        else:
            merged.append({
                "EQP_ID": eqp_id,
                "PPK": ppk,
                "START_TM": bar["START_TM"],
                "END_TM": bar["END_TM"],
                "lots": [bar.get("LOT_ID", "")],
            })
    return merged


def _scheduled_end(schedule: List[dict], eqp_id: str, ppk: str) -> int:
    ends = [b["END_TM"] for b in schedule if b["EQP_ID"] == eqp_id and b["PPK"] == ppk]
    return max(ends) if ends else 0


def render_step(step_data: dict, eqp_ids: list, sim_end: int, ppk_colors: dict) -> str:
    step_no = step_data["step"]
    schedule = step_data["schedule"]
    blocks: List[Dict[str, Any]] = step_data.get("blocks") or []
    ypos = {e: i for i, e in enumerate(eqp_ids)}

    fig, ax = plt.subplots(figsize=(7.2, 2.55), dpi=150)

    drawn_eqp_ppk: set[tuple[str, str]] = set()

    # ① 활성 블록: 커밋 구간 + 잔여(점선) + 라벨
    for blk in blocks:
        eqp = blk["eqp_id"]
        ppk = blk["ppk"]
        y = ypos.get(eqp, 0)
        color = ppk_colors.get(ppk, "#999")
        start = int(blk["start_tm"])
        committed_end = int(blk["committed_end_tm"])
        sched_end = int(blk.get("scheduled_end_tm") or _scheduled_end(schedule, eqp, ppk))
        total = int(blk["total"])
        done = int(blk["done"])
        width_commit = max(committed_end - start, 1)

        ax.barh(
            y, width_commit, left=start, height=0.78,
            color=color, alpha=0.15, hatch=BLOCK_HATCH,
            edgecolor=color, linewidth=1.0, zorder=1,
        )

        pending_w = max(committed_end - sched_end, 0)
        if pending_w > 0:
            ax.barh(
                y, pending_w, left=sched_end, height=0.58,
                facecolor="none", edgecolor=color,
                linestyle=(0, (4, 3)), linewidth=1.4, zorder=2,
            )

        label = f"BLKx{total}" if done <= 1 else f"{done}/{total}"
        if blk.get("block_start"):
            label = f"BLKx{total} start"
        ax.annotate(
            label,
            xy=(start + 3, y),
            fontsize=7,
            color=NAVY,
            fontweight="bold",
            va="center",
            zorder=6,
        )
        drawn_eqp_ppk.add((eqp, ppk))

    # ② 스케줄 막대 — 동일 셋업 연속 구간 병합 표시
    eqp_ppks = {(b["EQP_ID"], b["PPK"]) for b in schedule}
    for eqp_id, ppk in sorted(eqp_ppks):
        y = ypos.get(eqp_id, 0)
        color = ppk_colors.get(ppk, "#999")
        for seg in _merge_segments(schedule, eqp_id, ppk):
            st = seg["START_TM"]
            w = max(seg["END_TM"] - st, 1)
            n_lots = len(seg["lots"])
            ax.barh(
                y, w, left=st, height=0.62, color=color,
                edgecolor="white", linewidth=0.8, zorder=4,
            )
            if n_lots >= 2:
                mid = st + w / 2
                ax.text(
                    mid, y, f"x{n_lots}",
                    ha="center", va="center", fontsize=7,
                    color="white", fontweight="bold", zorder=5,
                )
            elif w >= 28:
                ax.text(
                    st + w / 2, y, seg["lots"][0][-3:] if seg["lots"] else "",
                    ha="center", va="center", fontsize=6.5,
                    color="white", zorder=5,
                )

    ax.set_xlim(0, min(sim_end, max(
        [sim_end]
        + [b["END_TM"] for b in schedule]
        + [int(blk["committed_end_tm"]) for blk in blocks],
        default=sim_end,
    )))
    ax.set_ylim(-0.65, len(eqp_ids) - 0.35)
    ax.set_yticks(range(len(eqp_ids)))
    ax.set_yticklabels(eqp_ids, fontsize=8, color=GRAY)
    ax.axvline(sim_end, color="#B3553A", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_xlabel("time (min)", fontsize=8.5, color=GRAY)

    act = step_data.get("action", {})
    blk = step_data.get("block")
    blk_hint = ""
    if blk:
        blk_hint = f" | bulk {blk['done']}/{blk['total']}"
    elif act.get("block_start"):
        blk_hint = f" | BLKx{act.get('block_size', '?')} start"

    title = (
        f"Step {step_no}  |  t={step_data['t']}min  |  "
        f"{step_data['eqp']} -> {step_data['ppk']}{blk_hint}"
    )
    ax.set_title(title, fontsize=9.2, color=NAVY, loc="left", fontweight="bold", pad=4)
    ax.tick_params(axis="x", labelsize=7.5, colors=GRAY)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#C4CFDB")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E4EAF1", linewidth=0.7)

    legend_items = [Patch(facecolor=c, label=p) for p, c in ppk_colors.items()]
    legend_items.extend([
        Patch(facecolor="#94a3b8", alpha=0.25, hatch=BLOCK_HATCH, edgecolor="#64748b",
              label="block commit (N)"),
        Patch(facecolor="none", edgecolor="#64748b", linestyle=(0, (4, 3)),
              label="pending in block"),
    ])
    ax.legend(
        handles=legend_items, loc="upper right", fontsize=6.2,
        frameon=True, framealpha=0.92, ncol=2,
    )

    plt.tight_layout()
    out = OUT_DIR / f"step_{step_no:02d}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out)


def main(target_steps=None):
    _setup_font()
    trace = json.load(open(TRACE_PATH, encoding="utf-8"))
    eqp_ids = trace["eqp_ids"]
    sim_end = trace["sim"]
    ppks = trace.get("prod_keys") or sorted({b["PPK"] for s in trace["steps"] for b in s["schedule"]})
    ppk_colors = {p: PPK_COLORS[i % len(PPK_COLORS)] for i, p in enumerate(ppks)}

    steps = trace["steps"]
    if target_steps:
        steps = [s for s in steps if s["step"] in target_steps]

    for s in steps:
        path = render_step(s, eqp_ids, sim_end, ppk_colors)
        print(f"  render: {Path(path).name}")


if __name__ == "__main__":
    targets = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1, 2, 3, 4, 6, 8]
    main(targets)
