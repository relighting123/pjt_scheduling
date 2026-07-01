"""trace_steps.json 스텝별 누적 간트 PNG 생성 (PPT 삽입용).

Bulk-Fill 시각화 (구현과 일치):
  - 연한 해칭 = N캐리어 연속 처리 커밋 구간 (약속)
  - 실선 막대 = 이 스텝까지 스케줄에 등록된 LOT
  - 캐리어 칸 경계 = 흰 세로선, 막대 내부 텍스트 없음
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]:
        if Path(fp).exists():
            name = fm.FontProperties(fname=fp).get_name()
            fm.fontManager.addfont(fp)
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def _merge_segments(schedule: List[dict], eqp_id: str, ppk: str) -> List[dict]:
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


def _carrier_dividers(ax, start: int, proc_min: int, total: int, y: float, height: float):
    y0 = y - height / 2
    y1 = y + height / 2
    for i in range(1, total):
        x = start + i * proc_min
        ax.plot([x, x], [y0, y1], color="#cbd5e1", linewidth=0.9, solid_capstyle="butt", zorder=3)


def _label_above(ax, x: float, y: float, text: str, color: str):
    ax.annotate(
        text,
        xy=(x, y + 0.52),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=NAVY,
        fontweight="bold",
        zorder=8,
        clip_on=False,
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor=color,
            linewidth=0.8,
            alpha=0.97,
        ),
    )


def _draw_bulk_block(ax, blk: Dict[str, Any], schedule: List[dict], y: float, color: str):
    start = int(blk["start_tm"])
    committed_end = int(blk["committed_end_tm"])
    proc_min = max(int(blk.get("proc_min") or 60), 1)
    total = int(blk["total"])
    done = int(blk.get("done") or 0)
    eqp = blk["eqp_id"]
    ppk = blk["ppk"]
    full_w = max(committed_end - start, proc_min)
    sched_end = int(blk.get("scheduled_end_tm") or _scheduled_end(schedule, eqp, ppk))
    sched_w = max(min(sched_end - start, full_w), proc_min if done > 0 else 0)

    # ① 커밋 구간 (N캐리어 연속 처리 약속)
    ax.barh(
        y, full_w, left=start, height=0.72,
        color=color, alpha=0.16, hatch=BLOCK_HATCH,
        edgecolor=color, linewidth=1.0, zorder=1,
    )
    _carrier_dividers(ax, start, proc_min, total, y, 0.72)

    # ② 실제 스케줄 등록 LOT (이 스텝까지 누적)
    if sched_w > 0:
        ax.barh(
            y, sched_w, left=start, height=0.58,
            color=color, alpha=0.92,
            edgecolor="white", linewidth=0.7, zorder=4,
        )

    if blk.get("block_start"):
        label = f"BLK×{total} 커밋  ·  LOT 1/{total}"
    else:
        label = f"BLK×{total}  ·  LOT {done}/{total}"
    _label_above(ax, start + full_w / 2, y, label, color)


def render_step(step_data: dict, eqp_ids: list, sim_end: int, ppk_colors: dict) -> str:
    step_no = step_data["step"]
    schedule = step_data["schedule"]
    blocks: List[Dict[str, Any]] = step_data.get("blocks") or []
    ypos = {e: i for i, e in enumerate(eqp_ids)}

    fig, ax = plt.subplots(figsize=(7.2, 2.75), dpi=150)

    block_eqp_ppk: Set[Tuple[str, str]] = set()
    for blk in blocks:
        eqp = blk["eqp_id"]
        ppk = blk["ppk"]
        y = ypos.get(eqp, 0)
        color = ppk_colors.get(ppk, "#999")
        _draw_bulk_block(ax, blk, schedule, y, color)
        block_eqp_ppk.add((eqp, ppk))

    eqp_ppks = {(b["EQP_ID"], b["PPK"]) for b in schedule}
    for eqp_id, ppk in sorted(eqp_ppks):
        if (eqp_id, ppk) in block_eqp_ppk:
            continue
        y = ypos.get(eqp_id, 0)
        color = ppk_colors.get(ppk, "#999")
        for seg in _merge_segments(schedule, eqp_id, ppk):
            st = seg["START_TM"]
            w = max(seg["END_TM"] - st, 1)
            ax.barh(
                y, w, left=st, height=0.62, color=color,
                edgecolor="white", linewidth=0.8, zorder=4,
            )

    ax.set_xlim(0, min(sim_end, max(
        [sim_end]
        + [b["END_TM"] for b in schedule]
        + [int(blk["committed_end_tm"]) for blk in blocks],
        default=sim_end,
    )))
    ax.set_ylim(-0.85, len(eqp_ids) - 0.15)
    ax.set_yticks(range(len(eqp_ids)))
    ax.set_yticklabels(eqp_ids, fontsize=8, color=GRAY)
    ax.axvline(sim_end, color="#B3553A", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_xlabel("time (min)", fontsize=8.5, color=GRAY)

    act = step_data.get("action", {})
    blk = step_data.get("block")
    blk_hint = ""
    if blk:
        if blk.get("block_start"):
            blk_hint = f" | BLK×{blk['total']} 커밋 · LOT 1/{blk['total']}"
        else:
            blk_hint = f" | LOT {blk['done']}/{blk['total']}"
    elif act.get("block_start"):
        n = act.get("block_size") or act.get("block_total") or "?"
        blk_hint = f" | BLK×{n} 커밋"

    title = (
        f"Step {step_no}  |  t={step_data['t']}min  |  "
        f"{step_data['eqp']} -> {step_data['ppk']}{blk_hint}"
    )
    ax.set_title(title, fontsize=9.0, color=NAVY, loc="left", fontweight="bold", pad=8)
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
              label="커밋 구간 (N약속)"),
        Patch(facecolor="#2E6FB0", alpha=0.92, edgecolor="#64748b",
              label="스케줄 등록 LOT"),
    ])
    ax.legend(
        handles=legend_items, loc="upper right", fontsize=6.0,
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
