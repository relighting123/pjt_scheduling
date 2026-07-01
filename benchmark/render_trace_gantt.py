"""trace_steps.json 스텝별 누적 간트 PNG 생성 (PPT 삽입용)."""
import json
import sys
from pathlib import Path

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


def render_step(step_data: dict, eqp_ids: list, sim_end: int, ppk_colors: dict) -> str:
    step_no = step_data["step"]
    schedule = step_data["schedule"]
    ypos = {e: i for i, e in enumerate(eqp_ids)}

    fig, ax = plt.subplots(figsize=(7.2, 2.4), dpi=150)
    for bar in schedule:
        y = ypos.get(bar["EQP_ID"], 0)
        st = bar["START_TM"]
        w = max(bar["END_TM"] - st, 1)
        color = ppk_colors.get(bar["PPK"], "#999")
        ax.barh(y, w, left=st, height=0.62, color=color,
                edgecolor="white", linewidth=0.6, zorder=3)

    ax.set_xlim(0, sim_end)
    ax.set_ylim(-0.6, len(eqp_ids) - 0.4)
    ax.set_yticks(range(len(eqp_ids)))
    ax.set_yticklabels(eqp_ids, fontsize=8, color=GRAY)
    ax.axvline(sim_end, color="#B3553A", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_xlabel("time (min)", fontsize=8.5, color=GRAY)
    title = (
        f"Step {step_no}  |  t={step_data['t']}min  |  "
        f"{step_data['eqp']} -> {step_data['ppk']}  |  "
        f"bars={len(schedule)}"
    )
    ax.set_title(title, fontsize=9.5, color=NAVY, loc="left", fontweight="bold", pad=4)
    ax.tick_params(axis="x", labelsize=7.5, colors=GRAY)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#C4CFDB")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E4EAF1", linewidth=0.7)

    legend_items = [Patch(facecolor=c, label=p) for p, c in ppk_colors.items()]
    if legend_items:
        ax.legend(handles=legend_items, loc="upper right", fontsize=7,
                  frameon=True, framealpha=0.9, ncol=len(legend_items))

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
        print(f"  렌더: {Path(path).name}")


if __name__ == "__main__":
    targets = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1, 2, 3, 4, 6, 8]
    main(targets)
