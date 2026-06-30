"""
벤치마크 스케줄 → 간트 차트 PNG 렌더러 (PPT 삽입용)
=========================================================
bench_suite_schedules.json을 읽어, 한 데이터셋에 대해 3개 알고리즘의
스케줄을 세로로 쌓은 비교 간트 이미지를 생성한다.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).parent.parent
SUITE_ROOT = ROOT / "data/dataset"
OUT_DIR = ROOT / "docs/gantt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 제품별 색상(코퍼레이트 톤)
PPK_COLORS = ["#2E6FB0", "#2E7D4F", "#C8861E", "#7A4FA0", "#0E8AA8",
              "#B3553A", "#4A6D8C", "#5A8F3C", "#A03D6E", "#3C6E8F"]
CONV_COLOR = "#E8B84B"
NAVY = "#1B3255"
GRAY = "#5C6670"

ALGO_LABEL = {"earliest_st": "Earliest-ST (단순 규칙)",
              "minprogress": "Min-Progress (휴리스틱)",
              "bulkfill": "Bulk-Fill PPO (학습 모델)"}
ALGO_ORDER = ["earliest_st", "minprogress", "bulkfill"]


def kpi_line(rows, sim, conv_plans):
    eqps = sorted({r["EQP_ID"] for r in rows})
    prod = len(rows)
    conv = len(conv_plans)
    from collections import defaultdict
    by = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by[r["EQP_ID"]][r["PLAN_PROD_KEY"]] += 1
    ded = sum(1 for e in eqps if by[e] and max(by[e].values()) / max(sum(by[e].values()), 1) >= 0.8)
    return prod, conv, ded, len(eqps)


def render(name, data, meta):
    ppks = sorted({r["PLAN_PROD_KEY"] for a in data.values() for r in a["schedule"]})
    color_of = {p: PPK_COLORS[i % len(PPK_COLORS)] for i, p in enumerate(ppks)}

    sim = next(iter(data.values()))["sim"]
    eqp_ids = sorted({e for a in data.values() for e in a["eqp_ids"]})
    n_algo = sum(1 for a in ALGO_ORDER if a in data)

    fig, axes = plt.subplots(n_algo, 1, figsize=(9.6, 1.0 + 1.35 * n_algo),
                             squeeze=False, dpi=150)
    axes = axes[:, 0]
    ai = 0
    for algo in ALGO_ORDER:
        if algo not in data:
            continue
        ax = axes[ai]; ai += 1
        d = data[algo]
        rows = d["schedule"]; conv_plans = d["conversion_plans"]
        ypos = {e: i for i, e in enumerate(eqp_ids)}
        for r in rows:
            y = ypos.get(r["EQP_ID"], 0)
            st = r["START_TM"]; w = r["END_TM"] - r["START_TM"]
            ax.barh(y, w, left=st, height=0.62, color=color_of.get(r["PLAN_PROD_KEY"], "#999"),
                    edgecolor="white", linewidth=0.6, zorder=3)
        for c in conv_plans:
            y = ypos.get(c["eqp_id"], 0)
            st = c["conv_start_min"]; w = c["conv_end_min"] - c["conv_start_min"]
            ax.barh(y, w, left=st, height=0.62, color=CONV_COLOR, hatch="////",
                    edgecolor="#9A7B1E", linewidth=0.6, zorder=4)
        prod, conv, ded, neqp = kpi_line(rows, sim, conv_plans)
        ax.set_xlim(0, sim)
        ax.set_ylim(-0.6, len(eqp_ids) - 0.4)
        ax.set_yticks(range(len(eqp_ids)))
        ax.set_yticklabels(eqp_ids, fontsize=7.5, color=GRAY)
        ax.axvline(sim, color="#B3553A", linestyle="--", linewidth=1.0, zorder=2)
        ax.set_title(f"{ALGO_LABEL[algo]}   ·   생산 {prod}/{meta['total']}  ·  전환 {conv}회  ·  전담 {ded}/{neqp}",
                     fontsize=9.5, color=NAVY, loc="left", fontweight="bold", pad=3)
        ax.tick_params(axis="x", labelsize=7.5, colors=GRAY)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#C4CFDB")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E4EAF1", linewidth=0.7)
        if ai == n_algo:
            ax.set_xlabel("시간 (분)", fontsize=8.5, color=GRAY)

    legend_items = [Patch(facecolor=color_of[p], label=p) for p in ppks]
    legend_items.append(Patch(facecolor=CONV_COLOR, hatch="////", edgecolor="#9A7B1E", label="전환(Conversion)"))
    fig.legend(handles=legend_items, loc="lower center", ncol=min(len(legend_items), 7),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("", fontsize=1)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = OUT_DIR / f"gantt_{name}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  렌더: {out.name}")
    return str(out)


def main(targets=None):
    # 한글 지원 폰트 등록 (WenQuanYi Zen Hei: CJK+Hangul 포함)
    import matplotlib.font_manager as fm
    for fp in ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
               "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
        if Path(fp).exists():
            fm.fontManager.addfont(fp)
            matplotlib.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

    sched = json.load(open(SUITE_ROOT / "bench_suite_schedules.json", encoding="utf-8"))
    meta = {m["id"]: m for m in json.load(open(SUITE_ROOT / "bench_suite_meta.json", encoding="utf-8"))}
    names = targets or list(sched.keys())
    for name in names:
        if name in sched:
            render(name, sched[name], meta[name])


if __name__ == "__main__":
    main(sys.argv[1:] or None)
