"""State 산식 해설(MINI-A) 슬라이드용 스냅샷 간트 PNG 생성.

docs/make_ppt.py의 state_source_slide()가 모든 State 항목 슬라이드 중앙에
공통으로 삽입하는 이미지. STATE_WALKTHROUGH의 모든 대입 계산이 참조하는
가상 시나리오(MINI-A, benchmark/state_source_walkthrough.py)를 t=120분
시점의 3대 설비 상태로 시각화해 소스 코드 대입값을 눈으로 먼저 확인할 수 있게 한다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT = ROOT / "docs/gantt/mini_a_snapshot.png"

NAVY = "#1B3255"
ACCENT = "#2E6FB0"
STEEL = "#4A6D8C"
GREEN = "#2E7D4F"
RED = "#B33A3A"
GRAY = "#5C6670"
LINE = "#C4CFDB"

CURRENT_TIME = 120
SIM_END = 480

# (row_label, status, free_at, bar_color, job_label)
ROWS = [
    ("EQP001  (세팅 A)", "busy", 320, STEEL, "가동 중 (작업 미상)"),
    ("EQP002 ★ (세팅 B · 결정대상)", "idle", 120, GREEN, "idle → 다음 배정 대기"),
    ("EQP003  (세팅 A)", "busy", 400, ACCENT, "PPK001·OPER002·세팅A"),
]

NOMINAL_BUSY_WIDTH = 110  # 시작시각 미상: free_at 기준 역산해 표시만 함


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


def render():
    _setup_font()
    fig, ax = plt.subplots(figsize=(11.6, 1.55), dpi=200)
    fig.subplots_adjust(left=0.205, right=0.985, top=0.90, bottom=0.30)

    n = len(ROWS)
    for i, (label, status, free_at, color, job_label) in enumerate(ROWS):
        y = n - 1 - i
        ax.barh(y, SIM_END, left=0, height=0.68, color="#F4F6F8", edgecolor=LINE, linewidth=0.6, zorder=1)
        if status == "busy":
            start = max(free_at - NOMINAL_BUSY_WIDTH, 0)
            ax.barh(y, free_at - start, left=start, height=0.5, color=color, alpha=0.92,
                    edgecolor="white", linewidth=0.6, zorder=3)
            ax.text(free_at + 6, y, f"free_at={free_at}", fontsize=6.6, color=color,
                    va="center", ha="left", fontweight="bold")
        else:
            ax.barh(y, SIM_END - free_at, left=free_at, height=0.5, color=color, alpha=0.20,
                    hatch="////", edgecolor=color, linewidth=0.8, zorder=3)
            ax.plot([free_at], [y], marker="o", color=color, markersize=4.5, zorder=4)
            ax.text(free_at + 6, y, f"idle (free_at={free_at})", fontsize=6.6, color=color,
                    va="center", ha="left", fontweight="bold")
        ax.text(-8, y, label, fontsize=7.3, color=NAVY, va="center", ha="right", fontweight="bold")
        ax.text(4, y + 0.35, job_label, fontsize=6.0, color=GRAY, va="bottom", ha="left")

    ax.axvline(CURRENT_TIME, color=RED, linestyle="--", linewidth=1.2, zorder=5)
    ax.text(CURRENT_TIME, n + 0.05, f"t = {CURRENT_TIME} (지금)", fontsize=7.0, color=RED,
            va="bottom", ha="center", fontweight="bold")

    ax.set_xlim(-95, SIM_END + 68)
    ax.set_ylim(-0.85, n + 0.15)
    ax.set_yticks([])
    ax.set_xticks([0, 120, 240, 360, 480])
    ax.set_xticklabels(["0", "120", "240", "360", "480(sim_end)"], fontsize=6.4, color=GRAY)
    ax.tick_params(axis="x", length=2.2, colors=GRAY)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E4EAF1", linewidth=0.6, zorder=0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor="white")
    plt.close(fig)
    print(f"저장 완료: {OUT}")


if __name__ == "__main__":
    render()
