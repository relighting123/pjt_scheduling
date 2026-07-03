"""State 산식 해설 슬라이드용 — 항목별로 서로 다른 개념 그림 17장 생성.

STATE_WALKTHROUGH(benchmark/state_source_walkthrough.py)의 각 항목이 "diagram" 키로
가리키는 PNG를 docs/gantt/state_items/<key>.png 에 만든다. 모든 수치는 MINI-A
가상 시나리오(같은 파일의 MINI_A_DATASET)와 정확히 일치시켜, 그림을 본 뒤 옆
슬라이드의 대입 계산을 그대로 따라갈 수 있게 한다.

그림마다 형태를 의도적으로 다르게 가져간다: 타임라인 / 막대비교 / 스택막대 /
게이지 / 카드대조 / 흐름도 / 인코딩 스케일 등.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "docs/gantt/state_items"

NAVY = "#1B3255"
ACCENT = "#2E6FB0"
STEEL = "#4A6D8C"
GREEN = "#2E7D4F"
RED = "#B33A3A"
AMBER = "#C8861E"
GRAY = "#5C6670"
LIGHT = "#EEF2F7"
LINE = "#C4CFDB"
PURPLE = "#7A4FA0"


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
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def _bare_ax(figsize, dpi=200):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_axis_off()
    return fig, ax


def _save(fig, key):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{key}.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"저장 완료: {out}")


def _card(ax, x, y, w, h, title, value, color, sub=""):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=LIGHT, edgecolor=color, linewidth=1.3))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=8.2, color=NAVY, fontweight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.05, value, ha="center", va="center", fontsize=12.5, color=color, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + 0.1, sub, ha="center", va="bottom", fontsize=6.6, color=GRAY)


# ── obs01 : time_norm / takt_margin — 타임라인 2구간 브래킷 ──────────────────
def draw_obs01():
    fig, ax = _bare_ax((10.6, 1.7))
    y = 0.55
    ax.plot([0, 480], [y, y], color=LINE, linewidth=3, solid_capstyle="round")
    ax.plot([0, 120], [y, y], color=ACCENT, linewidth=7, solid_capstyle="round")
    ax.plot([120, 480], [y, y], color=GREEN, linewidth=7, solid_capstyle="round")
    for x, lbl in [(0, "0"), (120, "t=120"), (480, "sim_end=\nsoft_cutoff=480")]:
        ax.plot([x], [y], marker="o", color=NAVY, markersize=5, zorder=5)
        ax.text(x, y - 0.16, lbl, ha="center", va="top", fontsize=7.6, color=NAVY, fontweight="bold")
    ax.annotate("", xy=(118, y + 0.32), xytext=(2, y + 0.32),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, linewidth=1.4))
    ax.text(60, y + 0.42, "time_norm = 120/480 = 0.25", ha="center", color=ACCENT, fontsize=8.2, fontweight="bold")
    ax.annotate("", xy=(478, y + 0.32), xytext=(122, y + 0.32),
                arrowprops=dict(arrowstyle="<->", color=GREEN, linewidth=1.4))
    ax.text(300, y + 0.42, "takt_margin = 360/480 = 0.75", ha="center", color=GREEN, fontsize=8.2, fontweight="bold")
    ax.set_xlim(-20, 500)
    ax.set_ylim(0, 1.1)
    _save(fig, "obs01")


# ── obs23 : remaining_lots / plan_progress — 두 저장고 막대 ──────────────────
def draw_obs23():
    fig, ax = _bare_ax((10.6, 1.7))
    bars = [
        (1.0, "LOT 풀", 25, 40, ACCENT, "remaining_lots = 25/40 = 0.625"),
        (6.2, "계획 달성", 10, 60, GREEN, "plan_progress = 10/60 = 0.1667"),
    ]
    for x0, title, part, total, color, label in bars:
        w = 4.0
        ax.add_patch(Rectangle((x0, 0.45), w, 0.5, facecolor=LIGHT, edgecolor=LINE, linewidth=1.0))
        fw = w * part / total
        ax.add_patch(Rectangle((x0, 0.45), fw, 0.5, facecolor=color, alpha=0.85))
        ax.text(x0, 1.05, title, fontsize=8.6, color=NAVY, fontweight="bold", va="bottom")
        ax.text(x0 + w / 2, 0.70, f"{part} / {total}", ha="center", va="center", fontsize=9.5, color="white" if fw > w*0.25 else NAVY, fontweight="bold")
        ax.text(x0 + w / 2, 0.30, label, ha="center", va="top", fontsize=7.6, color=color, fontweight="bold")
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 1.3)
    _save(fig, "obs23")


# ── ch0 : valid — 설비-공정 연결 그래프 ──────────────────────────────────────
def draw_ch0():
    fig, ax = _bare_ax((10.6, 1.9))
    opers = [("OPER001", 2.0), ("OPER002", 2.0)]
    eqps = [("EQP001", 0.0), ("EQP002", 0.0), ("EQP003", 0.0)]
    ox = {"OPER001": 8.4, "OPER002": 8.4}
    oy = {"OPER001": 1.35, "OPER002": 0.45}
    ex = 1.2
    ey = {"EQP001": 1.55, "EQP002": 0.9, "EQP003": 0.25}
    edges = [("EQP001", "OPER001"), ("EQP002", "OPER001"),
             ("EQP001", "OPER002"), ("EQP002", "OPER002"), ("EQP003", "OPER002")]
    for e, o in edges:
        color = GREEN if o == "OPER002" else LINE
        lw = 1.8 if o == "OPER002" else 1.1
        ax.plot([ex + 0.9, ox[o] - 0.9], [ey[e], oy[o]], color=color, linewidth=lw, zorder=1)
    for e, y in ey.items():
        ax.add_patch(Rectangle((ex - 0.55, y - 0.16), 1.1, 0.32, facecolor="white", edgecolor=STEEL, linewidth=1.2))
        ax.text(ex, y, e, ha="center", va="center", fontsize=7.6, color=NAVY, fontweight="bold")
    for o, y in oy.items():
        col = GREEN if o == "OPER002" else STEEL
        ax.add_patch(Rectangle((ox[o] - 0.75, y - 0.2), 1.5, 0.4, facecolor=LIGHT, edgecolor=col, linewidth=1.6))
        ax.text(ox[o], y, o, ha="center", va="center", fontsize=8.4, color=col, fontweight="bold")
    ax.text(ox["OPER002"] + 1.0, oy["OPER002"], "valid_mis=[0]\n→ ch0=1.0", ha="left", va="center",
            fontsize=8.0, color=GREEN, fontweight="bold")
    ax.set_xlim(0, 11.2)
    ax.set_ylim(-0.1, 1.9)
    _save(fig, "ch0")


# ── ch1_2 : wip_ratio_total / wip_ratio_ppk — 스택 막대 2단 ─────────────────
def draw_ch1_2():
    fig, ax = _bare_ax((10.6, 1.9))
    # 전체 175
    segs_total = [("PPK001·OP1", 140, GRAY), ("PPK001·OP2 (이 버킷)", 15, ACCENT), ("PPK002·OP1", 20, PURPLE)]
    x = 0.6
    y0 = 1.05
    w_total = 9.6
    cx = x
    for label, q, color in segs_total:
        w = w_total * q / 175
        hi = color == ACCENT
        ax.add_patch(Rectangle((cx, y0), w, 0.42, facecolor=color, alpha=0.95 if hi else 0.55,
                                edgecolor="white", linewidth=1.0))
        if q >= 15:
            ax.text(cx + w / 2, y0 + 0.21, f"{q}", ha="center", va="center", fontsize=7.6,
                     color="white", fontweight="bold")
        cx += w
    ax.text(x, y0 + 0.52, "팹 전체 WIP = 175", fontsize=8.0, color=NAVY, fontweight="bold")
    ax.text(x + w_total, y0 + 0.21, "  wip_ratio_total\n  = 15/175 = 0.0857", fontsize=7.6, color=ACCENT,
            fontweight="bold", va="center")

    segs_ppk = [("PPK001·OP1", 140, GRAY), ("PPK001·OP2 (이 버킷)", 15, ACCENT)]
    y1 = 0.15
    w_ppk = w_total * 155 / 175
    cx = x
    for label, q, color in segs_ppk:
        w = w_ppk * q / 155
        hi = color == ACCENT
        ax.add_patch(Rectangle((cx, y1), w, 0.42, facecolor=color, alpha=0.95 if hi else 0.55,
                                edgecolor="white", linewidth=1.0))
        if q >= 15:
            ax.text(cx + w / 2, y1 + 0.21, f"{q}", ha="center", va="center", fontsize=7.6,
                     color="white", fontweight="bold")
        cx += w
    ax.text(x, y1 + 0.52, "PPK001 안에서만 = 155", fontsize=8.0, color=NAVY, fontweight="bold")
    ax.text(x + w_ppk, y1 + 0.21, "  wip_ratio_ppk\n  = 15/155 = 0.0968", fontsize=7.6, color=ACCENT,
            fontweight="bold", va="center")
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 1.9)
    _save(fig, "ch1_2")


# ── ch3 : min_end_time — free_at 막대비교 ───────────────────────────────────
def draw_ch3():
    fig, ax = _bare_ax((10.8, 1.9))
    data = [("EQP001", 320, STEEL), ("EQP002", 120, GREEN), ("EQP003", 400, ACCENT)]
    maxv = 480
    bw = 1.5
    bar_x = {}
    for i, (name, v, color) in enumerate(data):
        x = 0.8 + i * 2.4
        bar_x[name] = x + bw / 2
        h = 1.15 * v / maxv
        hi = name == "EQP002"
        ax.add_patch(Rectangle((x, 0.15), bw, h, facecolor=color, alpha=1.0 if hi else 0.45,
                                edgecolor=color, linewidth=1.6 if hi else 0.8))
        ax.text(x + bw / 2, 0.15 + h + 0.08, f"free_at={v}", ha="center", fontsize=8.0,
                color=color, fontweight="bold")
        ax.text(x + bw / 2, 0.0, name, ha="center", va="top", fontsize=8.4, color=NAVY, fontweight="bold")
    ax.annotate("min = 120 (EQP002)\n→ min_end_time = 120/480 = 0.25", xy=(bar_x["EQP002"], 0.45),
                xytext=(8.4, 0.55), fontsize=7.8, color=GREEN, fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=1.3))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(-0.3, 1.7)
    _save(fig, "ch3")


# ── ch4 : throughput_ratio — 분모 경쟁 막대 ──────────────────────────────────
def draw_ch4():
    fig, ax = _bare_ax((10.6, 1.8))
    ax.add_patch(Rectangle((0.6, 0.9), 1.25 * 320 / 320, 0.4, facecolor=RED, alpha=0.85))
    ax.text(0.6, 1.38, "max_gantt_end = 320  (팹 전체 최댓값 → 분모로 채택)", fontsize=7.8, color=RED, fontweight="bold")
    ax.add_patch(Rectangle((0.6, 0.25), 1.25 * 125 / 320, 0.4, facecolor=GRAY, alpha=0.55))
    ax.text(0.6, 0.73, "이 버킷만: min_end_time(120) + 처리시간(5) = 125", fontsize=7.8, color=GRAY, fontweight="bold")
    ax.text(9.5, 1.1, "denom = max(320, 125)\n     = 320", ha="right", fontsize=8.4, color=RED, fontweight="bold")
    ax.text(9.5, 0.45, "throughput_ratio\n= wip_q(15) / 320\n= 0.0469", ha="right", fontsize=8.4, color=NAVY, fontweight="bold")
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 1.8)
    _save(fig, "ch4")


# ── ch5 : same_ppk — 카드 대조 ───────────────────────────────────────────────
def draw_ch5():
    fig, ax = _bare_ax((9.6, 1.7))
    _card(ax, 0.6, 0.3, 3.0, 1.1, "직전 배정 PPK", "PPK001", NAVY)
    _card(ax, 6.0, 0.3, 3.0, 1.1, "이 버킷 PPK", "PPK001", ACCENT)
    ax.annotate("", xy=(5.9, 0.85), xytext=(3.7, 0.85),
                arrowprops=dict(arrowstyle="-", color=GREEN, linewidth=2.0))
    ax.text(4.8, 1.05, "일치 OK", ha="center", fontsize=10, color=GREEN, fontweight="bold")
    ax.text(4.8, 0.55, "same_ppk = 1.0", ha="center", fontsize=8.4, color=GREEN, fontweight="bold")
    ax.text(4.8, 0.05, "(다른 제품이었다면 0.0)", ha="center", fontsize=7.0, color=GRAY)
    ax.set_xlim(0, 9.6)
    ax.set_ylim(0, 1.7)
    _save(fig, "ch5")


# ── ch6_7 : prev_takt / post_takt — 공정 흐름도 ─────────────────────────────
def draw_ch6_7():
    fig, ax = _bare_ax((10.6, 1.9))
    ax.add_patch(Rectangle((0.8, 0.9), 2.4, 0.6, facecolor=LIGHT, edgecolor=STEEL, linewidth=1.4))
    ax.text(2.0, 1.2, "OPER001", ha="center", va="center", fontsize=9.5, color=NAVY, fontweight="bold")
    ax.text(2.0, 0.7, "eff_takt = 1.0", ha="center", fontsize=7.6, color=STEEL)
    ax.annotate("", xy=(6.0, 1.2), xytext=(3.4, 1.2),
                arrowprops=dict(arrowstyle="->", color=ACCENT, linewidth=2.0))
    ax.add_patch(Rectangle((6.4, 0.9), 2.4, 0.6, facecolor=LIGHT, edgecolor=ACCENT, linewidth=1.4))
    ax.text(7.6, 1.2, "OPER002", ha="center", va="center", fontsize=9.5, color=ACCENT, fontweight="bold")
    ax.text(7.6, 0.7, "eff_takt = 7.2  (여유 ↑)", ha="center", fontsize=7.6, color=ACCENT, fontweight="bold")

    ax.text(0.8, 0.2, "[OPER001 버킷]  prev=0/360=0.0   post=1/360=0.02", fontsize=7.8, color=NAVY)
    ax.text(0.8, -0.15, "[OPER002 버킷]  prev=1/360=0.0028   post=0/360=0.0", fontsize=7.8, color=NAVY)
    ax.set_xlim(0, 10.6)
    ax.set_ylim(-0.4, 1.9)
    _save(fig, "ch6_7")


# ── ch8 : self_st — 가공시간 비교 막대 ───────────────────────────────────────
def draw_ch8():
    fig, ax = _bare_ax((8.6, 1.8))
    data = [("OPER001", 2, STEEL), ("OPER002 (이 버킷)", 5, RED)]
    for i, (name, v, color) in enumerate(data):
        x = 1.0 + i * 3.6
        h = 1.2 * v / 5
        ax.add_patch(Rectangle((x, 0.2), 1.6, h, facecolor=color, alpha=0.85))
        ax.text(x + 0.8, 0.2 + h + 0.08, f"{v}분/장", ha="center", fontsize=8.6, color=color, fontweight="bold")
        ax.text(x + 0.8, 0.05, name, ha="center", va="top", fontsize=8.2, color=NAVY, fontweight="bold")
    ax.text(6.4, 1.1, "max_arrange_st = 5\n(전체 중 최댓값)\n→ self_st = 5/5 = 1.0", fontsize=7.8, color=RED,
            fontweight="bold")
    ax.set_xlim(0, 8.6)
    ax.set_ylim(-0.2, 1.7)
    _save(fig, "ch8")


# ── ch9 : plan_urgency — 게이지 2개 비교 ─────────────────────────────────────
def _gauge(ax, cx, val, vmax, color, label, sub):
    ax.add_patch(Rectangle((cx - 1.5, 0.75), 3.0, 0.34, facecolor=LIGHT, edgecolor=LINE))
    w = 3.0 * min(val / vmax, 1.0)
    ax.add_patch(Rectangle((cx - 1.5, 0.75), w, 0.34, facecolor=color, alpha=0.9))
    ax.text(cx, 1.28, label, ha="center", fontsize=8.4, color=NAVY, fontweight="bold")
    ax.text(cx, 0.5, sub, ha="center", fontsize=7.6, color=color, fontweight="bold")


def draw_ch9():
    fig, ax = _bare_ax((9.6, 1.8))
    ax.set_axis_off()
    _gauge(ax, 2.6, 0.0023, 0.03, GREEN, "T_avail=360 (현재)", "urgency = 0.0023 (여유)")
    _gauge(ax, 7.0, 0.0278, 0.03, RED, "가정: T_avail=30", "urgency = 0.0278 (12배 ↑)")
    ax.text(4.8, 0.05, "같은 gap=50 · plan_qty=60 인데 남은 시간만 짧아져도 urgency가 뛴다", ha="center",
            fontsize=7.2, color=GRAY)
    ax.set_xlim(0, 9.6)
    ax.set_ylim(0, 1.8)
    _save(fig, "ch9")


# ── ch10_11 : wip_lot_cd / wip_temp — 인코딩 스케일 ──────────────────────────
def draw_ch10_11():
    fig, ax = _bare_ax((10.2, 1.8))
    for row, (title, ticks, val, color) in enumerate([
        ("LOT_CD 인덱스 {B:0, A:1}", [("B", 0.0), ("A", 1.0)], 1.0, ACCENT),
        ("TEMP 인덱스 (미사용)", [("None", 0.0)], 0.0, GRAY),
    ]):
        y = 1.15 - row * 0.7
        ax.plot([1.0, 8.5], [y, y], color=LINE, linewidth=2.5, solid_capstyle="round")
        for name, v in ticks:
            x = 1.0 + 7.5 * v
            ax.plot([x], [y], marker="|", color=NAVY, markersize=14, markeredgewidth=2)
            ax.text(x, y - 0.2, name, ha="center", fontsize=7.4, color=NAVY)
        xv = 1.0 + 7.5 * val
        ax.plot([xv], [y], marker="o", color=color, markersize=9, zorder=5)
        ax.text(0.8, y, title, ha="right", va="center", fontsize=8.0, color=NAVY, fontweight="bold")
        ax.text(xv, y + 0.22, f"이 버킷 값 = {val}", ha="center", fontsize=7.6, color=color, fontweight="bold")
    ax.set_xlim(0, 10.2)
    ax.set_ylim(-0.1, 1.7)
    _save(fig, "ch10_11")


# ── ch12_13 : needs_conversion / tool_can_assign — 세팅 불일치 카드 ─────────
def draw_ch12_13():
    fig, ax = _bare_ax((10.2, 1.8))
    _card(ax, 0.5, 0.3, 2.6, 1.1, "EQP002 현재 세팅", "B", STEEL)
    _card(ax, 3.5, 0.3, 2.6, 1.1, "버킷 요구 세팅", "A", RED)
    ax.text(3.15, 0.85, "≠", ha="center", va="center", fontsize=20, color=RED, fontweight="bold")
    ax.text(2.4, 0.05, "needs_conversion (ch12) = 1.0", ha="center", fontsize=7.8, color=RED, fontweight="bold")

    _card(ax, 7.1, 0.3, 2.6, 1.1, "공구(tool) 배정", "가능 OK", GREEN)
    ax.text(8.4, 0.05, "tool_can_assign (ch13) = 1.0", ha="center", fontsize=7.8, color=GREEN, fontweight="bold")
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 1.7)
    _save(fig, "ch12_13")


# ── ch14 : achievable_ratio — 누적 워터폴 막대 ───────────────────────────────
def draw_ch14():
    fig, ax = _bare_ax((10.2, 1.9))
    segs = [("완료 done", 10, GRAY), ("이 버킷 WIP", 15, ACCENT), ("상류 OPER001 WIP", 140, GREEN)]
    x = 0.8
    cum = 0
    scale = 8.0 / 165
    legend_bits = []
    for name, q, color in segs:
        w = q * scale
        ax.add_patch(Rectangle((x + cum * scale, 0.5), w, 0.5, facecolor=color, alpha=0.9, edgecolor="white"))
        ax.text(x + (cum + q / 2) * scale, 0.75, f"{q}", ha="center", va="center", fontsize=8.0,
                color="white", fontweight="bold")
        legend_bits.append((name, q, color))
        cum += q
    ax.text(x + (cum - 140 / 2) * scale, 1.12, "상류 OPER001 WIP", ha="center", fontsize=7.6, color=GREEN,
            fontweight="bold")
    ax.text(x, 1.5, "  +  ".join(f"{n}={q}" for n, q, _ in legend_bits[:2]), fontsize=7.6, color=NAVY,
            fontweight="bold")
    plan_x = x + 60 * scale
    ax.axvline(plan_x, color=RED, linestyle="--", linewidth=1.6, ymin=0.15, ymax=0.75)
    ax.text(plan_x, 0.15, "plan_qty=60", ha="center", fontsize=7.8, color=RED, fontweight="bold")
    ax.text(x + cum * scale + 0.3, 0.75, "→ 누적 165 ≥ 60\nachievable_ratio\n= min(60,165)/60 = 1.0",
            fontsize=7.8, color=NAVY, fontweight="bold", va="center")
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 1.9)
    _save(fig, "ch14")


# ── ch15 : projected_cover_ratio — 커버 설비 필터링 ─────────────────────────
def draw_ch15():
    fig, ax = _bare_ax((10.2, 1.8))
    _card(ax, 0.5, 0.3, 2.6, 1.1, "EQP001", "제외 X", GRAY, "prev_oper≠OPER002")
    _card(ax, 3.5, 0.3, 2.6, 1.1, "EQP003", "포함 OK", GREEN, "prev_oper=OPER002\n기여 cov=72")
    ax.annotate("", xy=(9.3, 1.0), xytext=(6.4, 0.9), arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=1.6))
    ax.text(9.5, 1.0, "cov(72) vs need(50)\nratio = min(72/50,2)/2\n= 0.72", ha="left", va="center",
            fontsize=7.8, color=NAVY, fontweight="bold")
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 1.7)
    _save(fig, "ch15")


# ── eqp : needs_conversion / avoidable_frac — 수요 커버 스택 막대 ────────────
def draw_eqp():
    fig, ax = _bare_ax((10.2, 1.9))
    x = 0.8
    scale = 8.5 / 155
    segs = [("EQP001 대신 커버 가능", 80, GREEN), ("EQP003 대신 커버 가능", 16, GREEN),
            ("EQP002만 가능 (잔여)", 59, RED)]
    cum = 0
    for name, q, color in segs:
        w = q * scale
        alpha = 0.55 if "EQP001" in name else (0.85 if "EQP003" in name else 0.85)
        ax.add_patch(Rectangle((x + cum * scale, 0.55), w, 0.55, facecolor=color, alpha=alpha, edgecolor="white"))
        ax.text(x + (cum + q / 2) * scale, 0.82, f"{q}", ha="center", va="center", fontsize=8.0,
                color="white", fontweight="bold")
        ax.text(x + (cum + q / 2) * scale, 1.22, name, ha="center", fontsize=6.6, color=NAVY, rotation=0)
        cum += q
    ax.text(x, 0.2, "세팅A 총수요 = 155  (140+15)", fontsize=7.8, color=NAVY, fontweight="bold")
    ax.text(x + cum * scale + 0.3, 0.82, "alt_cap=96 → α=0.619\n→ avoidable_frac=0.619", fontsize=7.8,
            color=GREEN, fontweight="bold", va="center")
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 1.9)
    _save(fig, "eqp")


# ── ctx : 직전 배정 레코드 → 인코딩 카드 4장 ─────────────────────────────────
def draw_ctx():
    fig, ax = _bare_ax((10.6, 1.9))
    fields = [
        ("last_ppk", "PPK001", "1/9=0.111", ACCENT),
        ("last_oper", "OPER002", "1/2=0.5", GREEN),
        ("last_eqp", "EQP003", "2/2=1.0", STEEL),
        ("last_lot_cd", "세팅 A", "1/1=1.0", RED),
    ]
    w = 2.15
    gap = 0.25
    for i, (title, raw, enc, color) in enumerate(fields):
        x = 0.5 + i * (w + gap)
        ax.add_patch(Rectangle((x, 0.85), w, 0.75, facecolor=LIGHT, edgecolor=color, linewidth=1.3))
        ax.text(x + w / 2, 1.42, title, ha="center", fontsize=7.8, color=NAVY, fontweight="bold")
        ax.text(x + w / 2, 1.12, raw, ha="center", fontsize=9.0, color=color, fontweight="bold")
        ax.annotate("", xy=(x + w / 2, 0.55), xytext=(x + w / 2, 0.82),
                     arrowprops=dict(arrowstyle="->", color=color, linewidth=1.4))
        ax.text(x + w / 2, 0.35, enc, ha="center", fontsize=7.8, color=color, fontweight="bold")
    ax.text(0.5, 1.75, "직전 배정: EQP003 → PPK001 / OPER002 / 세팅A", fontsize=8.2, color=NAVY, fontweight="bold")
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 1.95)
    _save(fig, "ctx")


DRAWERS = {
    "obs01": draw_obs01,
    "obs23": draw_obs23,
    "ch0": draw_ch0,
    "ch1_2": draw_ch1_2,
    "ch3": draw_ch3,
    "ch4": draw_ch4,
    "ch5": draw_ch5,
    "ch6_7": draw_ch6_7,
    "ch8": draw_ch8,
    "ch9": draw_ch9,
    "ch10_11": draw_ch10_11,
    "ch12_13": draw_ch12_13,
    "ch14": draw_ch14,
    "ch15": draw_ch15,
    "eqp": draw_eqp,
    "ctx": draw_ctx,
}


def render_all():
    _setup_font()
    for key, fn in DRAWERS.items():
        fn()
    # obs45는 3설비 간트 스냅샷(render_mini_a_gantt.py)을 그대로 재사용
    import shutil
    from benchmark.render_mini_a_gantt import OUT as MINI_A_SNAPSHOT
    from benchmark.render_mini_a_gantt import render as render_mini_a_snapshot
    if not MINI_A_SNAPSHOT.is_file():
        render_mini_a_snapshot()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MINI_A_SNAPSHOT, OUT_DIR / "obs45.png")


if __name__ == "__main__":
    render_all()
