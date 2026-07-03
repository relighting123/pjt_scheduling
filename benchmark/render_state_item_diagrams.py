"""State 산식 해설 슬라이드용 — 항목별로 서로 다른 개념 그림 17장 생성.

STATE_WALKTHROUGH(benchmark/state_source_walkthrough.py)의 각 항목이 "diagram" 키로
가리키는 PNG를 docs/gantt/state_items/<key>.png 에 만든다. 모든 수치는 MINI-A
가상 시나리오(같은 파일의 MINI_A_DATASET)와 정확히 일치시켜, 그림을 본 뒤 옆
슬라이드의 대입 계산을 그대로 따라갈 수 있게 한다.

그림마다 형태를 의도적으로 다르게 가져간다: 타임라인 / 막대비교 / 스택막대 /
게이지 / 카드대조 / 흐름도 / 인코딩 스케일 등. 슬라이드에 삽입될 때 가로로 긴
박스에 꽉 차도록, 각 그림의 캔버스 가로세로비를 ~10:1 근방으로 넓게 잡는다
(레이아웃 좌표는 K 배율로 일괄 확장해 실제 셀 간격이 넓어지도록 한다).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
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
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=9.0, color=NAVY, fontweight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.05, value, ha="center", va="center", fontsize=13.5, color=color, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + 0.1, sub, ha="center", va="bottom", fontsize=7.2, color=GRAY)


# ── obs01 : time_norm / takt_margin — 타임라인 2구간 브래킷 ──────────────────
def draw_obs01():
    # x축이 실제 분(min) 단위 값(0~480)이라 좌표는 그대로 두고 캔버스만 넓힌다.
    fig, ax = _bare_ax((17.5, 1.7))
    y = 0.55
    ax.plot([0, 480], [y, y], color=LINE, linewidth=4, solid_capstyle="round")
    ax.plot([0, 120], [y, y], color=ACCENT, linewidth=9, solid_capstyle="round")
    ax.plot([120, 480], [y, y], color=GREEN, linewidth=9, solid_capstyle="round")
    for x, lbl in [(0, "0"), (120, "t=120"), (480, "sim_end=soft_cutoff=480")]:
        ax.plot([x], [y], marker="o", color=NAVY, markersize=6, zorder=5)
        ax.text(x, y - 0.17, lbl, ha="center", va="top", fontsize=10.5, color=NAVY, fontweight="bold")
    ax.annotate("", xy=(116, y + 0.34), xytext=(4, y + 0.34),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, linewidth=1.8))
    ax.text(60, y + 0.46, "time_norm = 120/480 = 0.25", ha="center", color=ACCENT, fontsize=11, fontweight="bold")
    ax.annotate("", xy=(476, y + 0.34), xytext=(124, y + 0.34),
                arrowprops=dict(arrowstyle="<->", color=GREEN, linewidth=1.8))
    ax.text(300, y + 0.46, "takt_margin = 360/480 = 0.75", ha="center", color=GREEN, fontsize=11, fontweight="bold")
    ax.set_xlim(-24, 504)
    ax.set_ylim(0, 1.15)
    _save(fig, "obs01")


# ── obs23 : remaining_lots / plan_progress — 두 저장고 막대 ──────────────────
def draw_obs23():
    K = 1.75
    fig, ax = _bare_ax((10.6 * K, 1.7))
    bars = [
        (1.0 * K, "LOT 풀  (전체 시뮬레이션 초기 배정량 기준)", 25, 40, ACCENT,
         "remaining_lots = 25/40 = 0.625"),
        (6.2 * K, "계획 달성  ((PPK001,OPER002) 계획 기준)", 10, 60, GREEN,
         "plan_progress = 10/60 = 0.1667"),
    ]
    w = 4.0 * K
    for x0, title, part, total, color, label in bars:
        ax.add_patch(Rectangle((x0, 0.45), w, 0.5, facecolor=LIGHT, edgecolor=LINE, linewidth=1.0))
        fw = w * part / total
        ax.add_patch(Rectangle((x0, 0.45), fw, 0.5, facecolor=color, alpha=0.85))
        ax.text(x0, 1.05, title, fontsize=9.6, color=NAVY, fontweight="bold", va="bottom")
        ax.text(x0 + w / 2, 0.70, f"{part} / {total}", ha="center", va="center", fontsize=11,
                color="white" if fw > w * 0.25 else NAVY, fontweight="bold")
        ax.text(x0 + w / 2, 0.30, label, ha="center", va="top", fontsize=8.8, color=color, fontweight="bold")
    ax.set_xlim(0, 11.2 * K)
    ax.set_ylim(0, 1.3)
    _save(fig, "obs23")


# ── obs45 : conv_idle_ratio / tool_util — 3설비 간트 스냅샷은 render_mini_a_gantt.py 참고 ──


# ── ch0 : valid — 설비-모델-공정 연결 그래프 (PPK와 무관함을 명시) ───────────
def draw_ch0():
    K = 1.85
    fig, ax = _bare_ax((22.8, 2.0))
    model_x = 5.0 * K
    oper_x = 9.4 * K
    oy = {"OPER001": 1.5, "OPER002": 0.5}
    ex = 1.2 * K
    ey = {"EQP001": 1.7, "EQP002": 1.0, "EQP003": 0.3}
    edges = [("EQP001", "OPER001"), ("EQP002", "OPER001"),
             ("EQP001", "OPER002"), ("EQP002", "OPER002"), ("EQP003", "OPER002")]
    my = 1.0
    # EQP → model(M1) → OPER 2단 연결로, "설비가 아니라 model 단위" 판정임을 시각화
    for e in ey:
        color = GREEN
        ax.plot([ex + 0.9 * K, model_x - 0.9 * K], [ey[e], my], color=LINE, linewidth=1.3, zorder=1)
    for e, o in edges:
        color = GREEN if o == "OPER002" else LINE
        lw = 2.0 if o == "OPER002" else 1.2
        ax.plot([model_x + 0.9 * K, oper_x - 1.1 * K], [my, oy[o]], color=color, linewidth=lw, zorder=1)
    for e, y in ey.items():
        ax.add_patch(Rectangle((ex - 0.55 * K, y - 0.17), 1.1 * K, 0.34, facecolor="white", edgecolor=STEEL, linewidth=1.2))
        ax.text(ex, y, e, ha="center", va="center", fontsize=8.8, color=NAVY, fontweight="bold")
    ax.add_patch(Rectangle((model_x - 0.75 * K, my - 0.2), 1.5 * K, 0.4, facecolor=to_rgba(AMBER, 0.14),
                            edgecolor=AMBER, linewidth=1.6))
    ax.text(model_x, my, "model M1", ha="center", va="center", fontsize=9.6, color=AMBER, fontweight="bold")
    for o, y in oy.items():
        col = GREEN if o == "OPER002" else STEEL
        ax.add_patch(Rectangle((oper_x - 0.85 * K, y - 0.2), 1.7 * K, 0.4, facecolor=LIGHT, edgecolor=col, linewidth=1.6))
        ax.text(oper_x, y, o, ha="center", va="center", fontsize=9.6, color=col, fontweight="bold")
    ax.text(ex, 2.1, "개별 설비(EQP)", ha="center", fontsize=8.4, color=GRAY)
    ax.text(model_x, 2.1, "모델(model) 그룹", ha="center", fontsize=8.4, color=AMBER)
    ax.text(oper_x, 2.1, "공정(OPER)", ha="center", fontsize=8.4, color=GRAY)
    ax.text(oper_x + 1.6 * K, oy["OPER002"], "valid_mis=[0]\n→ ch0=1.0\n(PPK 무관)", ha="left", va="center",
            fontsize=9.2, color=GREEN, fontweight="bold")
    ax.set_xlim(0, oper_x + 3.6 * K)
    ax.set_ylim(-0.15, 2.35)
    _save(fig, "ch0")


# ── ch1_2 : wip_ratio_total / wip_ratio_ppk — 스택 막대 2단 ─────────────────
def draw_ch1_2():
    K = 1.85
    fig, ax = _bare_ax((10.6 * K, 1.9))
    segs_total = [("PPK001·OP1", 140, GRAY), ("PPK001·OP2 (이 버킷)", 15, ACCENT), ("PPK002·OP1", 20, PURPLE)]
    x = 0.6 * K
    y0 = 1.05
    w_total = 9.6 * K
    cx = x
    for label, q, color in segs_total:
        w = w_total * q / 175
        hi = color == ACCENT
        ax.add_patch(Rectangle((cx, y0), w, 0.42, facecolor=color, alpha=0.95 if hi else 0.55,
                                edgecolor="white", linewidth=1.2))
        ax.text(cx + w / 2, y0 + 0.21, f"{q}", ha="center", va="center", fontsize=9.2,
                color="white", fontweight="bold")
        cx += w
    ax.text(x, y0 + 0.54, "팹 전체 WIP = 175", fontsize=9.2, color=NAVY, fontweight="bold")
    ax.text(x + w_total + 0.15 * K, y0 + 0.21, "wip_ratio_total\n= 15/175 = 0.0857", fontsize=9.0, color=ACCENT,
            fontweight="bold", va="center")

    segs_ppk = [("PPK001·OP1", 140, GRAY), ("PPK001·OP2 (이 버킷)", 15, ACCENT)]
    y1 = 0.15
    w_ppk = w_total * 155 / 175
    cx = x
    for label, q, color in segs_ppk:
        w = w_ppk * q / 155
        hi = color == ACCENT
        ax.add_patch(Rectangle((cx, y1), w, 0.42, facecolor=color, alpha=0.95 if hi else 0.55,
                                edgecolor="white", linewidth=1.2))
        ax.text(cx + w / 2, y1 + 0.21, f"{q}", ha="center", va="center", fontsize=9.2,
                color="white", fontweight="bold")
        cx += w
    ax.text(x, y1 + 0.54, "PPK001 안에서만 = 155", fontsize=9.2, color=NAVY, fontweight="bold")
    ax.text(x + w_ppk + 0.15 * K, y1 + 0.21, "wip_ratio_ppk\n= 15/155 = 0.0968", fontsize=9.0, color=ACCENT,
            fontweight="bold", va="center")
    ax.set_xlim(0, 11.2 * K)
    ax.set_ylim(0, 1.9)
    _save(fig, "ch1_2")


# ── ch3 : min_end_time — (OPER,model) free_at 막대비교, PPK 무관 강조 ───────
def draw_ch3():
    K = 1.85
    fig, ax = _bare_ax((10.8 * K, 2.0))
    ax.text(0.2 * K, 1.9, "(OPER002, model M1) 소속 설비들의 free_at  —  PPK와는 무관", fontsize=9.6,
            color=NAVY, fontweight="bold")
    data = [("EQP001", 320, STEEL), ("EQP002", 120, GREEN), ("EQP003", 400, ACCENT)]
    maxv = 480
    bw = 1.5 * K
    bar_x = {}
    for i, (name, v, color) in enumerate(data):
        x = 0.8 * K + i * 2.4 * K
        bar_x[name] = x + bw / 2
        h = 1.2 * v / maxv
        hi = name == "EQP002"
        ax.add_patch(Rectangle((x, 0.15), bw, h, facecolor=color, alpha=1.0 if hi else 0.45,
                                edgecolor=color, linewidth=1.8 if hi else 0.9))
        ax.text(x + bw / 2, 0.15 + h + 0.08, f"free_at={v}", ha="center", fontsize=9.4,
                color=color, fontweight="bold")
        ax.text(x + bw / 2, 0.0, name, ha="center", va="top", fontsize=9.8, color=NAVY, fontweight="bold")
    ax.annotate("min = 120 (EQP002)\n→ min_end_time = 120/480 = 0.25", xy=(bar_x["EQP002"], 0.48),
                xytext=(8.4 * K, 0.6), fontsize=9.2, color=GREEN, fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=1.5))
    ax.set_xlim(0, 10.8 * K)
    ax.set_ylim(-0.3, 2.0)
    _save(fig, "ch3")


# ── ch4 : throughput_ratio — 분자(wip_q) · 분모 경쟁 막대 ───────────────────
def draw_ch4():
    fig, ax = _bare_ax((18.5, 2.0))
    bx = 1.0
    bw = 5.6
    ax.text(bx, 1.78, "①  max_gantt_end = 320  (팹 전체에서 이미 스케줄된 가장 늦은 종료시각)",
            fontsize=10.4, color=RED, fontweight="bold")
    ax.add_patch(Rectangle((bx, 1.05), bw, 0.42, facecolor=RED, alpha=0.85))
    ax.text(bx, 0.9, "②  이 버킷만 봤을 때: 125  (= min_end_time(120) + 처리시간(st×wf_unit=5))",
            fontsize=10.4, color=GRAY, fontweight="bold")
    ax.add_patch(Rectangle((bx, 0.2), bw * 125 / 320, 0.42, facecolor=GRAY, alpha=0.55))

    rx = bx + bw + 1.4
    ax.annotate("", xy=(rx - 0.3, 1.1), xytext=(bx + bw + 0.15, 1.1),
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.8))
    ax.add_patch(Rectangle((rx, 0.98), 7.2, 0.5, facecolor=to_rgba(RED, 0.1), edgecolor=RED, linewidth=1.3))
    ax.text(rx + 0.2, 1.23, "분모 = max(①, ②) = max(320,125) = 320", fontsize=10.4, color=RED,
            fontweight="bold", va="center")
    ax.add_patch(Rectangle((rx, 0.2), 7.2, 0.66, facecolor=to_rgba(NAVY, 0.06), edgecolor=NAVY, linewidth=1.3))
    ax.text(rx + 0.2, 0.68, "분자 = wip_q(이 버킷 대기 재공) = 15", fontsize=10.4, color=NAVY, fontweight="bold")
    ax.text(rx + 0.2, 0.38, "throughput_ratio = 15 / 320 = 0.0469", fontsize=10.4, color=NAVY, fontweight="bold")
    ax.set_xlim(0, 18.5)
    ax.set_ylim(0, 1.95)
    _save(fig, "ch4")


# ── ch5 : same_ppk — 카드 대조 ───────────────────────────────────────────────
def draw_ch5():
    K = 2.0
    fig, ax = _bare_ax((9.6 * K, 1.7))
    _card(ax, 0.6 * K, 0.3, 3.0 * K, 1.1, "직전 배정 PPK", "PPK001", NAVY)
    _card(ax, 6.0 * K, 0.3, 3.0 * K, 1.1, "이 버킷 PPK", "PPK001", ACCENT)
    ax.annotate("", xy=(5.9 * K, 0.85), xytext=(3.7 * K, 0.85),
                arrowprops=dict(arrowstyle="-", color=GREEN, linewidth=2.4))
    ax.text(4.8 * K, 1.08, "일치 OK", ha="center", fontsize=12, color=GREEN, fontweight="bold")
    ax.text(4.8 * K, 0.55, "same_ppk = 1.0", ha="center", fontsize=9.6, color=GREEN, fontweight="bold")
    ax.text(4.8 * K, 0.05, "(다른 제품이었다면 0.0)", ha="center", fontsize=8.0, color=GRAY)
    ax.set_xlim(0, 9.6 * K)
    ax.set_ylim(0, 1.7)
    _save(fig, "ch5")


# ── ch6_7 : prev_takt / post_takt — eff_takt 산식 분해 + 버킷별 정규화 ──────
def draw_ch6_7():
    """가장 중요한 항목이라 다른 그림보다 캔버스를 넉넉히 쓰고, eff_takt의 두 후보
    (설비능력 cap_takt vs 계획수요 demand_takt)를 직접 막대로 분해해 보여준다."""
    fig, ax = _bare_ax((19.5, 3.6))

    # ── 상단 공통 산식 (세 줄, 넉넉한 간격) ──
    ax.text(9.6, 3.35, "eff_takt(ppk,op) = max( cap_takt, demand_takt ) × wf_unit", ha="center",
            fontsize=12.5, color=NAVY, fontweight="bold")
    ax.text(9.6, 3.02, "— 설비 능력 한계(cap_takt)와 계획 수요(demand_takt) 중 더 '여유로운(큰)' 간격을 채택 —",
            ha="center", fontsize=9.4, color=GRAY)
    ax.text(9.6, 2.72, "max_takt = T_avail × wf_unit = 360 × 1 = 360   (prev/post_takt 정규화 분모)",
            ha="center", fontsize=9.6, color=STEEL, fontweight="bold")

    def op_block(cx, op, spw, n, cap_takt, has_plan, q_plan, demand_takt, eff, color):
        bx = cx - 2.3
        ax.add_patch(Rectangle((bx, 1.95), 4.6, 0.58, facecolor=LIGHT, edgecolor=color, linewidth=1.8))
        ax.text(cx, 2.34, f"{op}", ha="center", va="center", fontsize=13.5, color=color, fontweight="bold")
        ax.text(cx, 2.07, f"spw(장당 가공)={spw}분  ·  배정설비 n={n}", ha="center", va="center",
                fontsize=8.6, color=GRAY)
        # 후보 막대 2개: cap_takt(설비능력) vs demand_takt(계획수요)
        maxscale = 0.9 / max(cap_takt, demand_takt or 0, 1)
        ax.add_patch(Rectangle((bx, 1.15), 2.15, 0.03 + 0.9 * min(cap_takt * maxscale, 1),
                                facecolor=STEEL, alpha=0.55))
        ax.text(bx + 1.075, 1.02, f"cap_takt = spw/n\n= {cap_takt:.2f}", ha="center", va="top", fontsize=8.2,
                color=STEEL, fontweight="bold")
        if has_plan:
            ax.add_patch(Rectangle((bx + 2.45, 1.15), 2.15, 0.03 + 0.9 * min(demand_takt * maxscale, 1),
                                    facecolor=AMBER, alpha=0.65))
            ax.text(bx + 2.45 + 1.075, 1.02, f"demand_takt = T_avail/q_plan\n= 360/{q_plan} = {demand_takt:.2f}",
                    ha="center", va="top", fontsize=8.2, color=AMBER, fontweight="bold")
        else:
            ax.text(bx + 2.45 + 1.075, 1.02, "계획 없음\n(중간공정 → demand_takt 미사용)", ha="center", va="top",
                    fontsize=8.2, color=GRAY)
        box = Rectangle((bx, 0.05), 4.6, 0.4, facecolor=color, alpha=0.16, edgecolor=color, linewidth=1.4)
        ax.add_patch(box)
        ax.text(cx, 0.25, f"→ eff_takt = max(cap,demand)×wf_unit = {eff:.2f}", ha="center", va="center",
                fontsize=9.8, color=color, fontweight="bold")

    op_block(4.6, "OPER001", 2, 2, 1.0, False, None, None, 1.0, STEEL)
    op_block(14.6, "OPER002", 5, 3, 1.67, True, 50, 7.2, 7.2, ACCENT)
    ax.annotate("", xy=(11.9, 2.24), xytext=(7.3, 2.24),
                arrowprops=dict(arrowstyle="-|>", color=NAVY, linewidth=2.6))
    ax.text(9.6, 2.44, "PPK001 공정 흐름", ha="center", fontsize=10, color=NAVY, fontweight="bold")

    # 하단: 버킷별 prev_takt/post_takt 결과 패널
    def bucket_panel(x, title, prev_src, prev_val, post_src, post_val):
        ax.add_patch(Rectangle((x, -0.95), 8.6, 0.82, facecolor=to_rgba(NAVY, 0.05), edgecolor=NAVY, linewidth=1.3))
        ax.text(x + 0.25, -0.24, title, fontsize=10.4, color=NAVY, fontweight="bold")
        ax.text(x + 0.25, -0.53, f"prev_takt = eff_takt({prev_src}) / 360 = {prev_val}", fontsize=9.4, color=STEEL)
        ax.text(x + 0.25, -0.79, f"post_takt = eff_takt({post_src}) / 360 = {post_val}", fontsize=9.4, color=ACCENT)

    bucket_panel(0.4, "[OPER001 버킷]  (앞 공정 없음 · 뒤는 OPER002)",
                 "flow_prev=없음", "0/360=0.0", "OPER002=7.2", "7.2/360=0.02")
    bucket_panel(10.4, "[OPER002 버킷]  (앞은 OPER001 · 뒤 공정 없음)",
                 "OPER001=1.0", "1/360=0.0028", "flow_post=없음", "0/360=0.0")

    ax.set_xlim(-0.4, 19.4)
    ax.set_ylim(-1.15, 3.55)
    _save(fig, "ch6_7")


# ── ch8 : self_st — 가공시간 비교 막대 ───────────────────────────────────────
def draw_ch8():
    fig, ax = _bare_ax((15.5, 1.8))
    data = [("OPER001", 2, STEEL), ("OPER002 (이 버킷)", 5, RED)]
    bw = 2.6
    for i, (name, v, color) in enumerate(data):
        x = 1.6 + i * 3.4
        h = 1.25 * v / 5
        ax.add_patch(Rectangle((x, 0.2), bw, h, facecolor=color, alpha=0.85))
        ax.text(x + bw / 2, 0.2 + h + 0.09, f"{v}분/장", ha="center", fontsize=11, color=color, fontweight="bold")
        ax.text(x + bw / 2, 0.05, name, ha="center", va="top", fontsize=10, color=NAVY, fontweight="bold")
    ax.text(9.5, 1.15, "max_arrange_st = 5  (전체 (ppk,op,model) 중 최댓값)\n→ self_st = 5/5 = 1.0",
            fontsize=10.4, color=RED, fontweight="bold")
    ax.set_xlim(0, 15.5)
    ax.set_ylim(-0.2, 1.75)
    _save(fig, "ch8")


# ── ch9 : plan_urgency — 게이지 2개 비교 ─────────────────────────────────────
def _gauge(ax, cx, val, vmax, color, label, sub, gw=3.6):
    ax.add_patch(Rectangle((cx - gw / 2, 0.75), gw, 0.36, facecolor=LIGHT, edgecolor=LINE))
    w = gw * min(val / vmax, 1.0)
    ax.add_patch(Rectangle((cx - gw / 2, 0.75), w, 0.36, facecolor=color, alpha=0.9))
    ax.text(cx, 1.3, label, ha="center", fontsize=9.8, color=NAVY, fontweight="bold")
    ax.text(cx, 0.5, sub, ha="center", fontsize=9.0, color=color, fontweight="bold")


def draw_ch9():
    K = 2.0
    fig, ax = _bare_ax((9.6 * K, 1.8))
    _gauge(ax, 2.6 * K, 0.0023, 0.03, GREEN, "T_avail=360 (현재)", "urgency = min((50/360)/60,1) = 0.0023 (여유)")
    _gauge(ax, 7.0 * K, 0.0278, 0.03, RED, "가정: T_avail=30", "urgency = min((50/30)/60,1) = 0.0278 (12배 ↑)")
    ax.text(4.8 * K, 0.08, "같은 gap(=plan_qty-completed=50)·plan_qty=60 인데 남은 시간만 짧아져도 urgency가 뛴다",
            ha="center", fontsize=8.8, color=GRAY)
    ax.set_xlim(0, 9.6 * K)
    ax.set_ylim(0, 1.8)
    _save(fig, "ch9")


# ── ch10_11 : wip_lot_cd / wip_temp — 인코딩 스케일 ──────────────────────────
def draw_ch10_11():
    K = 1.9
    fig, ax = _bare_ax((10.2 * K, 1.8))
    for row, (title, ticks, val, color) in enumerate([
        ("LOT_CD 인덱스 {B:0, A:1}", [("B", 0.0), ("A", 1.0)], 1.0, ACCENT),
        ("TEMP 인덱스 (미사용)", [("None", 0.0)], 0.0, GRAY),
    ]):
        y = 1.15 - row * 0.7
        ax.plot([1.0 * K, 8.5 * K], [y, y], color=LINE, linewidth=3, solid_capstyle="round")
        for name, v in ticks:
            x = (1.0 + 7.5 * v) * K
            ax.plot([x], [y], marker="|", color=NAVY, markersize=16, markeredgewidth=2.2)
            ax.text(x, y - 0.22, name, ha="center", fontsize=8.6, color=NAVY)
        xv = (1.0 + 7.5 * val) * K
        ax.plot([xv], [y], marker="o", color=color, markersize=11, zorder=5)
        ax.text(0.8 * K, y, title, ha="right", va="center", fontsize=9.4, color=NAVY, fontweight="bold")
        ax.text(xv, y + 0.24, f"이 버킷 값 = {val}  (encode_normalized)", ha="center", fontsize=8.8, color=color,
                fontweight="bold")
    ax.set_xlim(0, 10.2 * K)
    ax.set_ylim(-0.1, 1.7)
    _save(fig, "ch10_11")


# ── ch12_13 : needs_conversion / tool_can_assign — 세팅 불일치 카드 ─────────
def draw_ch12_13():
    K = 1.9
    fig, ax = _bare_ax((10.2 * K, 1.8))
    _card(ax, 0.5 * K, 0.3, 2.6 * K, 1.1, "EQP002 현재 세팅", "B", STEEL)
    _card(ax, 3.5 * K, 0.3, 2.6 * K, 1.1, "버킷 요구 세팅", "A", RED)
    ax.text(3.15 * K, 0.85, "≠", ha="center", va="center", fontsize=24, color=RED, fontweight="bold")
    ax.text(2.4 * K, 0.05, "needs_conversion (ch12) = 1.0", ha="center", fontsize=9.0, color=RED, fontweight="bold")

    _card(ax, 7.1 * K, 0.3, 2.6 * K, 1.1, "공구(tool) 배정", "OK", GREEN,
          "needs_tool_swap=False → \"not swap\"=True → OR 만족")
    ax.text(8.4 * K, 0.05, "tool_can_assign (ch13) = 1.0", ha="center", fontsize=9.0, color=GREEN, fontweight="bold")
    ax.set_xlim(0, 10.2 * K)
    ax.set_ylim(0, 1.7)
    _save(fig, "ch12_13")


# ── ch14 : achievable_ratio — 누적 워터폴 막대 ───────────────────────────────
def draw_ch14():
    K = 1.9
    fig, ax = _bare_ax((10.2 * K, 1.9))
    segs = [("완료 done", 10, GRAY), ("이 버킷 WIP", 15, ACCENT), ("상류 OPER001 WIP", 140, GREEN)]
    x = 0.8 * K
    cum = 0
    scale = 8.0 * K / 165
    legend_bits = []
    for name, q, color in segs:
        w = q * scale
        ax.add_patch(Rectangle((x + cum * scale, 0.5), w, 0.5, facecolor=color, alpha=0.9, edgecolor="white"))
        ax.text(x + (cum + q / 2) * scale, 0.75, f"{q}", ha="center", va="center", fontsize=9.2,
                color="white", fontweight="bold")
        legend_bits.append((name, q, color))
        cum += q
    ax.text(x + (cum - 140 / 2) * scale, 1.12, "상류 OPER001 WIP  (flow_prev 역추적으로 합산)", ha="center",
            fontsize=8.8, color=GREEN, fontweight="bold")
    ax.text(x, 1.5, "  +  ".join(f"{n}={q}" for n, q, _ in legend_bits[:2]), fontsize=8.8, color=NAVY,
            fontweight="bold")
    plan_x = x + 60 * scale
    ax.axvline(plan_x, color=RED, linestyle="--", linewidth=1.8, ymin=0.15, ymax=0.75)
    ax.text(plan_x, 0.15, "plan_qty=60", ha="center", fontsize=9.0, color=RED, fontweight="bold")
    ax.text(x + cum * scale + 0.35 * K, 0.75, "→ 누적 165 ≥ 60\nachievable_ratio\n= min(60,165)/60 = 1.0",
            fontsize=9.0, color=NAVY, fontweight="bold", va="center")
    ax.set_xlim(0, 10.2 * K)
    ax.set_ylim(0, 1.9)
    _save(fig, "ch14")


# ── ch15 : projected_cover_ratio — 커버 설비 필터링 ─────────────────────────
def draw_ch15():
    K = 1.9
    fig, ax = _bare_ax((10.2 * K, 1.8))
    _card(ax, 0.5 * K, 0.3, 2.6 * K, 1.1, "EQP001", "제외 X", GRAY, "prev_oper=OPER001≠OPER002")
    _card(ax, 3.5 * K, 0.3, 2.6 * K, 1.1, "EQP003", "포함 OK", GREEN, "prev_oper=OPER002·prev_prod=PPK001 일치")
    ax.annotate("", xy=(9.3 * K, 1.0), xytext=(6.4 * K, 0.9), arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=1.8))
    ax.text(9.5 * K, 1.0, "cov=360/5=72  vs  need=50\nratio = min(72/50,2)/2\n= 0.72", ha="left", va="center",
            fontsize=9.0, color=NAVY, fontweight="bold")
    ax.set_xlim(0, 10.2 * K)
    ax.set_ylim(0, 1.7)
    _save(fig, "ch15")


# ── eqp : needs_conversion / avoidable_frac — 수요 커버 스택 막대 ────────────
def draw_eqp():
    K = 1.9
    fig, ax = _bare_ax((10.2 * K, 1.9))
    x = 0.8 * K
    scale = 8.5 * K / 155
    segs = [("EQP001 대신 커버\n(160/best_st2=80)", 80, GREEN), ("EQP003 대신 커버\n(80/best_st5=16)", 16, GREEN),
            ("EQP002만 가능 (잔여)", 59, RED)]
    cum = 0
    for name, q, color in segs:
        w = q * scale
        alpha = 0.55 if q == 80 else 0.85
        ax.add_patch(Rectangle((x + cum * scale, 0.55), w, 0.55, facecolor=color, alpha=alpha, edgecolor="white"))
        ax.text(x + (cum + q / 2) * scale, 0.82, f"{q}", ha="center", va="center", fontsize=9.2,
                color="white", fontweight="bold")
        ax.text(x + (cum + q / 2) * scale, 1.22, name, ha="center", fontsize=7.6, color=NAVY)
        cum += q
    ax.text(x, 0.2, "세팅A 총수요 = 155  (OPER001 WIP 140 + OPER002 WIP 15)", fontsize=8.8, color=NAVY, fontweight="bold")
    ax.text(x + cum * scale + 0.35 * K, 0.82, "alt_cap=96 → coverage=0.619\n→ avoidable_frac(α)=0.619", fontsize=9.0,
            color=GREEN, fontweight="bold", va="center")
    ax.set_xlim(0, 10.2 * K)
    ax.set_ylim(0, 1.9)
    _save(fig, "eqp")


# ── ctx : 직전 배정 레코드 → 인코딩 카드 4장 ─────────────────────────────────
def draw_ctx():
    K = 1.85
    fig, ax = _bare_ax((10.6 * K, 1.9))
    fields = [
        ("last_ppk\n(prod_idx, P=10)", "PPK001", "1/9=0.111", ACCENT),
        ("last_oper\n(lot_id 역추적)", "OPER002", "1/2=0.5", GREEN),
        ("last_eqp\n(eqp_idx)", "EQP003", "2/2=1.0", STEEL),
        ("last_lot_cd\n(lot_cd_idx)", "세팅 A", "1/1=1.0", RED),
    ]
    w = 2.15 * K
    gap = 0.3 * K
    for i, (title, raw, enc, color) in enumerate(fields):
        x = 0.5 * K + i * (w + gap)
        ax.add_patch(Rectangle((x, 0.7), w, 1.0, facecolor=LIGHT, edgecolor=color, linewidth=1.4))
        ax.text(x + w / 2, 1.66, title, ha="center", va="top", fontsize=8.6, color=NAVY, fontweight="bold")
        ax.text(x + w / 2, 1.0, raw, ha="center", va="center", fontsize=10.5, color=color, fontweight="bold")
        ax.annotate("", xy=(x + w / 2, 0.4), xytext=(x + w / 2, 0.67),
                     arrowprops=dict(arrowstyle="->", color=color, linewidth=1.6))
        ax.text(x + w / 2, 0.2, enc, ha="center", fontsize=9.0, color=color, fontweight="bold")
    ax.text(0.5 * K, 2.0, "직전 배정: EQP003 → PPK001 / OPER002 / 세팅A", fontsize=9.6, color=NAVY, fontweight="bold")
    ax.set_xlim(0, 10.6 * K)
    ax.set_ylim(0, 2.2)
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
