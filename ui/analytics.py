"""
ui/analytics.py – KPI 차트 및 비교 분석 빌더
WIP 수량, 계획 대비 달성률, 공정/제품 전환 횟수 등을 시각화합니다.
"""
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import PROD_COLORS, OPER_BORDER_COLORS
from utils.helpers import build_color_map


# ── WIP 수량 차트 ─────────────────────────────────────────────────────────────

def build_wip_chart(history_snap: dict, plan: List[dict]) -> go.Figure:
    """
    목적: 특정 스텝 시점의 (PLAN_PROD_KEY × OPER) WIP 잔여 수량 막대 차트
    Input:
        history_snap (dict): sim.history[step] 스냅숏
            {"completed": {("PPK001","OPER001"): 25, ...}, ...}
        plan (list): [{"plan_prod_key","oper_id","d0_plan_qty",...}, ...]
    Output:
        go.Figure
    """
    completed = history_snap.get("completed", {})

    labels, remaining_vals, completed_vals = [], [], []
    for p in plan:
        key   = (p["plan_prod_key"], p["oper_id"])
        done  = completed.get(key, 0)
        total = p["d0_plan_qty"]
        rem   = max(total - done, 0)
        label = f"{p['plan_prod_key']}\n{p['oper_id']}"
        labels.append(label)
        remaining_vals.append(rem)
        completed_vals.append(done)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="완료",
        x=labels,
        y=completed_vals,
        marker_color="#55A868",
    ))
    fig.add_trace(go.Bar(
        name="잔여(WIP)",
        x=labels,
        y=remaining_vals,
        marker_color="#C44E52",
    ))
    fig.update_layout(
        title=f"WIP 수량 현황 (스텝 {history_snap.get('step', 0)})",
        xaxis_title="제품 / 공정",
        yaxis_title="웨이퍼 수량 (매)",
        barmode="stack",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.25),
        height=320,
        margin=dict(t=50, b=80),
    )
    return fig


# ── 계획 달성률 차트 ──────────────────────────────────────────────────────────

def build_achievement_chart(history_snap: dict, plan: List[dict]) -> go.Figure:
    """
    목적: 특정 스텝 시점의 (PLAN_PROD_KEY × OPER) 계획 달성률 수평 막대
    Input:
        history_snap (dict): sim.history[step]
        plan (list): 계획 데이터
    Output:
        go.Figure
    """
    completed = history_snap.get("completed", {})

    labels, rates, targets, actuals = [], [], [], []
    for p in plan:
        key   = (p["plan_prod_key"], p["oper_id"])
        done  = completed.get(key, 0)
        target= p["d0_plan_qty"]
        rate  = min(done / max(target, 1) * 100, 100)
        labels.append(f"{p['plan_prod_key']} / {p['oper_id']}")
        rates.append(round(rate, 1))
        targets.append(target)
        actuals.append(done)

    colors = ["#55A868" if r >= 100 else "#DD8452" if r >= 60 else "#C44E52"
              for r in rates]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rates,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{a}/{t}매  ({r}%)" for a, t, r in zip(actuals, targets, rates)],
        textposition="outside",
    ))
    fig.add_vline(x=100, line_dash="dash", line_color="#4C72B0", line_width=1.5)

    fig.update_layout(
        title=f"계획 달성률 (스텝 {history_snap.get('step', 0)})",
        xaxis=dict(title="달성률 (%)", range=[0, 115]),
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=320,
        margin=dict(l=150, r=120, t=50, b=40),
    )
    return fig


# ── 전환 횟수 게이지 ──────────────────────────────────────────────────────────

def build_switch_metrics(history_snap: dict) -> go.Figure:
    """
    목적: 공정/제품 전환 횟수를 인디케이터(게이지)로 표시
    Input:  history_snap (dict): {"oper_sw": int, "prod_sw": int, ...}
    Output: go.Figure
    """
    oper_sw = history_snap.get("oper_sw", 0)
    prod_sw = history_snap.get("prod_sw", 0)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
    )
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=oper_sw,
        title={"text": "공정 전환 횟수"},
        number={"font": {"color": "#C44E52", "size": 40}},
    ), row=1, col=1)
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=prod_sw,
        title={"text": "제품 전환 횟수"},
        number={"font": {"color": "#DD8452", "size": 40}},
    ), row=1, col=2)

    fig.update_layout(
        height=180,
        margin=dict(t=40, b=10),
        paper_bgcolor="white",
    )
    return fig


# ── 초기 vs Post 비교 분석 ────────────────────────────────────────────────────

def _compute_stats(schedule: List[dict], plan: List[dict]) -> dict:
    """
    목적: 스케줄 레코드에서 비교용 KPI 계산
    Input:  schedule, plan
    Output: {"makespan", "idle_total", "oper_switches", "prod_switches",
              "achievement": {label: rate}}
    """
    if not schedule:
        return {"makespan": 0, "idle_total": 0, "oper_switches": 0,
                "prod_switches": 0, "achievement": {}}

    makespan = max(r["END_TM"] for r in schedule)

    # EQP별 전환 횟수 계산
    from collections import defaultdict
    eqp_seq: Dict[str, List[dict]] = defaultdict(list)
    for r in schedule:
        eqp_seq[r["EQP_ID"]].append(r)
    for eqp_id in eqp_seq:
        eqp_seq[eqp_id].sort(key=lambda x: x["START_TM"])

    oper_sw = prod_sw = 0
    idle_total = 0
    for eqp_id, recs in eqp_seq.items():
        for i in range(1, len(recs)):
            if recs[i].get("OPER_ID") != recs[i-1].get("OPER_ID"):
                oper_sw += 1
            if recs[i]["PLAN_PROD_KEY"] != recs[i-1]["PLAN_PROD_KEY"]:
                prod_sw += 1
            idle_total += max(recs[i]["START_TM"] - recs[i-1]["END_TM"], 0)

    # 달성률
    completed: Dict[Tuple, int] = {}
    for r in schedule:
        key = (r["PLAN_PROD_KEY"], r.get("OPER_ID", ""))
        completed[key] = completed.get(key, 0) + r.get("WF_QTY", 25)

    achievement = {}
    for p in plan:
        key   = (p["plan_prod_key"], p["oper_id"])
        done  = completed.get(key, 0)
        label = f"{p['plan_prod_key']}/{p['oper_id']}"
        achievement[label] = round(done / max(p["d0_plan_qty"], 1) * 100, 1)

    return {
        "makespan":      makespan,
        "idle_total":    idle_total,
        "oper_switches": oper_sw,
        "prod_switches": prod_sw,
        "achievement":   achievement,
    }


def build_comparison_kpi(
    initial_schedule: List[dict],
    post_schedule: List[dict],
    plan: List[dict],
) -> go.Figure:
    """
    목적: 초기 스케줄 대비 Post-Scheduling 개선 효과를 KPI 막대로 비교
    Input:
        initial_schedule (list): 초기 스케줄
        post_schedule    (list): RL 결과 스케줄
        plan             (list): 계획 데이터
    Output:
        go.Figure (makespan, idle, 전환 횟수, 달성률 4종 비교)
    """
    init_s = _compute_stats(initial_schedule, plan)
    post_s = _compute_stats(post_schedule,    plan)

    metrics = ["Makespan(분)", "Idle 합계(분)", "공정 전환", "제품 전환"]
    init_vals = [init_s["makespan"], init_s["idle_total"],
                 init_s["oper_switches"], init_s["prod_switches"]]
    post_vals = [post_s["makespan"], post_s["idle_total"],
                 post_s["oper_switches"], post_s["prod_switches"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="초기 스케줄",      x=metrics, y=init_vals,
                         marker_color="#4C72B0"))
    fig.add_trace(go.Bar(name="Post-Scheduling", x=metrics, y=post_vals,
                         marker_color="#55A868"))

    fig.update_layout(
        title="초기 스케줄 vs Post-Scheduling KPI 비교",
        barmode="group",
        yaxis_title="값",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.2),
        height=360,
        margin=dict(t=60, b=80),
    )
    return fig


def build_achievement_comparison(
    initial_schedule: List[dict],
    post_schedule: List[dict],
    plan: List[dict],
) -> go.Figure:
    """
    목적: 초기 vs Post 계획 달성률 비교 차트
    Input/Output: build_comparison_kpi와 동일 형식
    """
    init_s = _compute_stats(initial_schedule, plan)
    post_s = _compute_stats(post_schedule,    plan)

    labels     = sorted(set(init_s["achievement"]) | set(post_s["achievement"]))
    init_rates = [init_s["achievement"].get(l, 0) for l in labels]
    post_rates = [post_s["achievement"].get(l, 0) for l in labels]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="초기 스케줄",      x=labels, y=init_rates,
                         marker_color="#4C72B0"))
    fig.add_trace(go.Bar(name="Post-Scheduling", x=labels, y=post_rates,
                         marker_color="#55A868"))
    fig.add_hline(y=100, line_dash="dash", line_color="red", line_width=1)

    fig.update_layout(
        title="제품/공정별 계획 달성률 비교 (%)",
        barmode="group",
        yaxis=dict(title="달성률 (%)", range=[0, 120]),
        xaxis_title="제품 / 공정",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.25),
        height=360,
        margin=dict(t=60, b=80),
    )
    return fig
