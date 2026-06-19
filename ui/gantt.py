"""
ui/gantt.py – Plotly 기반 간트 차트 빌더
EQP × 시간 축으로 LOT 배정 현황을 시각화합니다.
  - 막대 색상: PLAN_PROD_KEY
  - 막대 테두리: OPER_ID
  - 막대 레이블: LOT_ID
  - 슬라이더 스텝으로 단계별 재생 가능
"""
from typing import Dict, List, Optional

import plotly.graph_objects as go

from config import PROD_COLORS, OPER_BORDER_COLORS
from utils.helpers import build_color_map


def build_gantt(
    schedule: List[dict],
    prod_keys: List[str],
    oper_ids: List[str],
    highlight_step: Optional[int] = None,
    title: str = "스케줄 간트 차트",
    time_label: str = "시뮬레이션 시간 (분)",
) -> go.Figure:
    """
    목적: 스케줄 레코드를 받아 수평 막대(간트) 차트 반환
    Input:
        schedule      (list): [{EQP_ID, LOT_ID, PLAN_PROD_KEY, OPER_ID,
                                START_TM, END_TM, ...}, ...]
        prod_keys     (list): 색상 매핑용 제품 키 전체 목록
        oper_ids      (list): 테두리 색상용 OPER 전체 목록
        highlight_step(int):  이 인덱스 이하의 레코드만 불투명하게 표시
        title         (str):  차트 제목
        time_label    (str):  X축 레이블
    Output:
        go.Figure (Plotly Figure 객체)
    """
    if not schedule:
        fig = go.Figure()
        fig.update_layout(title=title, xaxis_title=time_label)
        return fig

    prod_color_map = build_color_map(prod_keys, PROD_COLORS)
    oper_color_map = build_color_map(oper_ids,  OPER_BORDER_COLORS)
    eqp_ids = sorted({r["EQP_ID"] for r in schedule})

    fig = go.Figure()

    for idx, rec in enumerate(schedule):
        visible = highlight_step is None or idx <= highlight_step
        opacity = 1.0 if visible else 0.15

        width    = rec["END_TM"] - rec["START_TM"]
        prod_col = prod_color_map.get(rec["PLAN_PROD_KEY"], "#888888")
        oper_col = oper_color_map.get(rec.get("OPER_ID", ""), "#222222")
        eqp_y   = rec["EQP_ID"]

        fig.add_trace(go.Bar(
            x=[width],
            y=[eqp_y],
            base=[rec["START_TM"]],
            orientation="h",
            marker=dict(
                color=prod_col,
                opacity=opacity,
                line=dict(color=oper_col, width=3),
            ),
            text=rec["LOT_ID"] if visible else "",
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=(
                f"<b>LOT: {rec['LOT_ID']}</b><br>"
                f"EQP: {rec['EQP_ID']}<br>"
                f"제품: {rec['PLAN_PROD_KEY']}<br>"
                f"공정: {rec.get('OPER_ID','N/A')}<br>"
                f"시작: {rec['START_TM']}분<br>"
                f"종료: {rec['END_TM']}분<br>"
                f"소요: {width}분"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    # 범례용 더미 트레이스 – PLAN_PROD_KEY
    for pk in sorted(prod_keys):
        fig.add_trace(go.Bar(
            x=[0], y=[""],
            orientation="h",
            name=pk,
            marker_color=prod_color_map.get(pk, "#888888"),
            showlegend=True,
            visible="legendonly" if pk not in {r["PLAN_PROD_KEY"] for r in schedule} else True,
        ))

    # OPER 테두리 범례
    for op in sorted(oper_ids):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color=oper_color_map.get(op, "#222222"),
                        symbol="square", line=dict(width=2, color="white")),
            name=f"[OPER] {op}",
            showlegend=True,
        ))

    bar_height = max(40, 60 * len(eqp_ids))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title=time_label, showgrid=True, gridcolor="#E5E5E5"),
        yaxis=dict(
            categoryorder="category ascending",
            title="설비(EQP)",
            tickfont=dict(size=12),
        ),
        barmode="overlay",
        legend=dict(
            title="제품 / 공정",
            orientation="v",
            x=1.02,
        ),
        height=max(350, bar_height),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=180, t=60, b=60),
    )
    return fig


def build_step_gantt(history: List[dict], step: int,
                     prod_keys: List[str], oper_ids: List[str]) -> go.Figure:
    """
    목적: 시뮬레이션 히스토리에서 특정 스텝까지의 배정만 표시하는 간트 반환
    Input:
        history  (list): sim.history 리스트
        step     (int):  표시할 스텝 인덱스 (0-based)
        prod_keys/oper_ids: 색상 팔레트용 전체 목록
    Output:
        go.Figure
    """
    if step < 0 or not history:
        return build_gantt([], prod_keys, oper_ids)

    snap = history[min(step, len(history) - 1)]
    return build_gantt(
        snap["schedule"],
        prod_keys,
        oper_ids,
        highlight_step=len(snap["schedule"]) - 1,
        title=f"Post-Scheduling 간트 (스텝 {snap['step']} / 시각 {snap['time']}분)",
    )


def build_comparison_gantt(
    initial: List[dict],
    post: List[dict],
    prod_keys: List[str],
    oper_ids: List[str],
) -> go.Figure:
    """
    목적: 초기 스케줄과 Post-Scheduling 결과를 좌우로 비교하는 서브플롯 반환
    Input:
        initial  (list): 초기 스케줄 레코드
        post     (list): Post-Scheduling 결과 레코드
        prod_keys/oper_ids: 색상 매핑용
    Output:
        go.Figure (두 간트가 수직 배치된 서브플롯)
    """
    from plotly.subplots import make_subplots

    fig_init = build_gantt(initial, prod_keys, oper_ids, title="초기 스케줄")
    fig_post = build_gantt(post,    prod_keys, oper_ids, title="Post-Scheduling")

    eqp_ids = sorted({r["EQP_ID"] for r in (initial + post)})
    rows = 2

    fig = make_subplots(
        rows=rows, cols=1,
        subplot_titles=["초기 스케줄", "Post-Scheduling (RL 결과)"],
        shared_xaxes=True,
        vertical_spacing=0.12,
    )
    for trace in fig_init.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in fig_post.data:
        fig.add_trace(trace, row=2, col=1)

    fig.update_layout(
        height=max(600, 300 * len(eqp_ids)),
        barmode="overlay",
        showlegend=True,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(x=1.02),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E5E5E5")
    fig.update_yaxes(categoryorder="category ascending")
    return fig
