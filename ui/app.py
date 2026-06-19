"""
ui/app.py – Streamlit 메인 애플리케이션
실행: streamlit run ui/app.py
두 가지 모드를 사이드바에서 선택할 수 있습니다.
  1. 학습 모드  – 데이터 로드 → 전처리 → PPO 학습 실행
  2. 추론 모드  – 학습 모델로 Post-Scheduling 수행 + 결과 시각화
"""
import sys
import time
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (streamlit run 위치 무관하게 동작)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from config import CONFIG
from data.loader import load_data, validate_data, generate_sample_data
from data.preprocessor import preprocess
from agent.rl_agent import SchedulingAgent
from inference.runner import run_inference, save_result
from ui.gantt import build_step_gantt, build_comparison_gantt
from ui.analytics import (
    build_wip_chart,
    build_achievement_chart,
    build_switch_metrics,
    build_comparison_kpi,
    build_achievement_comparison,
)

st.set_page_config(
    page_title="Post-Scheduling RL System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── 사이드바 ──────────────────────────────────────────────────────────────────

def sidebar() -> str:
    st.sidebar.title("Post-Scheduling RL")
    st.sidebar.markdown("---")
    mode = st.sidebar.radio(
        "모드 선택",
        ["학습 (Train)", "추론 (Inference)"],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"모델 경로: `{CONFIG.path.model_dir}`")
    st.sidebar.caption(f"입력 경로: `{CONFIG.path.input_dir}`")
    st.sidebar.caption(f"출력 경로: `{CONFIG.path.output_dir}`")
    return mode


# ── 데이터 로딩 (공통) ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_and_preprocess():
    """데이터 로드 + 전처리 (캐시)"""
    raw = load_data()
    errors = validate_data(raw)
    if errors:
        return None, errors
    env_data = preprocess(raw)
    return env_data, []


def load_section() -> dict:
    """사이드바 아래 데이터 로딩 영역 – 샘플 생성 버튼 포함"""
    with st.sidebar.expander("데이터 관리", expanded=False):
        if st.button("샘플 데이터 생성"):
            generate_sample_data()
            st.cache_data.clear()
            st.success("샘플 데이터가 생성되었습니다.")

    with st.spinner("데이터 로딩 중..."):
        try:
            env_data, errors = _load_and_preprocess()
        except FileNotFoundError as e:
            st.error(str(e))
            st.info("사이드바 '샘플 데이터 생성' 버튼을 눌러 테스트 데이터를 먼저 생성하세요.")
            st.stop()

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    return env_data


# ── 학습 모드 ─────────────────────────────────────────────────────────────────

def page_train(env_data: dict):
    st.title("모델 학습")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("학습 파라미터")
        total_ts = st.number_input(
            "Total Timesteps", value=CONFIG.rl.total_timesteps,
            min_value=1000, step=10000,
        )
        lr = st.number_input(
            "Learning Rate", value=CONFIG.rl.learning_rate,
            format="%.5f", step=1e-4,
        )

    with col2:
        st.subheader("보상 가중치")
        w_same_oper = st.slider("동일 OPER 보너스", 0.0, 5.0,
                                CONFIG.reward.w_same_oper, 0.5)
        w_idle      = st.slider("Idle 패널티(분당)", -3.0, 0.0,
                                CONFIG.reward.w_idle_per_min, 0.1)

    # 파라미터 반영
    CONFIG.rl.total_timesteps  = int(total_ts)
    CONFIG.rl.learning_rate    = float(lr)
    CONFIG.reward.w_same_oper  = float(w_same_oper)
    CONFIG.reward.w_idle_per_min = float(w_idle)

    st.markdown("---")

    data_info_col, model_col = st.columns(2)
    with data_info_col:
        st.subheader("입력 데이터 요약")
        st.metric("EQP 수",        len(env_data["eqp_ids"]))
        st.metric("LOT 수",        len(env_data["lots"]))
        st.metric("제품 종류",     len(env_data["prod_keys"]))
        st.metric("공정 종류",     len(env_data["oper_ids"]))
        st.metric("시뮬 종료(분)", env_data["sim_end_minutes"])

    with model_col:
        st.subheader("모델 상태")
        agent = SchedulingAgent()
        model_exists = agent.model_exists()
        if model_exists:
            st.success("저장된 모델이 있습니다.")
        else:
            st.warning("저장된 모델이 없습니다.")

    st.markdown("---")

    if st.button("학습 시작", type="primary"):
        progress_bar = st.progress(0, text="학습 준비 중...")
        log_area = st.empty()

        start_t = time.time()
        with st.spinner("PPO 학습 진행 중..."):
            agent = SchedulingAgent()
            agent.train(env_data, verbose=0)
            agent.save()

        elapsed = time.time() - start_t
        st.success(f"학습 완료! (소요: {elapsed:.1f}초)")
        progress_bar.progress(1.0, text="학습 완료")

        # 간단 평가
        with st.spinner("평가 중..."):
            metrics = agent.evaluate(env_data, n_episodes=3)
        st.subheader("학습 결과 (3 에피소드 평균)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("평균 보상",       f"{metrics['mean_reward']:.1f}")
        c2.metric("공정 전환(평균)", f"{metrics['mean_oper_sw']:.1f}")
        c3.metric("제품 전환(평균)", f"{metrics['mean_prod_sw']:.1f}")
        c4.metric("Idle 합계(평균)", f"{metrics['mean_idle']:.0f}분")


# ── 추론 모드 ─────────────────────────────────────────────────────────────────

def page_inference(env_data: dict):
    st.title("Post-Scheduling 추론 및 시각화")

    agent = SchedulingAgent()
    if not agent.model_exists():
        st.warning("학습된 모델이 없습니다. 먼저 '학습' 모드에서 모델을 생성하세요.")
        st.stop()

    # ── 추론 실행 ─────────────────────────────────────────────────────────
    run_key = "inference_result"
    if st.button("추론 실행", type="primary") or run_key not in st.session_state:
        with st.spinner("Post-Scheduling 추론 중..."):
            loaded_agent = SchedulingAgent.load()
            result = run_inference(env_data, agent=loaded_agent)
            save_result(result)
            st.session_state[run_key] = result
        st.success("추론 완료! 결과가 저장되었습니다.")

    result = st.session_state.get(run_key)
    if not result:
        st.info("추론 버튼을 눌러 결과를 생성하세요.")
        st.stop()

    history  = result["history"]
    schedule = result["schedule"]
    initial  = result["initial_schedule"]
    plan     = result["plan"]
    prod_keys= env_data["prod_keys"]
    oper_ids = env_data["oper_ids"]

    # ── 탭 레이아웃 ──────────────────────────────────────────────────────
    tab_sim, tab_cmp = st.tabs(["시뮬레이션 재생", "초기 vs Post 비교"])

    # ────────────────────────────────────────────────────────────────────
    # TAB 1: 단계별 시뮬레이션 재생
    # ────────────────────────────────────────────────────────────────────
    with tab_sim:
        if not history:
            st.info("히스토리 데이터가 없습니다.")
        else:
            max_step = len(history) - 1
            step = st.slider(
                "시뮬레이션 스텝",
                min_value=0, max_value=max_step, value=0, step=1,
                key="sim_step_slider",
            )
            snap = history[step]

            # ── 간트 차트 ─────────────────────────────────────────────
            st.subheader("설비(EQP) 배정 현황")
            fig_gantt = build_step_gantt(history, step, prod_keys, oper_ids)
            st.plotly_chart(fig_gantt, use_container_width=True)

            st.markdown("---")

            # ── 하단 좌/우 분할 ─────────────────────────────────────
            left, right = st.columns([6, 4])

            with left:
                st.subheader("WIP 수량 현황")
                fig_wip = build_wip_chart(snap, plan)
                st.plotly_chart(fig_wip, use_container_width=True)

            with right:
                st.subheader("계획 달성 현황")
                fig_ach = build_achievement_chart(snap, plan)
                st.plotly_chart(fig_ach, use_container_width=True)

            # ── 공정/제품 전환 카운터 ────────────────────────────────
            st.markdown("---")
            st.subheader("전환 횟수")
            fig_sw = build_switch_metrics(snap)
            st.plotly_chart(fig_sw, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────
    # TAB 2: 초기 vs Post 비교
    # ────────────────────────────────────────────────────────────────────
    with tab_cmp:
        st.subheader("간트 비교")
        fig_cmp_gantt = build_comparison_gantt(initial, schedule, prod_keys, oper_ids)
        st.plotly_chart(fig_cmp_gantt, use_container_width=True)

        st.markdown("---")

        kpi_col, ach_col = st.columns(2)
        with kpi_col:
            st.subheader("KPI 비교")
            fig_kpi = build_comparison_kpi(initial, schedule, plan)
            st.plotly_chart(fig_kpi, use_container_width=True)

        with ach_col:
            st.subheader("계획 달성률 비교")
            fig_ach_cmp = build_achievement_comparison(initial, schedule, plan)
            st.plotly_chart(fig_ach_cmp, use_container_width=True)

        # ── 결과 테이블 ─────────────────────────────────────────────
        st.markdown("---")
        with st.expander("Post-Scheduling 결과 테이블 (전체)", expanded=False):
            import pandas as pd
            df = pd.DataFrame(schedule)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "CSV 다운로드",
                data=csv,
                file_name="post_schedule_result.csv",
                mime="text/csv",
            )


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    mode = sidebar()
    env_data = load_section()

    if mode == "학습 (Train)":
        page_train(env_data)
    else:
        page_inference(env_data)


if __name__ == "__main__":
    main()
