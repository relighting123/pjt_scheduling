# -*- coding: utf-8 -*-
"""Bulk-Fill 강화학습 스케줄링 — 사무용 발표 자료 생성기."""
import json
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION

# ── 색상 팔레트 (코퍼레이트 네이비) ──────────────────────────────────────────
NAVY   = RGBColor(0x1B, 0x32, 0x55)
ACCENT = RGBColor(0x2E, 0x6F, 0xB0)
STEEL  = RGBColor(0x4A, 0x6D, 0x8C)
LIGHT  = RGBColor(0xEE, 0xF2, 0xF7)
LIGHT2 = RGBColor(0xDD, 0xE6, 0xF0)
INK    = RGBColor(0x22, 0x2A, 0x33)
GRAY   = RGBColor(0x5C, 0x66, 0x70)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
GREEN  = RGBColor(0x2E, 0x7D, 0x4F)
RED    = RGBColor(0xB3, 0x3A, 0x3A)
AMBER  = RGBColor(0xC8, 0x86, 0x1E)
LINE   = RGBColor(0xC4, 0xCF, 0xDB)

FONT = "맑은 고딕"
FONT_L = "맑은 고딕 Semilight"

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
KPI = json.load(open(os.path.join(_HERE, "kpi_conv_bench.json"), encoding="utf-8"))
SUITE = json.load(open(os.path.join(_HERE, "bench_suite_results.json"), encoding="utf-8"))

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height

# ── 저수준 헬퍼 ───────────────────────────────────────────────────────────────
def _set_fill(shape, color):
    shape.fill.solid(); shape.fill.fore_color.rgb = color
    shape.line.fill.background()

def box(slide, x, y, w, h, color=None, line_color=None, line_w=0.75):
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.shadow.inherit = False
    if color is not None:
        sp.fill.solid(); sp.fill.fore_color.rgb = color
    else:
        sp.fill.background()
    if line_color is not None:
        sp.line.color.rgb = line_color; sp.line.width = Pt(line_w)
    else:
        sp.line.fill.background()
    return sp

def txt(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
        space_after=4, line_spacing=1.0, wrap=True):
    """runs: list of paragraphs; each para = list of (text, size, color, bold, font)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(space_after); p.space_before = Pt(0)
        p.line_spacing = line_spacing
        for (t, sz, col, bold, fn) in para:
            r = p.add_run(); r.text = t
            r.font.size = Pt(sz); r.font.color.rgb = col; r.font.bold = bold
            r.font.name = fn
    return tb

def R(t, sz=14, col=INK, bold=False, fn=FONT):
    return (t, sz, col, bold, fn)

def page_header(slide, kicker, title, idx):
    box(slide, 0, 0, 13.333, 1.18, WHITE)
    box(slide, 0.55, 0.34, 0.09, 0.52, ACCENT)
    txt(slide, 0.78, 0.30, 11.0, 0.34, [[R(kicker, 11.5, ACCENT, True)]])
    txt(slide, 0.77, 0.52, 11.6, 0.52, [[R(title, 23, NAVY, True)]])
    box(slide, 0, 1.17, 13.333, 0.022, LIGHT2)
    # footer
    txt(slide, 0.55, 7.06, 9.0, 0.3,
        [[R("Bulk-Fill 강화학습 기반 설비 스케줄링", 9, GRAY)]])
    txt(slide, 11.5, 7.06, 1.28, 0.3,
        [[R(f"{idx:02d}", 9, GRAY, True)]], align=PP_ALIGN.RIGHT)

def content_slide(kicker, title, idx):
    s = prs.slides.add_slide(BLANK)
    box(s, 0, 0, 13.333, 7.5, WHITE)
    page_header(s, kicker, title, idx)
    return s

def chip(slide, x, y, w, h, label, fill, fg=WHITE, sz=12.5, bold=True, line_color=None):
    sp = box(slide, x, y, w, h, fill, line_color=line_color, line_w=1.0)
    tf = sp.text_frame; tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    for i, ln in enumerate(label.split("\n")):
        pp = p if i == 0 else tf.add_paragraph()
        pp.alignment = PP_ALIGN.CENTER
        r = pp.add_run(); r.text = ln
        r.font.size = Pt(sz); r.font.color.rgb = fg; r.font.bold = bold; r.font.name = FONT
    return sp

def arrow(slide, x, y, w, h, color=STEEL, direction=MSO_SHAPE.RIGHT_ARROW):
    sp = slide.shapes.add_shape(direction, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.shadow.inherit = False
    sp.fill.solid(); sp.fill.fore_color.rgb = color; sp.line.fill.background()
    return sp

def bullet(slide, x, y, w, items, gap=0.0, sz=13.5, lead_col=ACCENT, head_col=NAVY,
           body_col=GRAY, line_spacing=1.04):
    """items: list of (head, body) ; head bold colored, body gray."""
    paras = []
    for head, body in items:
        run = [R("▪  ", 13, lead_col, True), R(head, sz, head_col, True)]
        if body:
            run.append(R("   " + body, sz-0.5, body_col, False))
        paras.append(run)
    txt(slide, x, y, w, 2.6, paras, space_after=9, line_spacing=line_spacing)

# ════════════════════════════════════════════════════════════════════════════
# 1. 표지
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
box(s, 0, 0, 13.333, 7.5, NAVY)
box(s, 0, 0, 13.333, 7.5, NAVY)
# 우측 사선 강조 패널
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(9.7), Inches(0), Inches(3.633), Inches(7.5))
band.shadow.inherit = False; band.fill.solid(); band.fill.fore_color.rgb = RGBColor(0x16,0x29,0x46)
band.line.fill.background()
box(s, 0.9, 1.0, 0.12, 1.7, ACCENT)
txt(s, 1.15, 1.0, 9.0, 0.5, [[R("설비 스케줄링 · 강화학습 적용", 14, RGBColor(0x9D,0xBE,0xE0), True)]])
txt(s, 1.13, 1.55, 9.2, 2.0, [
    [R("Bulk-Fill 강화학습 기반", 38, WHITE, True)],
    [R("반도체 설비 스케줄링 모델", 38, WHITE, True)],
], line_spacing=1.05, space_after=2)
box(s, 1.18, 3.95, 5.2, 0.02, RGBColor(0x3A,0x55,0x7A))
txt(s, 1.15, 4.12, 9.5, 1.2, [
    [R("MDP 모델 정의(State · Action · Reward)와", 16, RGBColor(0xC9,0xD8,0xE8))],
    [R("테스트 데이터 기반 알고리즘 KPI 비교 · 효과성 검증", 16, RGBColor(0xC9,0xD8,0xE8))],
], line_spacing=1.2, space_after=3)
txt(s, 1.15, 6.5, 8.0, 0.4, [[R("Project  pjt_scheduling", 12, RGBColor(0x8F,0xA6,0xC0), True)]])
txt(s, 9.9, 6.5, 3.0, 0.4, [[R("2026", 12, RGBColor(0x8F,0xA6,0xC0), True)]], align=PP_ALIGN.RIGHT)

# ════════════════════════════════════════════════════════════════════════════
# 2. 목차
# ════════════════════════════════════════════════════════════════════════════
s = content_slide("CONTENTS", "목차", 2)
agenda = [
    ("01", "문제 정의 및 시스템 구조", "스케줄링 과제 · 전환 비용 · 모듈 아키텍처와 구조적 강점"),
    ("02", "Bulk-Fill MDP 모델 정의", "State · Action · Reward 상세 정의와 마스킹 · 블록 점유 메커니즘"),
    ("03", "보상 산출 로직 예시", "전환 페널티 · 페이싱 · 벌크 블록 보상의 수치 계산 예시"),
    ("04", "알고리즘 KPI 비교 및 효과성 검증", "테스트 데이터 기반 정량 비교 · 학습 모델 효과 입증"),
]
y = 1.65
for no, t, d in agenda:
    box(s, 0.9, y, 0.86, 0.86, NAVY)
    txt(s, 0.9, y, 0.86, 0.86, [[R(no, 22, WHITE, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    txt(s, 1.95, y+0.07, 10.6, 0.45, [[R(t, 18, NAVY, True)]])
    txt(s, 1.97, y+0.5, 10.6, 0.35, [[R(d, 12.5, GRAY)]])
    y += 1.18

# ════════════════════════════════════════════════════════════════════════════
# 3. 문제 정의
# ════════════════════════════════════════════════════════════════════════════
s = content_slide("01  문제 정의 및 시스템 구조", "스케줄링 과제와 전환(Conversion) 비용", 3)
txt(s, 0.9, 1.45, 11.6, 0.6, [[
    R("설비가 유휴 상태가 될 때마다 ", 14, INK),
    R("어떤 제품·공정(PPK·OPER)을 투입할지", 14, NAVY, True),
    R(" 결정하는 순차적 의사결정 문제입니다.", 14, INK),
]])

# 좌: 핵심 난점
box(s, 0.9, 2.25, 5.75, 0.5, NAVY)
txt(s, 0.9, 2.25, 5.75, 0.5, [[R("핵심 난점", 14, WHITE, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
box(s, 0.9, 2.75, 5.75, 2.95, LIGHT)
bullet(s, 1.15, 2.95, 5.3, [
    ("제품 전환마다 셋업 손실", "LOT_CD/TEMP가 바뀌면 전환(conversion) 시간이 발생해 그만큼 가동 시간이 사라집니다."),
    ("계획·재공·설비 제약 동시 고려", "계획량, 가용 재공(WIP), 설비-제품 적합성, 동시 가공 공구(tool) 한도를 함께 만족해야 합니다."),
    ("근시안적 1매 배정의 한계", "유휴 설비에 1캐리어씩만 배정하면 같은 셋업을 길게 이어가지 못해 전환이 잦아집니다."),
], sz=12.8)

# 우: 전환 비용 정량
box(s, 6.95, 2.25, 5.45, 0.5, ACCENT)
txt(s, 6.95, 2.25, 5.45, 0.5, [[R("전환 비용의 정량적 의미", 14, WHITE, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
box(s, 6.95, 2.75, 5.45, 2.95, LIGHT)
txt(s, 7.2, 2.95, 5.0, 0.5, [[R("처리시간 = 전환시간 = 60분, 가동 8시간 기준", 12.5, GRAY, True)]])
# 등식 강조
box(s, 7.2, 3.5, 4.95, 0.95, WHITE, line_color=LINE, line_w=1.0)
txt(s, 7.2, 3.5, 4.95, 0.95, [
    [R("전환 1회  =  캐리어 1개 손실", 17, NAVY, True)],
    [R("설비당 8캐리어 처리 가능 → 전환 n회 시 (8 − n)개", 11.5, GRAY)],
], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.1, space_after=2)
txt(s, 7.2, 4.6, 5.0, 1.0, [[
    R("전환 비용이 ", 12.5, INK),
    R("생산 수량 손실로 직접 환산", 12.5, RED, True),
    R("되므로, 전환 최소화가 곧 처리량 극대화로 이어집니다.", 12.5, INK),
]], line_spacing=1.15)

# 하단 목표 배너
box(s, 0.9, 5.95, 11.5, 0.85, NAVY)
txt(s, 1.2, 5.95, 11.0, 0.85, [[
    R("설계 목표  ", 14, RGBColor(0x9D,0xBE,0xE0), True),
    R("재공이 충분하면 takt에 맞춰 균등 생산하고, 편중되면 몰린 공정에 집중하되 ", 13.5, WHITE),
    R("불필요한 전환은 회피", 13.5, WHITE, True),
]], anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.05)

# ════════════════════════════════════════════════════════════════════════════
# 4. 시스템 구조도
# ════════════════════════════════════════════════════════════════════════════
s = content_slide("01  문제 정의 및 시스템 구조", "모듈 아키텍처 — 데이터에서 추론·시각화까지", 4)

# 파이프라인 5단계
stages = [
    ("data/loader", "데이터 적재·전처리", "Oracle·JSON → env_data\n(arrange·plan·flow·tool)", STEEL),
    ("simulation", "이산사건 시뮬레이터", "DES 엔진·상태/보상\nLOT·EQP 규칙 배정", NAVY),
    ("env", "Gym 환경", "SchedulingEnv\nBulkFillEnv(마스킹)", ACCENT),
    ("agent", "정책·휴리스틱", "PPO(학습)\nMin-Progress·Earliest-ST", NAVY),
    ("inference", "추론·결과 출력", "스케줄·KPI·전환계획\noutput.json / DB", STEEL),
]
x = 0.62; w = 2.28; gap = 0.18; y = 1.7; h = 2.0
for i, (mod, name, desc, col) in enumerate(stages):
    chip(s, x, y, w, 0.5, mod, col, sz=12.5)
    box(s, x, y+0.5, w, h-0.5, LIGHT, line_color=LINE, line_w=1.0)
    txt(s, x+0.08, y+0.62, w-0.16, 0.4, [[R(name, 12.5, NAVY, True)]], align=PP_ALIGN.CENTER)
    txt(s, x+0.08, y+1.08, w-0.16, 0.8,
        [[R(ln, 10.5, GRAY)] for ln in desc.split("\n")],
        align=PP_ALIGN.CENTER, line_spacing=1.05, space_after=1)
    if i < len(stages)-1:
        arrow(s, x+w+0.005, y+0.78, gap+0.02, 0.42, color=LIGHT2)
    x += w + gap

# 하단 보조 모듈
box(s, 0.62, 4.0, 12.1, 0.02, LIGHT2)
sub = [
    ("config.py", "환경 축(O·P·K)·보상 가중치·경로 설정"),
    ("api · frontend", "FastAPI + React 대시보드 · 간트/KPI 시각화"),
    ("benchmark", "직관형 검증 데이터셋(CONV_BENCH) 생성·평가"),
    ("models", "PPO 체크포인트(rl · bulkfill 분리 저장)"),
]
x = 0.62; w = 2.96
for mod, desc in sub:
    box(s, x, 4.25, w, 1.0, WHITE, line_color=LINE, line_w=1.0)
    box(s, x, 4.25, 0.08, 1.0, ACCENT)
    txt(s, x+0.2, 4.36, w-0.3, 0.4, [[R(mod, 12.5, NAVY, True)]])
    txt(s, x+0.2, 4.74, w-0.32, 0.5, [[R(desc, 10.8, GRAY)]], line_spacing=1.05)
    x += w + 0.09

box(s, 0.62, 5.55, 12.1, 1.25, LIGHT)
txt(s, 0.85, 5.66, 11.7, 0.4, [[R("설계 포인트", 13, ACCENT, True)]])
txt(s, 0.85, 6.04, 11.8, 0.75, [[
    R("정책(RL)은 ", 12.5, INK),
    R("(PPK, OPER) 버킷 선택에만 집중", 12.5, NAVY, True),
    R("하고, LOT·EQP 세부 배정과 이산사건 무결성은 시뮬레이터 규칙이 보장합니다. ", 12.5, INK),
    R("행동 공간을 작게 유지", 12.5, NAVY, True),
    R("해 학습이 안정적이며, 시뮬레이터·환경·정책이 느슨하게 결합되어 휴리스틱과 동일 조건에서 비교할 수 있습니다.", 12.5, INK),
]], line_spacing=1.25)

# ════════════════════════════════════════════════════════════════════════════
# 5. 구조적 강점
# ════════════════════════════════════════════════════════════════════════════
s = content_slide("01  문제 정의 및 시스템 구조", "구조적 특징과 강점", 5)
cards = [
    ("이산사건 시뮬레이션 기반", "설비 배정 즉시 busy 전환·종료 이벤트로 진행하는 DES로 실제 라인 동작에 부합하는 일정·전환을 재현합니다."),
    ("벌크 점유(Bulk-Fill) 행동", "정책이 (버킷 + 블록 크기)를 함께 선택해 같은 셋업으로 N캐리어를 연속 점유 → 전환 빈도를 구조적으로 억제합니다."),
    ("행동 마스킹(MaskablePPO)", "유효하지 않은 (PPK·OPER)·블록 크기는 마스크로 차단해 무효 행동 탐색을 제거하고 학습 효율을 높입니다."),
    ("계획·재공 적응형 보상", "고정 계획이 아닌 달성가능 상한(achievable) 기준 takt를 추종해 재공 한계 내에서만 페이싱합니다."),
    ("전환 비용의 정량 모델링", "전환 1회=설비 가동 손실로 환산해 회피가능 전환에 추가 패널티를 부여, 불필요한 셋업 변경을 억제합니다."),
    ("동일 조건 알고리즘 비교", "PPO·휴리스틱이 같은 시뮬레이터·KPI로 평가되어 효과를 객관적으로 검증할 수 있습니다."),
]
x0, y0 = 0.9, 1.55; cw, ch = 3.72, 1.62; gx, gy = 0.18, 0.2
for i, (t, d) in enumerate(cards):
    cx = x0 + (i % 3) * (cw + gx)
    cy = y0 + (i // 3) * (ch + gy)
    box(s, cx, cy, cw, ch, LIGHT, line_color=LINE, line_w=1.0)
    box(s, cx, cy, cw, 0.09, ACCENT)
    txt(s, cx+0.2, cy+0.22, cw-0.4, 0.55, [[R(t, 13.5, NAVY, True)]], line_spacing=1.0)
    txt(s, cx+0.2, cy+0.74, cw-0.38, 0.8, [[R(d, 11, GRAY)]], line_spacing=1.12)

box(s, 0.9, 5.2, 11.5, 1.55, NAVY)
txt(s, 1.2, 5.34, 11.0, 0.4, [[R("요약", 13, RGBColor(0x9D,0xBE,0xE0), True)]])
txt(s, 1.2, 5.7, 11.0, 0.95, [[
    R("작은 행동 공간 · 마스킹 · 벌크 점유", 13.5, WHITE, True),
    R("를 결합해 전환을 줄이면서 계획을 추종하도록 설계했으며, 이산사건 시뮬레이터 위에서 ", 13, RGBColor(0xD7,0xE2,0xEE)),
    R("휴리스틱과 1:1로 비교 가능한 검증 체계", 13.5, WHITE, True),
    R("를 갖췄습니다.", 13, RGBColor(0xD7,0xE2,0xEE)),
]], line_spacing=1.3)

# ════════════════════════════════════════════════════════════════════════════
# 6. MDP 개요
# ════════════════════════════════════════════════════════════════════════════
s = content_slide("02  Bulk-Fill MDP 모델 정의", "강화학습 의사결정 구조 (MDP) 개요", 6)
txt(s, 0.9, 1.4, 11.6, 0.5, [[
    R("매 유휴 설비 결정 시점을 1 스텝으로 보고, ", 13.5, INK),
    R("상태(State) → 행동(Action) → 보상(Reward)", 13.5, NAVY, True),
    R(" 순환으로 정책을 학습합니다.", 13.5, INK),
]])
# 순환 다이어그램
chip(s, 1.1, 2.25, 3.3, 1.15, "정책 (Agent)\nMaskablePPO", NAVY, sz=14)
chip(s, 8.9, 2.25, 3.3, 1.15, "환경 (Environment)\n이산사건 시뮬레이터", ACCENT, sz=14)
arrow(s, 4.5, 2.45, 4.3, 0.4, color=STEEL)
txt(s, 4.5, 2.06, 4.3, 0.35, [[R("행동 a : (버킷, 블록크기)", 11.5, NAVY, True)]], align=PP_ALIGN.CENTER)
arrow(s, 4.5, 3.0, 4.3, 0.4, color=STEEL, direction=MSO_SHAPE.LEFT_ARROW)
txt(s, 4.5, 3.42, 4.3, 0.35, [[R("상태 s′ · 보상 r", 11.5, NAVY, True)]], align=PP_ALIGN.CENTER)

# S/A/R 3열 요약
cols = [
    ("State  (관측)", "2,412차원 벡터", "전역 6 + 버킷 O×P×K×16 + 설비 2 + 맥락 4", STEEL),
    ("Action  (행동)", "MultiDiscrete([O·P, L])", "투입할 (PPK·OPER) 버킷 + 블록 크기 레벨 4단계", ACCENT),
    ("Reward  (보상)", "전환·페이싱·벌크 합산", "전환 회피 + takt 추종 + 큰 블록 점유 유도", NAVY),
]
x = 0.9; w = 3.78
for t, big, d, col in cols:
    box(s, x, 4.1, w, 2.35, LIGHT, line_color=LINE, line_w=1.0)
    box(s, x, 4.1, w, 0.55, col)
    txt(s, x, 4.1, w, 0.55, [[R(t, 14, WHITE, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    txt(s, x+0.18, 4.85, w-0.36, 0.6, [[R(big, 14.5, NAVY, True)]], align=PP_ALIGN.CENTER, line_spacing=1.0)
    box(s, x+0.5, 5.5, w-1.0, 0.02, LINE)
    txt(s, x+0.2, 5.62, w-0.4, 0.8, [[R(d, 11.8, GRAY)]], align=PP_ALIGN.CENTER, line_spacing=1.18)
    x += w + 0.19

# ════════════════════════════════════════════════════════════════════════════
# 7. State 정의
# ════════════════════════════════════════════════════════════════════════════
s = content_slide("02  Bulk-Fill MDP 모델 정의", "State — 관측 공간 정의", 7)
txt(s, 0.9, 1.4, 11.6, 0.5, [[
    R("관측은 [0,1]로 정규화된 ", 13.5, INK),
    R("2,412차원 Box 벡터", 13.5, NAVY, True),
    R(" 입니다.  (O=3, P=10, K=5, 버킷 채널 F=16 기준)", 13, GRAY),
]])
# 공식 박스
box(s, 0.9, 2.05, 11.5, 0.66, NAVY)
txt(s, 0.9, 2.05, 11.5, 0.66, [[
    R("obs_dim = 6 (전역)  +  O×P×K×16 (버킷)  +  2 (현재 설비)  +  4 (직전 맥락)  =  2,412", 14.5, WHITE, True)
]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# 표
rows = [
    ("구성 블록", "차원", "주요 내용", True),
    ("전역 상태 (Global)", "6", "경과시간 · 잔여 takt여유 · 재공 소진율 · 계획 달성률 · 전환대기 설비 비율 · 공구 가동률", False),
    ("버킷 특징 (Bucket)", "2,400", "(OPER×PPK×MODEL) 격자의 16개 채널: 유효성·WIP비중·종료시각·처리율·takt·전환필요·공구여유·달성가능비·투영커버 등", False),
    ("현재 설비 (EQP local)", "2", "현재 결정 설비의 전환 필요 여부 · 전환 회피가능 정도(0~1)", False),
    ("직전 맥락 (Context)", "4", "직전 배정의 PPK · OPER · EQP · LOT_CD (정규화 인덱스)", False),
]
y = 2.95; widths = [3.0, 1.0, 7.5]
for r_i, (c1, c2, c3, is_h) in enumerate(rows):
    h = 0.5 if is_h else (0.95 if r_i == 2 else 0.78)
    x = 0.9
    fill = NAVY if is_h else (LIGHT if r_i % 2 else WHITE)
    for c_i, (cv, cwd) in enumerate(zip((c1, c2, c3), widths)):
        cell = box(s, x, y, cwd, h, fill, line_color=LINE, line_w=0.75)
        tf = cell.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = Inches(0.1); tf.margin_right = Inches(0.08)
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER if (c_i == 1 or is_h) else PP_ALIGN.LEFT
        run = p.add_run(); run.text = cv
        run.font.size = Pt(12 if is_h else (11 if c_i == 2 else 12))
        run.font.bold = is_h or c_i == 0
        run.font.color.rgb = WHITE if is_h else (NAVY if c_i <= 1 else GRAY)
        run.font.name = FONT
        x += cwd
    y += h
txt(s, 0.9, 6.55, 11.5, 0.6, [[
    R("핵심 ", 12.5, ACCENT, True),
    R("버킷 채널이 제품·공정·설비모델별 \"지금 이 버킷을 잡으면 얼마나 유효/시급/중복인가\"를 정책에 제공", 12.5, INK),
    R("하여, 전환 회피와 페이싱 판단의 근거가 됩니다.", 12.5, GRAY),
]], line_spacing=1.1)

# ════════════════════════════════════════════════════════════════════════════
# 8. Action 정의
# ════════════════════════════════════════════════════════════════════════════
s = content_slide("02  Bulk-Fill MDP 모델 정의", "Action — 행동 공간과 블록 점유 메커니즘", 8)
box(s, 0.9, 1.4, 11.5, 0.66, ACCENT)
txt(s, 0.9, 1.4, 11.5, 0.66, [[
    R("action = MultiDiscrete( [ O×P 버킷 ,  L 블록크기 레벨 ] )", 15, WHITE, True)
]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

box(s, 0.9, 2.3, 5.7, 1.5, LIGHT, line_color=LINE, line_w=1.0)
txt(s, 1.1, 2.42, 5.3, 0.4, [[R("① 버킷 선택 (O×P)", 13.5, NAVY, True)]])
txt(s, 1.1, 2.82, 5.35, 0.95, [[
    R("투입할 ", 12, INK), R("(PPK·OPER) 조합", 12, NAVY, True),
    R("을 선택합니다. LOT과 EQP는 우선순위 규칙으로 자동 배정되어 행동 차원을 줄입니다.", 12, GRAY),
]], line_spacing=1.18)

box(s, 6.7, 2.3, 5.7, 1.5, LIGHT, line_color=LINE, line_w=1.0)
txt(s, 6.9, 2.42, 5.3, 0.4, [[R("② 블록 크기 레벨 (L=4)", 13.5, NAVY, True)]])
txt(s, 6.9, 2.82, 5.35, 0.95, [[
    R("같은 셋업으로 ", 12, INK), R("연속 점유할 캐리어 수", 12, NAVY, True),
    R("를 takt 예산 분율로 결정합니다. 레벨↑ → 더 큰 블록.", 12, GRAY),
]], line_spacing=1.18)

# 블록 재생 메커니즘
box(s, 0.9, 4.0, 11.5, 0.45, NAVY)
txt(s, 0.9, 4.0, 11.5, 0.45, [[R("블록 점유(Masked Replay) 동작 순서", 13.5, WHITE, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
steps = [
    ("블록 시작", "버킷·크기 선택\nN 산출, 1캐리어 배정\nremaining = N−1"),
    ("블록 진행", "같은 설비의 다음\n결정은 같은 버킷으로\n강제(마스킹) 재생"),
    ("블록 종료", "remaining=0 또는\n재공·계획·공구 소진\n→ 블록 해제"),
    ("상한 규칙", "N ≤ min(takt예산,\n가용 WIP, 잔여 계획)\n공구는 매 캐리어 검사"),
]
x = 0.9; w = 2.74
for i, (t, d) in enumerate(steps):
    box(s, x, 4.6, w, 1.75, LIGHT, line_color=LINE, line_w=1.0)
    chip(s, x+0.2, 4.78, 0.5, 0.5, str(i+1), ACCENT, sz=15)
    txt(s, x+0.82, 4.82, w-0.95, 0.45, [[R(t, 13, NAVY, True)]])
    txt(s, x+0.22, 5.42, w-0.42, 0.85,
        [[R(ln, 10.8, GRAY)] for ln in d.split("\n")], line_spacing=1.05, space_after=1)
    if i < 3:
        arrow(s, x+w+0.0, 5.35, 0.18, 0.32, color=LIGHT2)
    x += w + 0.18
txt(s, 0.9, 6.5, 11.5, 0.5, [[
    R("효과 ", 12.5, ACCENT, True),
    R("이산사건 무결성을 깨지 않으면서도 한 번의 결정으로 \"같은 셋업 N캐리어 연속\"을 실현 → 전환 횟수를 근본적으로 감소", 12.3, INK),
]], line_spacing=1.1)

# ════════════════════════════════════════════════════════════════════════════
# 9. Reward 정의
# ════════════════════════════════════════════════════════════════════════════
s = content_slide("02  Bulk-Fill MDP 모델 정의", "Reward — 보상 항목과 가중치", 9)
txt(s, 0.9, 1.38, 11.6, 0.45, [[
    R("스텝 보상은 아래 항의 합이며 ", 13.5, INK),
    R("±10 범위로 클리핑", 13.5, NAVY, True),
    R(" 됩니다. (Bulk-Fill 학습 기준 가중치)", 13, GRAY),
]])
rows = [
    ("보상 항", "가중치", "의미 / 발생 조건", None),
    ("전환 패널티  w_conversion", "−10.0", "LOT_CD·TEMP 전환(셋업 변경) 1회마다 부과", RED),
    ("회피가능 전환  w_avoidable_conversion", "−8.0", "다른 무전환 설비가 커버 가능한데도 전환 시 추가 부과(×회피가능비율)", RED),
    ("페이싱  w_pacing", "+2.5", "달성가능 상한 기준 선형 takt에 가까워지면 +, 과생산이면 −", GREEN),
    ("동일 셋업 연속  w_same_setup", "+1.0", "직전과 제품·공정이 모두 같고 재공이 남아있을 때 +", GREEN),
    ("블록 크기 보너스  w_bulk_block_bonus", "+3.0", "큰 블록으로 커밋할수록 + (takt예산 대비 블록 비율)", GREEN),
    ("전용 오용 패널티  w_dedication_misuse", "−4.0", "더 전용적인 유휴 설비가 있는데 범용 설비가 그 버킷을 잡으면 −", RED),
    ("중복 커버 패널티  w_redundant_cover", "−5.0", "이미 다른 설비가 충분히 덮는 버킷을 또 잡으면 − (전환 유도)", RED),
]
y = 1.95; widths = [4.35, 1.35, 5.8]
for r_i, (c1, c2, c3, tag) in enumerate(rows):
    is_h = (r_i == 0)
    h = 0.46 if is_h else 0.605
    x = 0.9
    fill = NAVY if is_h else (LIGHT if r_i % 2 else WHITE)
    vals = (c1, c2, c3)
    for c_i, (cv, cwd) in enumerate(zip(vals, widths)):
        cell = box(s, x, y, cwd, h, fill, line_color=LINE, line_w=0.75)
        tf = cell.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = Inches(0.1); tf.margin_right = Inches(0.08)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER if (c_i == 1) else PP_ALIGN.LEFT
        run = p.add_run(); run.text = cv
        run.font.size = Pt(11.5 if is_h else (11.5 if c_i < 2 else 10.6))
        run.font.bold = is_h or c_i <= 1
        if is_h:
            run.font.color.rgb = WHITE
        elif c_i == 1:
            run.font.color.rgb = tag if tag else NAVY
        elif c_i == 0:
            run.font.color.rgb = NAVY
        else:
            run.font.color.rgb = GRAY
        run.font.name = FONT
        x += cwd
    y += h
txt(s, 0.9, 6.62, 11.6, 0.5, [[
    R("위 3개 벌크 전용 항(블록·전용오용·중복커버)은 Bulk-Fill 환경에서만 활성화되어 ", 11.8, INK),
    R("\"큰 블록으로 전담\"", 11.8, NAVY, True),
    R(" 행동을 유도합니다.", 11.8, GRAY),
]], line_spacing=1.1)

# ════════════════════════════════════════════════════════════════════════════
# 10. Reward 산출 예시
# ════════════════════════════════════════════════════════════════════════════
s = content_slide("03  보상 산출 로직 예시", "수치 예시 — 한 스텝의 보상은 이렇게 계산됩니다", 10)
txt(s, 0.9, 1.35, 11.6, 0.42, [[
    R("검증 데이터(3설비·3제품·각 8캐리어, takt 예산 8) 기준, 대표 결정 두 가지의 보상 계산입니다.", 12.8, GRAY)
]])

# 예시 A: 좋은 결정 (큰 블록·전담)
box(s, 0.9, 1.95, 5.7, 4.05, LIGHT, line_color=LINE, line_w=1.0)
box(s, 0.9, 1.95, 5.7, 0.5, GREEN)
txt(s, 0.9, 1.95, 5.7, 0.5, [[R("예시 A · 전담 블록 시작 (바람직)", 13, WHITE, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
txt(s, 1.12, 2.6, 5.3, 0.55, [[
    R("상황 ", 11.5, NAVY, True),
    R("EQP001이 직전과 같은 PPK001을 블록 크기 8(=takt 예산 전량)로 커밋, 다른 설비가 덮지 않는 제품", 11, GRAY),
]], line_spacing=1.12)
calcA = [
    ("동일 셋업 연속", "+1.0", "제품·공정 동일, 재공 잔존"),
    ("페이싱", "+2.5 ×Δ", "takt 목표선에 근접"),
    ("블록 보너스", "+3.0 ×(8/8)", "= +3.0  (전량 블록)"),
    ("전환 패널티", "0", "셋업 변경 없음"),
    ("중복 커버", "0", "다른 설비가 안 덮음"),
]
yy = 3.25
for a, b, c in calcA:
    txt(s, 1.12, yy, 1.9, 0.34, [[R(a, 11, INK, True)]])
    txt(s, 3.0, yy, 1.25, 0.34, [[R(b, 11, GREEN, True)]])
    txt(s, 4.2, yy, 2.3, 0.34, [[R(c, 10, GRAY)]])
    yy += 0.42
box(s, 1.12, 5.42, 5.26, 0.02, LINE)
txt(s, 1.12, 5.5, 5.26, 0.45, [[
    R("합계  ", 12.5, NAVY, True), R("≈ +6 수준의 강한 양(+) 보상", 13, GREEN, True)
]])
txt(s, 1.12, 5.95, 5.3, 0.45, [[R("→ 같은 셋업으로 길게 점유하는 행동을 강화", 10.8, GRAY)]])

# 예시 B: 나쁜 결정 (회피가능 전환)
box(s, 6.7, 1.95, 5.7, 4.05, LIGHT, line_color=LINE, line_w=1.0)
box(s, 6.7, 1.95, 5.7, 0.5, RED)
txt(s, 6.7, 1.95, 5.7, 0.5, [[R("예시 B · 회피가능 전환 (지양)", 13, WHITE, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
txt(s, 6.92, 2.6, 5.3, 0.55, [[
    R("상황 ", 11.5, NAVY, True),
    R("EQP가 PPK001→PPK002로 셋업을 바꿔 1캐리어만 처리. 다른 무전환 설비가 PPK002를 70% 커버 가능", 11, GRAY),
]], line_spacing=1.12)
calcB = [
    ("전환 패널티", "−10.0", "LOT_CD 변경 1회"),
    ("회피가능 전환", "−8.0 ×0.7", "= −5.6  (커버 가능)"),
    ("중복 커버", "−5.0 ×비율", "이미 덮이는 버킷"),
    ("블록 보너스", "0", "블록 크기 1"),
    ("동일 셋업", "0", "셋업이 바뀜"),
]
yy = 3.25
for a, b, c in calcB:
    txt(s, 6.92, yy, 1.9, 0.34, [[R(a, 11, INK, True)]])
    txt(s, 8.8, yy, 1.4, 0.34, [[R(b, 11, RED, True)]])
    txt(s, 10.15, yy, 2.25, 0.34, [[R(c, 10, GRAY)]])
    yy += 0.42
box(s, 6.92, 5.42, 5.26, 0.02, LINE)
txt(s, 6.92, 5.5, 5.26, 0.45, [[
    R("합계  ", 12.5, NAVY, True), R("≈ −20 수준의 강한 음(−) 보상", 13, RED, True)
]])
txt(s, 6.92, 5.95, 5.3, 0.45, [[R("→ 불필요한 셋업 변경을 강하게 억제", 10.8, GRAY)]])

box(s, 0.9, 6.25, 11.5, 0.7, NAVY)
txt(s, 1.15, 6.25, 11.1, 0.7, [[
    R("학습 결과  ", 13, RGBColor(0x9D,0xBE,0xE0), True),
    R("정책은 \"전담 블록은 키우고, 회피가능한 전환은 피한다\"는 행동을 자연스럽게 획득합니다.", 13, WHITE),
]], anchor=MSO_ANCHOR.MIDDLE)

# ════════════════════════════════════════════════════════════════════════════
# 11. 검증 설계
# ════════════════════════════════════════════════════════════════════════════
s = content_slide("04  알고리즘 KPI 비교 및 효과성 검증", "검증 설계 — CONV_BENCH 데이터셋", 11)
txt(s, 0.9, 1.4, 11.6, 0.5, [[
    R("전환 비용이 ", 13.5, INK), R("생산 수량 손실로 1:1 환산", 13.5, NAVY, True),
    R("되도록 설계한 직관형 검증 데이터셋입니다.", 13.5, INK),
]])
specs = [
    ("설비 / 제품", "3 설비 × 3 제품(PPK)", "동일 성능 A모델 3대, 제품별 LOT_CD 상이"),
    ("재공 / 계획", "제품당 8캐리어 = 24", "계획도 제품당 8 (총 24캐리어)"),
    ("시간 파라미터", "ST=60 · 전환=60 · 8h", "설비당 480/60 = 8캐리어 처리 가능"),
    ("배정 난이도", "홈 LOT 혼합 배치", "단순 규칙은 전환 불가피, 전략적 선택만 전담 달성"),
]
x = 0.9; w = 2.83
for t, big, d in specs:
    box(s, x, 2.1, w, 1.7, LIGHT, line_color=LINE, line_w=1.0)
    box(s, x, 2.1, w, 0.09, ACCENT)
    txt(s, x+0.15, 2.28, w-0.3, 0.35, [[R(t, 11.5, ACCENT, True)]])
    txt(s, x+0.15, 2.66, w-0.3, 0.5, [[R(big, 13, NAVY, True)]], line_spacing=1.0)
    txt(s, x+0.15, 3.18, w-0.28, 0.6, [[R(d, 10.5, GRAY)]], line_spacing=1.1)
    x += w + 0.12

box(s, 0.9, 4.1, 5.7, 0.5, NAVY)
txt(s, 0.9, 4.1, 5.7, 0.5, [[R("이론적 최적해", 13, WHITE, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
box(s, 0.9, 4.6, 5.7, 1.95, LIGHT)
txt(s, 1.15, 4.75, 5.3, 1.8, [
    [R("설비별 1제품 전담 → 전환 0회", 12.5, NAVY, True)],
    [R("EQP001 → PPK001  (8캐리어)", 11.5, GRAY)],
    [R("EQP002 → PPK002  (8캐리어)", 11.5, GRAY)],
    [R("EQP003 → PPK003  (8캐리어)", 11.5, GRAY)],
    [R("생산 24 · 전환 0 · 가동률 100%", 12.5, GREEN, True)],
], line_spacing=1.15, space_after=4)

box(s, 6.95, 4.1, 5.45, 0.5, ACCENT)
txt(s, 6.95, 4.1, 5.45, 0.5, [[R("전환 → 손실 환산표", 13, WHITE, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
box(s, 6.95, 4.6, 5.45, 1.95, LIGHT)
conv_rows = [("전환 횟수", "생산 수량", "손실"),
             ("0 회", "24 개", "0"),
             ("2 회", "22 개", "−2"),
             ("6 회", "18 개", "−6")]
yy = 4.72; rh = 0.44
for i, (a, b, c) in enumerate(conv_rows):
    ish = (i == 0)
    fill = NAVY if ish else (WHITE if i % 2 else LIGHT2)
    xx = 7.15
    for j, (v, cw2) in enumerate(zip((a, b, c), (2.0, 1.9, 1.2))):
        cell = box(s, xx, yy, cw2, rh, fill, line_color=LINE, line_w=0.6)
        tf = cell.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = v
        r.font.size = Pt(11 if not ish else 11); r.font.bold = ish or j == 0
        r.font.color.rgb = WHITE if ish else (NAVY if j == 0 else (RED if j == 2 and i > 0 else GRAY))
        r.font.name = FONT
        xx += cw2
    yy += rh
txt(s, 0.9, 6.7, 11.5, 0.4, [[
    R("동일 시뮬레이터·동일 데이터에서 세 알고리즘을 실행해 KPI를 직접 비교합니다.", 12, INK)
]])

# ════════════════════════════════════════════════════════════════════════════
# 12. KPI 비교 표
# ════════════════════════════════════════════════════════════════════════════
s = content_slide("04  알고리즘 KPI 비교 및 효과성 검증", "KPI 비교 결과 (CONV_BENCH)", 12)
mp, es, bf = KPI["minprogress"], KPI["earliest_st"], KPI["bulkfill"]
header = ["KPI 지표", "Earliest-ST\n(단순 규칙)", "Min-Progress\n(휴리스틱)", "Bulk-Fill PPO\n(학습 모델)"]
data_rows = [
    ("생산 수량 (최대 24)", f"{es['prod']} 개", f"{mp['prod']} 개", f"{bf['prod']} 개"),
    ("전환 횟수 (낮을수록 우수)", f"{es['conv']} 회", f"{mp['conv']} 회", f"{bf['conv']} 회"),
    ("전환 손실 캐리어", f"−{es['loss']} 개", f"−{mp['loss']} 개", f"−{bf['loss']} 개"),
    ("설비 가동률", f"{es['util']:.0f} %", f"{mp['util']:.0f} %", f"{bf['util']:.0f} %"),
    ("제품 전담 설비", f"{es['ded']} / 3", f"{mp['ded']} / 3", f"{bf['ded']} / 3"),
    ("최적해 도달", "미달", "달성", "달성"),
]
x0 = 0.9; col_w = [3.5, 2.85, 2.85, 2.85]; y = 1.55
# header
xx = x0
for j, htxt in enumerate(header):
    fill = NAVY if j < 3 else ACCENT
    if j == 0: fill = NAVY
    cell = box(s, xx, y, col_w[j], 0.82, NAVY if j != 3 else ACCENT, line_color=WHITE, line_w=1.0)
    tf = cell.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True
    for k, ln in enumerate(htxt.split("\n")):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = ln
        r.font.size = Pt(12.5 if k == 0 else 10.5)
        r.font.bold = True; r.font.color.rgb = WHITE; r.font.name = FONT
    xx += col_w[j]
y += 0.82
for i, (label, v1, v2, v3) in enumerate(data_rows):
    rh = 0.69
    xx = x0
    vals = [label, v1, v2, v3]
    for j, v in enumerate(vals):
        if j == 0:
            fill = LIGHT2
        elif j == 3:
            fill = RGBColor(0xE2,0xEE,0xE7)  # 연녹 강조열
        else:
            fill = WHITE if i % 2 == 0 else LIGHT
        cell = box(s, xx, y, col_w[j], rh, fill, line_color=LINE, line_w=0.75)
        tf = cell.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True
        tf.margin_left = Inches(0.12)
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
        r = p.add_run(); r.text = v
        r.font.name = FONT
        r.font.size = Pt(12 if j == 0 else 13)
        r.font.bold = (j == 0) or (j == 3)
        # 색상 규칙
        if j == 0:
            r.font.color.rgb = NAVY
        elif label.startswith("최적해"):
            r.font.color.rgb = GREEN if v == "달성" else RED
            r.font.bold = True
        elif j == 1 and (label.startswith("전환") or label.startswith("전환 손실")):
            r.font.color.rgb = RED
        elif j == 3:
            r.font.color.rgb = GREEN
        else:
            r.font.color.rgb = GRAY
        xx += col_w[j]
    y += rh
txt(s, 0.9, 6.7, 11.6, 0.45, [[
    R("Bulk-Fill PPO는 학습만으로 최고 휴리스틱과 동일한 이론적 최적해(생산 24·전환 0)에 도달", 12.5, NAVY, True),
    R("했으며, 단순 규칙 대비 모든 지표에서 우위입니다.", 12.5, INK),
]], line_spacing=1.1)

# ════════════════════════════════════════════════════════════════════════════
# 13. 효과성 차트
# ════════════════════════════════════════════════════════════════════════════
s = content_slide("04  알고리즘 KPI 비교 및 효과성 검증", "효과성 검증 — 정량 비교", 13)

# 차트 1: 생산 수량
cd1 = CategoryChartData()
cd1.categories = ["Earliest-ST", "Min-Progress", "Bulk-Fill PPO"]
cd1.add_series("생산 수량", (es['prod'], mp['prod'], bf['prod']))
gframe = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,
    Inches(0.7), Inches(1.55), Inches(4.0), Inches(3.7), cd1)
ch1 = gframe.chart; ch1.has_legend = False; ch1.has_title = True
ch1.chart_title.text_frame.text = "생산 수량 (개, 최대 24)"
ch1.chart_title.text_frame.paragraphs[0].font.size = Pt(12.5)
ch1.chart_title.text_frame.paragraphs[0].font.bold = True
ch1.chart_title.text_frame.paragraphs[0].font.color.rgb = NAVY
ch1.chart_title.text_frame.paragraphs[0].font.name = FONT
plot1 = ch1.plots[0]; plot1.has_data_labels = True
plot1.data_labels.font.size = Pt(11); plot1.data_labels.font.bold = True
plot1.data_labels.font.color.rgb = NAVY; plot1.data_labels.font.name = FONT
plot1.gap_width = 80
pts1 = plot1.series[0].points
pts1[0].format.fill.solid(); pts1[0].format.fill.fore_color.rgb = RGBColor(0xC2,0x8A,0x8A)
pts1[1].format.fill.solid(); pts1[1].format.fill.fore_color.rgb = STEEL
pts1[2].format.fill.solid(); pts1[2].format.fill.fore_color.rgb = ACCENT
va = ch1.value_axis; va.minimum_scale = 0; va.maximum_scale = 26
va.tick_labels.font.size = Pt(9); va.tick_labels.font.name = FONT
ch1.category_axis.tick_labels.font.size = Pt(9.5)
ch1.category_axis.tick_labels.font.name = FONT

# 차트 2: 전환 횟수
cd2 = CategoryChartData()
cd2.categories = ["Earliest-ST", "Min-Progress", "Bulk-Fill PPO"]
cd2.add_series("전환 횟수", (es['conv'], mp['conv'], bf['conv']))
gframe2 = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,
    Inches(4.75), Inches(1.55), Inches(4.0), Inches(3.7), cd2)
ch2 = gframe2.chart; ch2.has_legend = False; ch2.has_title = True
ch2.chart_title.text_frame.text = "전환 횟수 (회, 낮을수록 우수)"
ch2.chart_title.text_frame.paragraphs[0].font.size = Pt(12.5)
ch2.chart_title.text_frame.paragraphs[0].font.bold = True
ch2.chart_title.text_frame.paragraphs[0].font.color.rgb = NAVY
ch2.chart_title.text_frame.paragraphs[0].font.name = FONT
plot2 = ch2.plots[0]; plot2.has_data_labels = True
plot2.data_labels.font.size = Pt(11); plot2.data_labels.font.bold = True
plot2.data_labels.font.color.rgb = NAVY; plot2.data_labels.font.name = FONT
plot2.gap_width = 80
pts2 = plot2.series[0].points
pts2[0].format.fill.solid(); pts2[0].format.fill.fore_color.rgb = RED
pts2[1].format.fill.solid(); pts2[1].format.fill.fore_color.rgb = STEEL
pts2[2].format.fill.solid(); pts2[2].format.fill.fore_color.rgb = ACCENT
va2 = ch2.value_axis; va2.minimum_scale = 0; va2.maximum_scale = 8
va2.tick_labels.font.size = Pt(9); va2.tick_labels.font.name = FONT
ch2.category_axis.tick_labels.font.size = Pt(9.5)
ch2.category_axis.tick_labels.font.name = FONT

# 우측 효과 요약
box(s, 9.0, 1.55, 3.55, 3.7, LIGHT, line_color=LINE, line_w=1.0)
box(s, 9.0, 1.55, 3.55, 0.5, NAVY)
txt(s, 9.0, 1.55, 3.55, 0.5, [[R("학습 모델 효과", 13, WHITE, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
gain_prod = bf['prod'] - es['prod']
gain_pct = round(100*gain_prod/max(es['prod'],1))
effs = [
    (f"+{gain_prod} 개", f"단순 규칙 대비 생산량 (+{gain_pct}%)"),
    (f"−{es['conv']} 회", "전환을 전량 제거 (6→0)"),
    (f"+{bf['util']-es['util']:.0f}%p", f"가동률 {es['util']:.0f}%→{bf['util']:.0f}%"),
    ("3 / 3", "제품 전담 설비 달성"),
]
yy = 2.2
for big, d in effs:
    txt(s, 9.25, yy, 3.1, 0.4, [[R(big, 18, ACCENT, True)]])
    txt(s, 9.25, yy+0.42, 3.15, 0.32, [[R(d, 10.8, GRAY)]])
    yy += 0.78

box(s, 0.7, 5.5, 11.85, 1.3, NAVY)
txt(s, 1.0, 5.62, 11.4, 0.4, [[R("효과성 결론", 13, RGBColor(0x9D,0xBE,0xE0), True)]])
txt(s, 1.0, 5.98, 11.5, 0.75, [[
    R("Bulk-Fill PPO는 별도 규칙 주입 없이 ", 13, RGBColor(0xD7,0xE2,0xEE)),
    R("학습만으로 이론적 최적해에 도달", 13.5, WHITE, True),
    R("했으며, 단순 규칙 대비 ", 13, RGBColor(0xD7,0xE2,0xEE)),
    R("생산량 +33% · 전환 100% 제거 · 가동률 +25%p", 13.5, WHITE, True),
    R(" 의 개선을 입증했습니다.", 13, RGBColor(0xD7,0xE2,0xEE)),
]], line_spacing=1.25)

# ════════════════════════════════════════════════════════════════════════════
# 14. 다양한 테스트 데이터셋 — 상세 결과표
# ════════════════════════════════════════════════════════════════════════════
s = content_slide("04  알고리즘 KPI 비교 및 효과성 검증", "다양한 테스트 데이터셋 10종 — 일반화 검증", 14)
txt(s, 0.9, 1.32, 11.6, 0.45, [[
    R("설비·제품 수(2×2~6×6), 처리시간(30~72분), 전환시간(30~90분), 혼합도를 달리한 ", 12, INK),
    R("10종 데이터셋", 12, NAVY, True),
    R("에 ", 12, INK),
    R("단일 Bulk-Fill 모델을 공동 학습", 12, NAVY, True),
    R(" 후 평가했습니다.", 12, INK),
]])
ds = SUITE["datasets"]
# 표 헤더
hdr = ["데이터셋", "구성", "최대", "Earliest-ST\n생산 / 전환", "Min-Progress\n생산", "Bulk-Fill\n생산 / 전환 / 전담"]
cw = [2.35, 1.55, 0.85, 2.45, 1.75, 2.95]
y = 1.9; x0 = 0.62
xx = x0
for j, h in enumerate(hdr):
    fill = ACCENT if j == 5 else NAVY
    cell = box(s, xx, y, cw[j], 0.62, fill, line_color=WHITE, line_w=1.0)
    tf = cell.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True
    for k, ln in enumerate(h.split("\n")):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = ln
        r.font.size = Pt(10.5 if k == 0 else 9); r.font.bold = True
        r.font.color.rgb = WHITE; r.font.name = FONT
    xx += cw[j]
y += 0.62
short = {"BENCH01_3x3_st60":"BENCH01","BENCH02_4x4_st60":"BENCH02","BENCH03_5x5_st45":"BENCH03",
         "BENCH04_2x2_st40":"BENCH04","BENCH05_3x3_st48":"BENCH05","BENCH06_6x6_st72":"BENCH06",
         "BENCH07_4x4_cv50":"BENCH07","BENCH08_3x3_st30":"BENCH08","BENCH09_5x5_cv90":"BENCH09",
         "BENCH10_4x4_st36":"BENCH10"}
rh = 0.4
for i, r in enumerate(ds):
    a = r["algos"]; es = a["earliest_st"]; mp = a["minprogress"]; bf = a["bulkfill"]
    cfg = f"{r['n']}×{r['n']}·ST{r['st']}"
    bf_opt = (bf["conv"] == 0 and bf["prod"] == bf["max"])
    cells = [
        (short.get(r["name"], r["name"]), "L", NAVY, True),
        (cfg, "C", GRAY, False),
        (str(r["total"]), "C", GRAY, False),
        (f"{es['prod']} / {es['conv']}", "C", RED, False),
        (f"{mp['prod']}", "C", GRAY, False),
        (f"{bf['prod']} / {bf['conv']} / {bf['ded']}/{bf['n_eqp']}", "C", GREEN if bf_opt else AMBER, True),
    ]
    xx = x0
    for j, (v, al, col, bd) in enumerate(cells):
        if j == 5:
            fill = RGBColor(0xE2,0xEE,0xE7) if bf_opt else RGBColor(0xF6,0xEE,0xDD)
        else:
            fill = WHITE if i % 2 == 0 else LIGHT
        cell = box(s, xx, y, cw[j], rh, fill, line_color=LINE, line_w=0.6)
        tf = cell.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True
        tf.margin_left = Inches(0.08); tf.margin_top = Inches(0.0); tf.margin_bottom = Inches(0.0)
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if al == "L" else PP_ALIGN.CENTER
        rn = p.add_run(); rn.text = v
        rn.font.size = Pt(10); rn.font.bold = bd; rn.font.color.rgb = col; rn.font.name = FONT
        xx += cw[j]
    y += rh
txt(s, 0.62, 6.45, 12.1, 0.7, [[
    R("Bulk-Fill는 별도 규칙 주입 없이 ", 11.5, INK),
    R("10종 중 9종에서 이론 최적(전환 0·전량 생산)에 도달", 11.5, GREEN, True),
    R("했고, 나머지 1종(6×6)도 최고 수준에 근접했습니다. (녹색=최적, 주황=근접)", 11.5, GRAY),
]], line_spacing=1.1)

# ════════════════════════════════════════════════════════════════════════════
# 15. 다양한 테스트 데이터셋 — 집계 효과
# ════════════════════════════════════════════════════════════════════════════
s = content_slide("04  알고리즘 KPI 비교 및 효과성 검증", "10종 데이터셋 종합 — 효과성 집계", 15)
sm = SUITE["summary"]; es = sm["earliest_st"]; mp = sm["minprogress"]; bf = sm["bulkfill"]

# 좌측: 종합 차트(생산률·가동률)
cd = CategoryChartData()
cd.categories = ["Earliest-ST", "Min-Progress", "Bulk-Fill"]
cd.add_series("생산률(%)", (es["prod_pct"], mp["prod_pct"], bf["prod_pct"]))
cd.add_series("평균 가동률(%)", (es["util"], mp["util"], bf["util"]))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,
    Inches(0.7), Inches(1.55), Inches(6.5), Inches(3.55), cd)
ch = gf.chart; ch.has_title = True
ch.chart_title.text_frame.text = "10종 종합 — 생산률 · 평균 가동률"
ch.chart_title.text_frame.paragraphs[0].font.size = Pt(13)
ch.chart_title.text_frame.paragraphs[0].font.bold = True
ch.chart_title.text_frame.paragraphs[0].font.color.rgb = NAVY
ch.chart_title.text_frame.paragraphs[0].font.name = FONT
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(10); ch.legend.font.name = FONT
plot = ch.plots[0]; plot.has_data_labels = True; plot.gap_width = 90
plot.data_labels.font.size = Pt(9.5); plot.data_labels.font.bold = True
plot.data_labels.font.color.rgb = NAVY; plot.data_labels.font.name = FONT
plot.data_labels.number_format = '0"%"'; plot.data_labels.number_format_is_linked = False
plot.series[0].format.fill.solid(); plot.series[0].format.fill.fore_color.rgb = ACCENT
plot.series[1].format.fill.solid(); plot.series[1].format.fill.fore_color.rgb = STEEL
va = ch.value_axis; va.minimum_scale = 0; va.maximum_scale = 110
va.tick_labels.font.size = Pt(9); va.tick_labels.font.name = FONT
ch.category_axis.tick_labels.font.size = Pt(10); ch.category_axis.tick_labels.font.name = FONT

# 우측: 핵심 집계 지표 카드
box(s, 7.5, 1.55, 5.05, 3.55, LIGHT, line_color=LINE, line_w=1.0)
box(s, 7.5, 1.55, 5.05, 0.5, NAVY)
txt(s, 7.5, 1.55, 5.05, 0.5, [[R("종합 집계 (10종 합산)", 13, WHITE, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
metrics = [
    ("총 생산률", f"{es['prod_pct']:.0f}%", f"{bf['prod_pct']:.0f}%", f"{mp['prod_pct']:.0f}%"),
    ("총 전환 횟수", f"{es['conv']}", f"{bf['conv']}", f"{mp['conv']}"),
    ("평균 가동률", f"{es['util']:.0f}%", f"{bf['util']:.0f}%", f"{mp['util']:.0f}%"),
    ("전담 달성률", f"{es['ded_pct']:.0f}%", f"{bf['ded_pct']:.0f}%", f"{mp['ded_pct']:.0f}%"),
    ("최적 도달", f"{es['optimal']}/10", f"{bf['optimal']}/10", f"{mp['optimal']}/10"),
]
# 헤더행
hy = 2.18
for lbl, xx, w in [("지표", 7.7, 2.0), ("Earliest", 9.65, 1.0), ("Bulk-Fill", 10.62, 1.05), ("Min-Prog", 11.62, 0.85)]:
    txt(s, xx, hy, w, 0.3, [[R(lbl, 10, ACCENT, True)]], align=PP_ALIGN.LEFT if lbl=="지표" else PP_ALIGN.CENTER)
yy = 2.5
for name, e, b, mv in metrics:
    txt(s, 7.7, yy, 2.0, 0.34, [[R(name, 11, NAVY, True)]])
    txt(s, 9.5, yy, 1.05, 0.34, [[R(e, 11, RED)]], align=PP_ALIGN.CENTER)
    txt(s, 10.55, yy, 1.1, 0.34, [[R(b, 12, GREEN, True)]], align=PP_ALIGN.CENTER)
    txt(s, 11.62, yy, 0.85, 0.34, [[R(mv, 10.5, GRAY)]], align=PP_ALIGN.CENTER)
    yy += 0.52

box(s, 0.7, 5.4, 11.85, 1.4, NAVY)
txt(s, 1.0, 5.52, 11.4, 0.4, [[R("일반화 효과 입증", 13, RGBColor(0x9D,0xBE,0xE0), True)]])
gp = round(bf["prod_pct"] - es["prod_pct"], 1)
txt(s, 1.0, 5.9, 11.5, 0.85, [[
    R("단일 모델이 ", 13, RGBColor(0xD7,0xE2,0xEE)),
    R("서로 다른 10종 데이터셋에서 평균 생산률 ", 13, RGBColor(0xD7,0xE2,0xEE)),
    R(f"{bf['prod_pct']:.0f}% · 전환 {es['conv']}→{bf['conv']}회 · 최적 {bf['optimal']}/10", 13.5, WHITE, True),
    R(" 달성으로, 단순 규칙 대비 ", 13, RGBColor(0xD7,0xE2,0xEE)),
    R(f"생산률 +{gp:.0f}%p", 13.5, WHITE, True),
    R(" 향상되어 학습 모델의 ", 13, RGBColor(0xD7,0xE2,0xEE)),
    R("일반화 성능을 검증", 13.5, WHITE, True),
    R("했습니다.", 13, RGBColor(0xD7,0xE2,0xEE)),
]], line_spacing=1.28)

# ════════════════════════════════════════════════════════════════════════════
# 16. 결론
# ════════════════════════════════════════════════════════════════════════════
s = content_slide("CONCLUSION", "결론 및 기대 효과", 16)
left = [
    ("검증된 성과", "단일 학습 모델이 10종 데이터셋 중 9종에서 이론 최적(전환 0·전량 생산)에 도달."),
    ("일반화 성능", "구성·시간·난이도가 다른 데이터셋에서도 평균 생산률 99%로 안정적으로 동작."),
    ("구조적 강점", "벌크 점유 + 행동 마스킹으로 전환을 근본적으로 억제하고 학습을 안정화."),
]
right = [
    ("확장성", "더 많은 설비·제품·다단 공정으로 데이터 교체만으로 확장 적용 가능."),
    ("운영 적용", "전환 비용·계획·재공 제약을 함께 고려한 일배치 의사결정 지원."),
    ("향후 과제", "실데이터 적용·보상 가중치 튜닝·다공정 흐름 균형의 추가 검증."),
]
box(s, 0.9, 1.55, 5.65, 0.5, NAVY)
txt(s, 0.9, 1.55, 5.65, 0.5, [[R("핵심 성과", 13.5, WHITE, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
box(s, 6.75, 1.55, 5.65, 0.5, ACCENT)
txt(s, 6.75, 1.55, 5.65, 0.5, [[R("기대 효과 및 향후 과제", 13.5, WHITE, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
yy = 2.25
for (t, d), (t2, d2) in zip(left, right):
    box(s, 0.9, yy, 5.65, 1.18, LIGHT, line_color=LINE, line_w=1.0)
    box(s, 0.9, yy, 0.08, 1.18, ACCENT)
    txt(s, 1.15, yy+0.16, 5.3, 0.4, [[R(t, 13, NAVY, True)]])
    txt(s, 1.15, yy+0.56, 5.3, 0.6, [[R(d, 11.3, GRAY)]], line_spacing=1.12)
    box(s, 6.75, yy, 5.65, 1.18, LIGHT, line_color=LINE, line_w=1.0)
    box(s, 6.75, yy, 0.08, 1.18, ACCENT)
    txt(s, 7.0, yy+0.16, 5.3, 0.4, [[R(t2, 13, NAVY, True)]])
    txt(s, 7.0, yy+0.56, 5.3, 0.6, [[R(d2, 11.3, GRAY)]], line_spacing=1.12)
    yy += 1.33
box(s, 0.9, 6.35, 11.5, 0.62, NAVY)
txt(s, 0.9, 6.35, 11.5, 0.62, [[
    R("강화학습 기반 Bulk-Fill 스케줄링 — 전환 최소화와 계획 추종을 동시에 달성하는 검증된 의사결정 모델", 13, WHITE, True)
]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

OUT = os.path.join(_HERE, "BulkFill_스케줄링_발표자료.pptx")
prs.save(OUT)
print("저장 완료:", OUT, "| 슬라이드", len(prs.slides._sldIdLst))
