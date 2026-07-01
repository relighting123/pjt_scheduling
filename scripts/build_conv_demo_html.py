#!/usr/bin/env python3
"""
scripts/build_conv_demo_html.py – Conversion 시나리오 HTML 데모 + RL/휴리스틱 벤치마크 생성

사용:
    python3 scripts/build_conv_demo_html.py
    python3 scripts/build_conv_demo_html.py --train --timesteps 50000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONFIG, set_input_folder
from data.loader import load_data, validate_data
from data.preprocessor import normalize_raw, preprocess
from data.conversion_scenarios import bootstrap_conv_test_suite
from inference.runner import run_inference, run_inference_compare
from agent.rl_agent import SchedulingAgent
from api.serializers import serialize_inference_result

DEMO_DIR = ROOT / "demo"
OUTPUT_HTML = DEMO_DIR / "conv_flow_demo.html"
BENCHMARK_JSON = DEMO_DIR / "conv_benchmark.json"


def load_conv_env() -> tuple[dict, str]:
    info = bootstrap_conv_test_suite()
    folder = info["folder"]
    set_input_folder(folder)
    raw = normalize_raw(load_data())
    errors = validate_data(raw)
    if errors:
        raise RuntimeError(f"데이터 검증 실패: {errors}")
    return preprocess(raw), folder


def run_benchmark(env_data: dict, train: bool, timesteps: int) -> dict:
    rl_agent = None
    train_info: dict = {"trained": False}

    if train:
        CONFIG.rl.total_timesteps = timesteps
        rl_agent = SchedulingAgent()
        rl_agent.train(env_data, verbose=0)
        rl_agent.save()
        train_info = {
            "trained": True,
            "timesteps": timesteps,
            "eval": rl_agent.evaluate(env_data, n_episodes=3),
        }
    else:
        agent = SchedulingAgent()
        if agent.model_exists():
            rl_agent = SchedulingAgent.load()

    algorithms = ["minprogress", "earliest_st"]
    if rl_agent is not None:
        algorithms.append("rl")

    cmp = run_inference_compare(
        env_data,
        algorithms,
        rl_agent=rl_agent,
        record_history=False,
    )

    results = []
    for r in cmp["results"]:
        s = r["stats"]
        results.append({
            "algorithm": r["algorithm"],
            "idle_total": s["idle_total"],
            "conversions": s.get("conversions", 0),
            "oper_switches": s["oper_switches"],
            "prod_switches": s["prod_switches"],
            "completed_qty": sum(int(v) for v in s["completed_qty"].values()),
            "n_assignments": len(r["schedule"]),
        })

    return {
        "scenario": "conv_2ppk_1oper",
        "folder": None,
        "train": train_info,
        "results": results,
        "errors": cmp.get("errors", []),
    }


def compact_schedule(schedule: list[dict]) -> list[dict]:
    return [
        {
            "eqp": r["EQP_ID"],
            "lot": r["LOT_ID"],
            "ppk": r["PLAN_PROD_KEY"],
            "oper": r["OPER_ID"],
            "start": r["START_TM"],
            "end": r["END_TM"],
            "conv": bool(r.get("CONVERSION")),
            "wf": r.get("WF_QTY", 0),
            "lot_cd": r.get("LOT_CD", ""),
            "temp": r.get("TEMP", ""),
            "start_str": r.get("START_TM_STR", ""),
            "end_str": r.get("END_TM_STR", ""),
        }
        for r in schedule
    ]


def _key_str(k) -> str:
    if isinstance(k, tuple):
        return "|".join(str(x) for x in k)
    return str(k)


def compact_history(history: list[dict]) -> list[dict]:
    out = []
    for h in history:
        a = h.get("assigned")
        assigned = None
        if a:
            assigned = {
                "eqp": a.get("eqp_id"),
                "ppk": a.get("plan_prod_key"),
                "oper": a.get("oper_id") or a.get("abs_key", "").split("|")[1] if "|" in str(a.get("abs_key", "")) else None,
                "lot": a.get("lot_id"),
                "conv": a.get("conversion"),
                "start": a.get("start_tm"),
                "wf": a.get("wf_qty"),
                "lot_cd": a.get("lot_cd"),
                "temp": a.get("temp"),
            }
        completed = {_key_str(k): int(v) for k, v in (h.get("completed") or {}).items()}
        out.append({
            "step": h["step"],
            "time": h["time"],
            "idle": h.get("idle_total", 0),
            "assigned": assigned,
            "schedule_len": len(h.get("schedule", [])),
            "eqp_states": h.get("eqp_states", {}),
            "completed": completed,
        })
    return out


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Conversion 시나리오 – 스케줄링 데모</title>
<style>
  :root {
    --bg: #0f1419;
    --panel: #1a2332;
    --border: #2d3a4f;
    --text: #e7ecf3;
    --muted: #8b9cb3;
    --accent: #3b82f6;
    --ppk1: #22c55e;
    --ppk2: #f59e0b;
    --conv: #ef4444;
    --eqp1: #6366f1;
    --eqp2: #06b6d4;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: "Segoe UI", system-ui, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
  }
  header {
    padding: 1.5rem 2rem; border-bottom: 1px solid var(--border);
    background: linear-gradient(135deg, #1a2332 0%, #0f1419 100%);
  }
  header h1 { margin: 0 0 .25rem; font-size: 1.5rem; }
  header p { margin: 0; color: var(--muted); font-size: .95rem; }
  main { max-width: 1200px; margin: 0 auto; padding: 1.5rem 2rem 3rem; }
  section { margin-bottom: 2rem; }
  h2 { font-size: 1.1rem; margin: 0 0 1rem; color: var(--accent); }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  @media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem 1.25rem;
  }
  .scenario-tags { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .75rem; }
  .tag {
    font-size: .8rem; padding: .2rem .6rem; border-radius: 999px;
    background: #243044; border: 1px solid var(--border);
  }
  .controls { display: flex; align-items: center; gap: .75rem; flex-wrap: wrap; margin-bottom: 1rem; }
  button {
    background: var(--accent); color: #fff; border: none; border-radius: 6px;
    padding: .45rem .9rem; cursor: pointer; font-size: .9rem;
  }
  button:hover { filter: brightness(1.1); }
  button.secondary { background: #334155; }
  input[type=range] { flex: 1; min-width: 200px; }
  .step-info { font-size: .95rem; }
  .step-info strong { color: #fff; }
  .conv-badge {
    display: inline-block; background: var(--conv); color: #fff;
    font-size: .75rem; padding: .1rem .45rem; border-radius: 4px; margin-left: .5rem;
  }
  table { width: 100%; border-collapse: collapse; font-size: .85rem; }
  th, td { padding: .5rem .6rem; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; }
  tr.best td { background: rgba(34, 197, 94, .12); }
  .gantt-wrap { overflow-x: auto; }
  .gantt { min-width: 700px; }
  .gantt-row { display: flex; align-items: center; margin-bottom: .35rem; }
  .gantt-label { width: 72px; font-size: .8rem; color: var(--muted); flex-shrink: 0; }
  .gantt-track {
    flex: 1; height: 28px; position: relative; background: #121a26;
    border-radius: 4px; border: 1px solid var(--border);
  }
  .gantt-bar {
    position: absolute; top: 3px; height: 22px; border-radius: 3px;
    font-size: .65rem; color: #fff; overflow: hidden; white-space: nowrap;
    text-overflow: ellipsis; padding: 0 4px; line-height: 22px;
    transition: opacity .2s;
  }
  .gantt-bar.dim { opacity: .25; }
  .gantt-bar.conv { outline: 2px dashed var(--conv); }
  .gantt-bar.highlight { box-shadow: 0 0 0 2px #fff; z-index: 2; }
  .time-axis {
    display: flex; justify-content: space-between; font-size: .7rem;
    color: var(--muted); margin: .25rem 0 1rem 72px;
  }
  .flow-step {
    display: flex; gap: .75rem; align-items: flex-start;
    padding: .75rem 0; border-bottom: 1px solid var(--border);
  }
  .flow-num {
    width: 28px; height: 28px; border-radius: 50%; background: var(--accent);
    display: flex; align-items: center; justify-content: center;
    font-size: .8rem; font-weight: 700; flex-shrink: 0;
  }
  .flow-num.idle { background: #475569; }
  .issues { font-size: .9rem; }
  .issues li { margin-bottom: .5rem; }
  .issues .warn { color: #fbbf24; }
  .issues .ok { color: #4ade80; }
</style>
</head>
<body>
<header>
  <h1>Conversion 시나리오 데모 (conv_2ppk_1oper)</h1>
  <p>단일 공정 · 2제품 · EQP001 초기 PPK001(LC001/T650) → PPK002 투입 시 conversion 검증</p>
</header>
<main>
  <section>
    <h2>시나리오 개요</h2>
    <div class="card">
      <div id="scenario-desc"></div>
      <div class="scenario-tags" id="scenario-tags"></div>
    </div>
  </section>

  <section>
    <h2>단계별 실행 (Step ↔ Gantt 연동)</h2>
    <div class="card">
      <div class="controls">
        <button id="btn-prev">◀ 이전</button>
        <button id="btn-play">▶ 재생</button>
        <button id="btn-next">다음 ▶</button>
        <input type="range" id="step-slider" min="0" max="0" value="0">
        <span id="step-label">Step 0</span>
      </div>
      <div class="step-info" id="step-detail"></div>
      <div class="gantt-wrap" style="margin-top:1.25rem">
        <div class="time-axis" id="time-axis"></div>
        <div class="gantt" id="gantt"></div>
      </div>
      <div id="flow-detail" style="margin-top:1rem"></div>
    </div>
  </section>

  <section>
    <h2>알고리즘 성능 비교</h2>
    <div class="card">
      <table id="bench-table">
        <thead>
          <tr>
            <th>알고리즘</th><th>Idle(분)</th><th>Conversion</th>
            <th>제품전환</th><th>완료수량</th><th>배정건수</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
      <p style="color:var(--muted);font-size:.85rem;margin:1rem 0 0" id="bench-note"></p>
    </div>
  </section>

  <section>
    <h2>RL vs 휴리스틱 – 문제점 체크</h2>
    <div class="card issues">
      <ul id="issues-list"></ul>
    </div>
  </section>
</main>
<script>
const DATA = __EMBEDDED_JSON__;

const PPK_COLORS = { PPK001: "#22c55e", PPK002: "#f59e0b" };
const EQP_COLORS = { EQP001: "#6366f1", EQP002: "#06b6d4" };

let currentStep = 0;
let playTimer = null;

function init() {
  renderScenario();
  renderBenchmark();
  renderIssues();
  const maxStep = DATA.history.length - 1;
  document.getElementById("step-slider").max = maxStep;
  showStep(0);
  bindControls(maxStep);
}

function renderScenario() {
  const s = DATA.scenario_meta;
  document.getElementById("scenario-desc").innerHTML =
    `<p>EQP <strong>${s.eqps.join(", ")}</strong> · 제품 <strong>${s.prods.join(", ")}</strong> · 공정 <strong>${s.opers.join(", ")}</strong></p>` +
    `<p>EQP001 초기 상태: <strong>PPK001 / LC001 / T650</strong> (conversion 없이 PPK001 가능, PPK002는 conversion 60분)</p>`;
  const tags = [
    `알고리즘: ${DATA.algorithm}`,
    `총 Step: ${DATA.history.length}`,
    `스케줄: ${DATA.schedule.length}건`,
    `Idle: ${DATA.stats.idle_total}분`,
    `Conversion: ${DATA.stats.conversions}회`,
  ];
  document.getElementById("scenario-tags").innerHTML =
    tags.map(t => `<span class="tag">${t}</span>`).join("");
}

function showStep(step) {
  currentStep = step;
  const h = DATA.history[step];
  document.getElementById("step-slider").value = step;
  document.getElementById("step-label").textContent = `Step ${step} / ${DATA.history.length - 1}`;
  renderStepDetail(h);
  renderGantt(step);
  renderFlow(h);
}

function renderStepDetail(h) {
  const el = document.getElementById("step-detail");
  if (!h.assigned) {
    el.innerHTML = `<p><strong>시뮬 시작 (t=${h.time}분)</strong> — EQP001은 PPK001 상태, 재공 대기 중</p>`;
    return;
  }
  const a = h.assigned;
  const conv = a.conv ? '<span class="conv-badge">CONVERSION 60min</span>' : "";
  el.innerHTML =
    `<p><strong>Step ${h.step}</strong> · 시뮬시각 t=${h.time}분 · 누적 Idle ${h.idle}분</p>` +
    `<p>현재 EQP <strong>${a.eqp}</strong> → 배정 <strong>${a.ppk}</strong> / ${a.oper} / LOT ${a.lot} ` +
    `(${a.lot_cd}/${a.temp}, ${a.wf}매) START=${a.start}분 ${conv}</p>`;
}

function renderGantt(upToStep) {
  const h = DATA.history[upToStep];
  const visibleCount = h.schedule_len;
  const schedule = DATA.schedule.slice(0, visibleCount);
  const maxEnd = Math.max(DATA.sim_end_minutes, ...DATA.schedule.map(r => r.end), 480);
  const eqps = DATA.scenario_meta.eqps;

  const axis = document.getElementById("time-axis");
  const ticks = [0, Math.round(maxEnd/4), Math.round(maxEnd/2), Math.round(maxEnd*3/4), maxEnd];
  axis.innerHTML = ticks.map(t => `<span>${t}분</span>`).join("");

  const gantt = document.getElementById("gantt");
  gantt.innerHTML = eqps.map(eqp => {
    const bars = schedule
      .map((r, i) => ({ ...r, idx: i }))
      .filter(r => r.eqp === eqp)
      .map(r => {
        const left = (r.start / maxEnd) * 100;
        const width = Math.max(((r.end - r.start) / maxEnd) * 100, 0.5);
        const isLast = upToStep > 0 && h.assigned && r.idx === visibleCount - 1;
        const cls = ["gantt-bar", r.conv ? "conv" : "", isLast ? "highlight" : ""].filter(Boolean).join(" ");
        const color = PPK_COLORS[r.ppk] || "#64748b";
        const label = `${r.ppk}${r.conv ? " ⚡" : ""}`;
        return `<div class="${cls}" style="left:${left}%;width:${width}%;background:${color}" title="${r.lot} ${r.start_str}~${r.end_str}">${label}</div>`;
      }).join("");
    return `<div class="gantt-row"><div class="gantt-label">${eqp}</div><div class="gantt-track">${bars}</div></div>`;
  }).join("");
}

function renderFlow(h) {
  const el = document.getElementById("flow-detail");
  if (!h.assigned) {
    el.innerHTML = "<p style='color:var(--muted)'>슬라이더를 움직이면 배정·간트가 연동됩니다.</p>";
    return;
  }
  const a = h.assigned;
  const states = Object.entries(h.eqp_states || {})
    .map(([id, s]) => `${id}: ${s.status}${s.current_prod ? " ("+s.current_prod+")" : ""} free@${s.free_at}`)
    .join(" · ");
  el.innerHTML =
    `<div class="flow-step"><div class="flow-num">${h.step}</div><div>` +
    `<div><strong>Action</strong>: (PPK, OPER) = (${a.ppk}, ${a.oper}) · 시스템 EQP=${a.eqp}, LOT 자동선택</div>` +
    `<div style="color:var(--muted);margin-top:.35rem">EQP 상태: ${states}</div></div></div>`;
}

function renderBenchmark() {
  const tbody = document.querySelector("#bench-table tbody");
  const results = DATA.benchmark.results;
  const bestIdle = Math.min(...results.map(r => r.idle_total));
  const bestCompleted = Math.max(...results.map(r => r.completed_qty));
  tbody.innerHTML = results.map(r => {
    const isBest = r.idle_total === bestIdle && r.completed_qty === bestCompleted;
    const name = { minprogress: "MinProgress", earliest_st: "Earliest ST", rl: "RL (PPO)" }[r.algorithm] || r.algorithm;
    return `<tr class="${isBest ? "best" : ""}"><td>${name}</td><td>${r.idle_total}</td><td>${r.conversions}</td>` +
      `<td>${r.prod_switches}</td><td>${r.completed_qty}</td><td>${r.n_assignments}</td></tr>`;
  }).join("");
  const train = DATA.benchmark.train;
  document.getElementById("bench-note").textContent = train.trained
    ? `RL 학습: ${train.timesteps.toLocaleString()} timesteps · eval mean_reward=${train.eval.mean_reward.toFixed(1)}`
    : "RL 모델 없음 — python3 scripts/build_conv_demo_html.py --train 로 학습 가능";
}

function renderIssues() {
  const results = DATA.benchmark.results;
  const rl = results.find(r => r.algorithm === "rl");
  const mp = results.find(r => r.algorithm === "minprogress");
  const est = results.find(r => r.algorithm === "earliest_st");
  const items = [];

  if (!rl) {
    items.push({ cls: "warn", text: "학습된 RL 모델 없음 — 벤치마크는 휴리스틱만 포함" });
  } else if (rl.idle_total <= (est?.idle_total ?? Infinity) && rl.completed_qty >= (mp?.completed_qty ?? 0)) {
    items.push({ cls: "ok", text: `conv 시나리오: RL이 idle=${rl.idle_total}, conversion=${rl.conversions}로 earliest_st와 동급, minprogress(idle=${mp?.idle_total})보다 우수` });
  } else {
    items.push({ cls: "warn", text: "conv 시나리오에서 RL이 휴리스틱 대비 열위 — 학습 budget·보상 설계 재검토 필요" });
  }

  if (mp && mp.conversions > (rl?.conversions ?? 0)) {
    items.push({ cls: "warn", text: `MinProgress는 제품 진행률 우선 → EQP001(PPK001 상태)에 PPK002를 먼저 넣어 conversion ${mp.conversions}회 발생 (비효율)` });
  }

  items.push({ cls: "warn", text: "액션 공간이 (PPK, OPER)만 선택 — EQP·LOT은 규칙 자동. RL은 제품/공정 순서만 학습" });
  items.push({ cls: "warn", text: "단일 conv 시나리오 50k step 학습은 일반화 부족 — 다양한 train 기간·VecEnv 필요" });
  items.push({ cls: "warn", text: "obs/action 차원 변경 시 기존 checkpoint 무효 — 재학습 필수" });
  items.push({ cls: "warn", text: "보상이 idle·전환·plan pacing 혼합 — conversion 비용 신호가 약하면 RL이 휴리스틱과 동일 정책 수렴" });

  document.getElementById("issues-list").innerHTML =
    items.map(i => `<li class="${i.cls}">${i.text}</li>`).join("");
}

function bindControls(maxStep) {
  document.getElementById("btn-prev").onclick = () => showStep(Math.max(0, currentStep - 1));
  document.getElementById("btn-next").onclick = () => showStep(Math.min(maxStep, currentStep + 1));
  document.getElementById("step-slider").oninput = e => showStep(+e.target.value);
  document.getElementById("btn-play").onclick = () => {
    if (playTimer) { clearInterval(playTimer); playTimer = null; document.getElementById("btn-play").textContent = "▶ 재생"; return; }
    document.getElementById("btn-play").textContent = "⏸ 정지";
    playTimer = setInterval(() => {
      if (currentStep >= maxStep) { clearInterval(playTimer); playTimer = null; document.getElementById("btn-play").textContent = "▶ 재생"; return; }
      showStep(currentStep + 1);
    }, 800);
  };
}

init();
</script>
</body>
</html>
"""


def build_html(env_data: dict, demo_result: dict, benchmark: dict) -> str:
    payload = {
        "algorithm": demo_result["algorithm"],
        "scenario_meta": {
            "eqps": env_data["eqp_ids"],
            "prods": env_data["prod_keys"],
            "opers": env_data["oper_ids"],
        },
        "sim_end_minutes": env_data["sim_end_minutes"],
        "stats": demo_result["stats"],
        "schedule": compact_schedule(demo_result["schedule"]),
        "history": compact_history(demo_result["history"]),
        "benchmark": benchmark,
    }
    embedded = json.dumps(payload, ensure_ascii=False)
    return HTML_TEMPLATE.replace("__EMBEDDED_JSON__", embedded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="RL 학습 후 벤치마크")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--algorithm", default="minprogress", choices=["minprogress", "earliest_st", "rl"])
    args = parser.parse_args()

    env_data, folder = load_conv_env()
    print(f"[demo] 시나리오 로드: {folder}")

    benchmark = run_benchmark(env_data, train=args.train, timesteps=args.timesteps)
    benchmark["folder"] = folder

    demo_algo = args.algorithm
    rl_agent = None
    if demo_algo == "rl":
        agent = SchedulingAgent()
        if not agent.model_exists():
            print("[demo] RL 모델 없음 — minprogress로 데모 생성")
            demo_algo = "minprogress"
        else:
            rl_agent = SchedulingAgent.load()

    demo_result = run_inference(
        env_data,
        algorithm=demo_algo,
        agent=rl_agent,
        record_history=True,
    )
    print(f"[demo] 추론 완료: {demo_algo}, steps={len(demo_result['history'])}")

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    html = build_html(env_data, demo_result, benchmark)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    BENCHMARK_JSON.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[demo] HTML → {OUTPUT_HTML}")
    print(f"[demo] 벤치마크 → {BENCHMARK_JSON}")


if __name__ == "__main__":
    main()
