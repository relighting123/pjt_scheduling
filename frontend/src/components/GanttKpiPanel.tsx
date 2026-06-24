import { useMemo, useState } from "react";
import PlotChart from "./PlotChart";
import {
  buildAchievementTableChart,
  buildEqpUtilChart,
  buildModelUtilChart,
  buildTatChart,
} from "../lib/charts";
import {
  computeAchievement,
  computeAvgAchievement,
  computeAvgUtil,
  computeEqpUtil,
  computeModelUtil,
  computeTAT,
  countToolSwitches,
} from "../lib/metrics";
import type { InferenceResult } from "../types";

interface Props {
  result: InferenceResult;
  eqpModelMap: Record<string, string>;
}

type DetailTab = "ach" | "eqp" | "model" | "tat" | "sw";

const DETAIL_TABS: { id: DetailTab; label: string }[] = [
  { id: "ach",   label: "달성률" },
  { id: "eqp",   label: "장비 가동률" },
  { id: "model", label: "모델 가동률" },
  { id: "tat",   label: "TAT" },
  { id: "sw",    label: "전환 횟수" },
];

export default function GanttKpiPanel({ result, eqpModelMap }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [tab, setTab] = useState<DetailTab>("ach");

  const sched = result.schedule;
  const makespan = sched.length ? Math.max(...sched.map((r) => r.END_TM)) : 0;

  const utils  = useMemo(() => computeEqpUtil(sched, result.eqp_ids, result.sim_end_minutes, eqpModelMap), [sched, result, eqpModelMap]);
  const models = useMemo(() => computeModelUtil(utils), [utils]);
  const ach    = useMemo(() => computeAchievement(sched, result.plan), [sched, result.plan]);
  const tat    = useMemo(() => computeTAT(sched), [sched]);
  const toolSw = countToolSwitches(sched, result.conversion_plans ?? []);

  const avgUtil = computeAvgUtil(utils);
  const avgAch  = computeAvgAchievement(ach);

  const kpis = [
    { label: "Makespan",  value: `${makespan}분`,                cls: "" },
    { label: "평균 가동률", value: `${avgUtil}%`,                 cls: avgUtil >= 80 ? "ok" : avgUtil >= 50 ? "warn" : "bad" },
    { label: "공정 전환",  value: `${result.stats.oper_switches}회`, cls: "" },
    { label: "제품 전환",  value: `${result.stats.prod_switches}회`, cls: "" },
    { label: "Tool 전환",  value: `${toolSw}회`,                  cls: "" },
    { label: "평균 달성률", value: `${avgAch}%`,                  cls: avgAch >= 90 ? "ok" : avgAch >= 70 ? "warn" : "bad" },
  ];

  return (
    <div>
      <div className="gantt-kpi-strip">
        {kpis.map((k) => (
          <div key={k.label} className="gantt-kpi-item">
            <div className="gantt-kpi-label">{k.label}</div>
            <div className={`gantt-kpi-val ${k.cls}`}>{k.value}</div>
          </div>
        ))}
        <div className="gantt-kpi-expand">
          <button
            type="button"
            className={`btn btn-xs ${expanded ? "btn-accent" : "btn-ghost"}`}
            onClick={() => setExpanded((v) => !v)}
          >
            {expanded ? "▲ 닫기" : "▼ 세부 지표"}
          </button>
        </div>
      </div>

      {expanded && (
        <div className="gantt-detail card mt-1">
          <div className="tabs mb-2">
            {DETAIL_TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                className={`tab-btn${tab === t.id ? " active" : ""}`}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>

          {tab === "ach"   && <div className="chart-wrap"><PlotChart {...buildAchievementTableChart(ach)} /></div>}
          {tab === "eqp"   && <div className="chart-wrap"><PlotChart {...buildEqpUtilChart(utils)} /></div>}
          {tab === "model" && <div className="chart-wrap"><PlotChart {...buildModelUtilChart(models)} /></div>}

          {tab === "tat" && (
            <>
              {tat.length > 0
                ? (
                  <>
                    <div className="chart-wrap mb-2"><PlotChart {...buildTatChart(tat)} /></div>
                    <div className="table-wrap">
                      <table>
                        <thead><tr><th>제품</th><th>LOT 수</th><th>평균 TAT</th><th>최소</th><th>최대</th></tr></thead>
                        <tbody>
                          {tat.map((r) => (
                            <tr key={r.prod}>
                              <td>{r.prod}</td>
                              <td>{r.count}</td>
                              <td>{r.avgMin}분</td>
                              <td>{r.minMin}분</td>
                              <td>{r.maxMin}분</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )
                : <p className="hint">스케줄 데이터 없음</p>
              }
            </>
          )}

          {tab === "sw" && (
            <div className="table-wrap">
              <table>
                <thead><tr><th>전환 종류</th><th>횟수</th></tr></thead>
                <tbody>
                  <tr><td>공정(OPER) 전환</td><td className="mono">{result.stats.oper_switches}회</td></tr>
                  <tr><td>제품(PROD) 전환</td><td className="mono">{result.stats.prod_switches}회</td></tr>
                  <tr><td>Tool 전환 (Conversion)</td><td className="mono">{toolSw}회</td></tr>
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
