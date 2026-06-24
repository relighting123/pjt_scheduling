import { useMemo } from "react";
import { buildShortCodeMap } from "../lib/ganttLabels";
import {
  computeAchievement,
  computeEqpScheduleSummary,
  computeInferenceKpi,
} from "../lib/metrics";
import type { InferenceResult } from "../types";

interface Props {
  result: InferenceResult;
  eqpModelMap: Record<string, string>;
}

function fmtMin(v: number | null): string {
  return v === null ? "—" : `${v}분`;
}

function utilClass(pct: number): string {
  if (pct >= 80) return "ok";
  if (pct >= 50) return "warn";
  return "bad";
}

function achClass(pct: number): string {
  if (pct >= 90) return "ok";
  if (pct >= 70) return "warn";
  return "bad";
}

export default function GanttSummaryPanel({ result, eqpModelMap }: Props) {
  const prodMap = useMemo(
    () => buildShortCodeMap(result.prod_keys, "P"),
    [result.prod_keys],
  );
  const operMap = useMemo(
    () => buildShortCodeMap(result.oper_ids, "O"),
    [result.oper_ids],
  );

  const eqpRows = useMemo(
    () => computeEqpScheduleSummary(
      result.schedule,
      result.eqp_ids,
      result.sim_end_minutes,
      eqpModelMap,
      result.conversion_plans ?? [],
    ),
    [result, eqpModelMap],
  );

  const achRows = useMemo(
    () => computeAchievement(result.schedule, result.plan),
    [result.schedule, result.plan],
  );

  const kpi = useMemo(
    () => computeInferenceKpi(result, eqpModelMap),
    [result, eqpModelMap],
  );

  return (
    <div className="gantt-summary">
      <div className="gantt-summary-kpi card">
        <div className="gantt-summary-kpi-grid">
          <div><span className="gantt-summary-kpi-label">Makespan</span><strong>{kpi.makespan}분</strong></div>
          <div><span className="gantt-summary-kpi-label">평균 가동률</span><strong className={utilClass(kpi.avgUtilPct)}>{kpi.avgUtilPct}%</strong></div>
          <div><span className="gantt-summary-kpi-label">평균 달성률</span><strong className={achClass(kpi.avgAchPct)}>{kpi.avgAchPct}%</strong></div>
          <div><span className="gantt-summary-kpi-label">공정 전환</span><strong>{kpi.operSwitches}회</strong></div>
          <div><span className="gantt-summary-kpi-label">제품 전환</span><strong>{kpi.prodSwitches}회</strong></div>
          <div><span className="gantt-summary-kpi-label">Tool 전환</span><strong>{kpi.toolSwitches}회</strong></div>
        </div>
      </div>

      <div className="gantt-summary-codes card">
        <div className="gantt-summary-section-title">코드 매핑</div>
        <div className="gantt-summary-code-grid">
          <div>
            <div className="gantt-summary-subtitle">제품 (P)</div>
            <div className="table-wrap">
              <table className="gantt-summary-table">
                <thead><tr><th>코드</th><th>PPK (PLAN_PROD_KEY)</th></tr></thead>
                <tbody>
                  {prodMap.ordered.map(({ code, key }) => (
                    <tr key={code}>
                      <td className="mono code-chip">{code}</td>
                      <td>{key}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div>
            <div className="gantt-summary-subtitle">공정 (O)</div>
            <div className="table-wrap">
              <table className="gantt-summary-table">
                <thead><tr><th>코드</th><th>OPER_ID</th></tr></thead>
                <tbody>
                  {operMap.ordered.map(({ code, key }) => (
                    <tr key={code}>
                      <td className="mono code-chip">{code}</td>
                      <td>{key}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="gantt-summary-eqp card">
        <div className="gantt-summary-section-title">장비별 스케줄 · KPI</div>
        <div className="table-wrap gantt-summary-scroll">
          <table className="gantt-summary-table">
            <thead>
              <tr>
                <th>장비</th>
                <th>모델</th>
                <th>할당 시작</th>
                <th>할당 완료</th>
                <th>작업</th>
                <th>가동(분)</th>
                <th>Conv(분)</th>
                <th>유휴(분)</th>
                <th>가동률</th>
                <th>누적 실적</th>
                <th>공정전환</th>
                <th>제품전환</th>
              </tr>
            </thead>
            <tbody>
              {eqpRows.map((row) => (
                <tr key={row.eqp_id}>
                  <td className="mono">{row.eqp_id}</td>
                  <td>{row.model ?? "—"}</td>
                  <td className="mono">{fmtMin(row.firstStart)}</td>
                  <td className="mono">{fmtMin(row.lastEnd)}</td>
                  <td className="mono">{row.jobCount}건</td>
                  <td className="mono">{row.busyMin}</td>
                  <td className="mono">{row.convMin}</td>
                  <td className="mono">{row.idleMin}</td>
                  <td className={`mono ${utilClass(row.utilPct)}`}>{row.utilPct}%</td>
                  <td className="mono">{row.outputQty.toLocaleString()}매</td>
                  <td className="mono">{row.operSwitches}회</td>
                  <td className="mono">{row.prodSwitches}회</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="gantt-summary-plan card">
        <div className="gantt-summary-section-title">제품·공정 계획 vs 실적</div>
        <div className="table-wrap gantt-summary-scroll">
          <table className="gantt-summary-table">
            <thead>
              <tr>
                <th>제품</th>
                <th>공정</th>
                <th>PPK</th>
                <th>OPER</th>
                <th>계획량</th>
                <th>누적 실적</th>
                <th>달성률</th>
              </tr>
            </thead>
            <tbody>
              {achRows.map((row) => {
                const pCode = prodMap.codeByKey[row.prod] ?? row.prod;
                const oCode = operMap.codeByKey[row.oper] ?? row.oper;
                const pct = Math.min(row.pct, 100);
                return (
                  <tr key={row.key}>
                    <td className="mono code-chip">{pCode}</td>
                    <td className="mono code-chip">{oCode}</td>
                    <td>{row.prod}</td>
                    <td>{row.oper}</td>
                    <td className="mono">{row.planQty.toLocaleString()}매</td>
                    <td className="mono">{row.doneQty.toLocaleString()}매</td>
                    <td className={`mono ${achClass(pct)}`}>{row.pct}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
