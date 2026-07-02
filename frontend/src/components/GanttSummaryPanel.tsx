import { useCallback, useMemo } from "react";
import { buildGanttLegendItems, type GanttLegendItem } from "../lib/charts";
import {
  compareNumbers,
  compareStrings,
  useTableFilterSort,
} from "../hooks/useTableFilterSort";
import {
  computeAchievement,
  computeEqpScheduleSummary,
  computeInferenceKpi,
  type AchievementRow,
  type EqpScheduleSummary,
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

function idleClass(pct: number): string {
  if (pct <= 20) return "ok";
  if (pct <= 50) return "warn";
  return "bad";
}

function achClass(pct: number): string {
  if (pct >= 90) return "ok";
  if (pct >= 70) return "warn";
  return "bad";
}

function TableToolbar({
  query,
  onQueryChange,
  shown,
  total,
  placeholder,
}: {
  query: string;
  onQueryChange: (value: string) => void;
  shown: number;
  total: number;
  placeholder: string;
}) {
  return (
    <div className="gantt-table-toolbar">
      <input
        type="search"
        className="gantt-table-search"
        placeholder={placeholder}
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
      />
      <span className="gantt-table-count">{shown}/{total}건</span>
    </div>
  );
}

function SortableTh({
  label,
  sortKey,
  currentKey,
  currentDir,
  onSort,
  className = "",
}: {
  label: string;
  sortKey: string;
  currentKey: string;
  currentDir: "asc" | "desc";
  onSort: (key: string) => void;
  className?: string;
}) {
  const active = currentKey === sortKey;
  return (
    <th
      className={`sortable${active ? " active" : ""}${className ? ` ${className}` : ""}`}
      onClick={() => onSort(sortKey)}
      aria-sort={active ? (currentDir === "asc" ? "ascending" : "descending") : "none"}
    >
      <span className="sortable-label">{label}</span>
      {active && <span className="sort-indicator">{currentDir === "asc" ? "↑" : "↓"}</span>}
    </th>
  );
}

function filterLegendItem(item: GanttLegendItem, query: string): boolean {
  return (
    item.label.toLowerCase().includes(query)
    || item.prodKey.toLowerCase().includes(query)
    || item.operId.toLowerCase().includes(query)
    || item.color.toLowerCase().includes(query)
  );
}

function sortLegendItems(a: GanttLegendItem, b: GanttLegendItem, key: string, dir: "asc" | "desc") {
  const pick = (item: GanttLegendItem) => {
    switch (key) {
      case "prodKey": return item.prodKey;
      case "operId": return item.operId;
      case "color": return item.color;
      default: return item.label;
    }
  };
  return compareStrings(pick(a), pick(b), dir);
}

function filterEqpRow(row: EqpScheduleSummary, query: string): boolean {
  const blob = [
    row.eqp_id,
    row.model ?? "",
    row.jobCount,
    row.busyMin,
    row.convMin,
    row.idleMin,
    row.utilPct,
    row.idlePct,
    row.outputQty,
    row.operSwitches,
    row.prodSwitches,
    row.firstStart ?? "",
    row.lastEnd ?? "",
  ].join(" ").toLowerCase();
  return blob.includes(query);
}

function sortEqpRow(a: EqpScheduleSummary, b: EqpScheduleSummary, key: string, dir: "asc" | "desc") {
  switch (key) {
    case "eqp_id": return compareStrings(a.eqp_id, b.eqp_id, dir);
    case "model": return compareStrings(a.model ?? "", b.model ?? "", dir);
    case "firstStart": return compareNumbers(a.firstStart ?? -1, b.firstStart ?? -1, dir);
    case "lastEnd": return compareNumbers(a.lastEnd ?? -1, b.lastEnd ?? -1, dir);
    case "jobCount": return compareNumbers(a.jobCount, b.jobCount, dir);
    case "busyMin": return compareNumbers(a.busyMin, b.busyMin, dir);
    case "convMin": return compareNumbers(a.convMin, b.convMin, dir);
    case "idleMin": return compareNumbers(a.idleMin, b.idleMin, dir);
    case "utilPct": return compareNumbers(a.utilPct, b.utilPct, dir);
    case "idlePct": return compareNumbers(a.idlePct, b.idlePct, dir);
    case "outputQty": return compareNumbers(a.outputQty, b.outputQty, dir);
    case "operSwitches": return compareNumbers(a.operSwitches, b.operSwitches, dir);
    case "prodSwitches": return compareNumbers(a.prodSwitches, b.prodSwitches, dir);
    default: return compareStrings(a.eqp_id, b.eqp_id, dir);
  }
}

function filterAchRow(row: AchievementRow, query: string, prodCode: string, operCode: string): boolean {
  const blob = [
    row.prod,
    row.oper,
    prodCode,
    operCode,
    row.planQty,
    row.doneQty,
    row.pct,
  ].join(" ").toLowerCase();
  return blob.includes(query);
}

function sortAchRow(a: AchievementRow, b: AchievementRow, key: string, dir: "asc" | "desc") {
  switch (key) {
    case "prod": return compareStrings(a.prod, b.prod, dir);
    case "oper": return compareStrings(a.oper, b.oper, dir);
    case "planQty": return compareNumbers(a.planQty, b.planQty, dir);
    case "doneQty": return compareNumbers(a.doneQty, b.doneQty, dir);
    case "pct": return compareNumbers(a.pct, b.pct, dir);
    default: return compareStrings(a.prod, b.prod, dir);
  }
}

export default function GanttSummaryPanel({ result, eqpModelMap }: Props) {
  const legendItems = useMemo(
    () => buildGanttLegendItems(result.schedule, result.prod_keys, result.oper_ids),
    [result.schedule, result.prod_keys, result.oper_ids],
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
    () => computeAchievement(result.schedule, result.plan, {
      prodKeys: result.prod_keys,
      operIds: result.oper_ids,
    }),
    [result.schedule, result.plan, result.prod_keys, result.oper_ids],
  );

  const kpi = useMemo(
    () => computeInferenceKpi(result, eqpModelMap),
    [result, eqpModelMap],
  );

  const codeTable = useTableFilterSort(
    legendItems,
    filterLegendItem,
    sortLegendItems,
    { key: "label", dir: "asc" },
  );

  const eqpTable = useTableFilterSort(
    eqpRows,
    filterEqpRow,
    sortEqpRow,
    { key: "eqp_id", dir: "asc" },
  );

  const achFilterFn = useCallback((row: AchievementRow, query: string) => {
    const prodCode = legendItems.find((i) => i.prodKey === row.prod)?.label.split("/")[0] ?? row.prod;
    const operCode = legendItems.find((i) => i.operId === row.oper)?.label.split("/")[1] ?? row.oper;
    return filterAchRow(row, query, prodCode, operCode);
  }, [legendItems]);

  const achTable = useTableFilterSort(
    achRows,
    achFilterFn,
    sortAchRow,
    { key: "prod", dir: "asc" },
  );

  const achCodeLookup = useMemo(() => {
    const prod = new Map<string, string>();
    const oper = new Map<string, string>();
    legendItems.forEach((item) => {
      const [p, o] = item.label.split("/");
      prod.set(item.prodKey, p ?? item.prodKey);
      oper.set(item.operId, o ?? item.operId);
    });
    return { prod, oper };
  }, [legendItems]);

  return (
    <div className="gantt-summary">
      <div className="gantt-summary-kpi card">
        <div className="gantt-summary-kpi-grid">
          <div><span className="gantt-summary-kpi-label">Makespan</span><strong>{kpi.makespan}분</strong></div>
          <div><span className="gantt-summary-kpi-label">평균 가동률</span><strong className={utilClass(kpi.avgUtilPct)}>{kpi.avgUtilPct}%</strong></div>
          <div><span className="gantt-summary-kpi-label">평균 유휴율</span><strong className={idleClass(kpi.avgIdlePct)}>{kpi.avgIdlePct}%</strong></div>
          <div><span className="gantt-summary-kpi-label">평균 달성률</span><strong className={achClass(kpi.avgAchPct)}>{kpi.avgAchPct}%</strong></div>
          <div><span className="gantt-summary-kpi-label">공정 전환</span><strong>{kpi.operSwitches}회</strong></div>
          <div><span className="gantt-summary-kpi-label">제품 전환</span><strong>{kpi.prodSwitches}회</strong></div>
          <div><span className="gantt-summary-kpi-label">Tool 전환</span><strong>{kpi.toolSwitches}회</strong></div>
        </div>
      </div>

      <div className="gantt-summary-codes card">
        <div className="gantt-summary-section-title">제품×공정 코드 매핑</div>
        <TableToolbar
          query={codeTable.query}
          onQueryChange={codeTable.setQuery}
          shown={codeTable.filtered.length}
          total={legendItems.length}
          placeholder="조합·PPK·OPER·색상 검색"
        />
        <div className="table-wrap gantt-summary-scroll">
          <table className="gantt-summary-table gantt-summary-code-table">
            <colgroup>
              <col className="col-code" />
              <col className="col-key" />
              <col className="col-key" />
              <col className="col-color" />
            </colgroup>
            <thead>
              <tr>
                <SortableTh label="조합" sortKey="label" currentKey={codeTable.sort.key} currentDir={codeTable.sort.dir} onSort={codeTable.toggleSort} className="col-code" />
                <SortableTh label="PPK" sortKey="prodKey" currentKey={codeTable.sort.key} currentDir={codeTable.sort.dir} onSort={codeTable.toggleSort} className="col-key" />
                <SortableTh label="OPER_ID" sortKey="operId" currentKey={codeTable.sort.key} currentDir={codeTable.sort.dir} onSort={codeTable.toggleSort} className="col-key" />
                <SortableTh label="색상" sortKey="color" currentKey={codeTable.sort.key} currentDir={codeTable.sort.dir} onSort={codeTable.toggleSort} className="col-color" />
              </tr>
            </thead>
            <tbody>
              {codeTable.filtered.map((item) => (
                <tr key={item.pairKey}>
                  <td className="mono col-code">
                    <span className="code-text" style={{ color: item.color }}>{item.label}</span>
                  </td>
                  <td className="cell-key col-key">{item.prodKey}</td>
                  <td className="cell-key col-key">{item.operId}</td>
                  <td className="mono col-color">{item.color}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {codeTable.filtered.length === 0 && (
            <p className="gantt-table-empty">검색 결과가 없습니다.</p>
          )}
        </div>
      </div>

      <div className="gantt-summary-eqp card">
        <div className="gantt-summary-section-title">장비별 스케줄 · KPI</div>
        <TableToolbar
          query={eqpTable.query}
          onQueryChange={eqpTable.setQuery}
          shown={eqpTable.filtered.length}
          total={eqpRows.length}
          placeholder="장비·모델·수치 검색"
        />
        <div className="table-wrap gantt-summary-scroll">
          <table className="gantt-summary-table gantt-summary-eqp-table">
            <colgroup>
              <col className="col-eqp" />
              <col className="col-model" />
              <col className="col-time" />
              <col className="col-time" />
              <col className="col-jobs" />
              <col className="col-num" />
              <col className="col-num" />
              <col className="col-num" />
              <col className="col-num-sm" />
              <col className="col-num-sm" />
              <col className="col-num-sm" />
              <col className="col-output" />
              <col className="col-num-sm" />
              <col className="col-num-sm" />
            </colgroup>
            <thead>
              <tr>
                <SortableTh label="장비" sortKey="eqp_id" currentKey={eqpTable.sort.key} currentDir={eqpTable.sort.dir} onSort={eqpTable.toggleSort} />
                <SortableTh label="모델" sortKey="model" currentKey={eqpTable.sort.key} currentDir={eqpTable.sort.dir} onSort={eqpTable.toggleSort} />
                <SortableTh label="할당 시작" sortKey="firstStart" currentKey={eqpTable.sort.key} currentDir={eqpTable.sort.dir} onSort={eqpTable.toggleSort} className="num" />
                <SortableTh label="할당 완료" sortKey="lastEnd" currentKey={eqpTable.sort.key} currentDir={eqpTable.sort.dir} onSort={eqpTable.toggleSort} className="num" />
                <SortableTh label="작업" sortKey="jobCount" currentKey={eqpTable.sort.key} currentDir={eqpTable.sort.dir} onSort={eqpTable.toggleSort} className="num" />
                <SortableTh label="가동(분)" sortKey="busyMin" currentKey={eqpTable.sort.key} currentDir={eqpTable.sort.dir} onSort={eqpTable.toggleSort} className="num" />
                <SortableTh label="Conv(분)" sortKey="convMin" currentKey={eqpTable.sort.key} currentDir={eqpTable.sort.dir} onSort={eqpTable.toggleSort} className="num" />
                <SortableTh label="유휴(분)" sortKey="idleMin" currentKey={eqpTable.sort.key} currentDir={eqpTable.sort.dir} onSort={eqpTable.toggleSort} className="num" />
                <SortableTh label="가동률" sortKey="utilPct" currentKey={eqpTable.sort.key} currentDir={eqpTable.sort.dir} onSort={eqpTable.toggleSort} className="num" />
                <SortableTh label="유휴율" sortKey="idlePct" currentKey={eqpTable.sort.key} currentDir={eqpTable.sort.dir} onSort={eqpTable.toggleSort} className="num" />
                <SortableTh label="누적 실적" sortKey="outputQty" currentKey={eqpTable.sort.key} currentDir={eqpTable.sort.dir} onSort={eqpTable.toggleSort} className="num" />
                <SortableTh label="공정전환" sortKey="operSwitches" currentKey={eqpTable.sort.key} currentDir={eqpTable.sort.dir} onSort={eqpTable.toggleSort} className="num" />
                <SortableTh label="제품전환" sortKey="prodSwitches" currentKey={eqpTable.sort.key} currentDir={eqpTable.sort.dir} onSort={eqpTable.toggleSort} className="num" />
              </tr>
            </thead>
            <tbody>
              {eqpTable.filtered.map((row) => (
                <tr key={row.eqp_id}>
                  <td className="mono">{row.eqp_id}</td>
                  <td className="cell-key">{row.model ?? "—"}</td>
                  <td className="mono num">{fmtMin(row.firstStart)}</td>
                  <td className="mono num">{fmtMin(row.lastEnd)}</td>
                  <td className="mono num">{row.jobCount}건</td>
                  <td className="mono num">{row.busyMin}</td>
                  <td className="mono num">{row.convMin}</td>
                  <td className="mono num">{row.idleMin}</td>
                  <td className={`mono num ${utilClass(row.utilPct)}`}>{row.utilPct}%</td>
                  <td className={`mono num ${idleClass(row.idlePct)}`}>{row.idlePct}%</td>
                  <td className="mono num">{row.outputQty.toLocaleString()}매</td>
                  <td className="mono num">{row.operSwitches}회</td>
                  <td className="mono num">{row.prodSwitches}회</td>
                </tr>
              ))}
            </tbody>
          </table>
          {eqpTable.filtered.length === 0 && (
            <p className="gantt-table-empty">검색 결과가 없습니다.</p>
          )}
        </div>
      </div>

      <div className="gantt-summary-plan card">
        <div className="gantt-summary-section-title">제품·공정 계획 vs 실적</div>
        <TableToolbar
          query={achTable.query}
          onQueryChange={achTable.setQuery}
          shown={achTable.filtered.length}
          total={achRows.length}
          placeholder="제품·공정·수량 검색"
        />
        <div className="table-wrap gantt-summary-scroll">
          <table className="gantt-summary-table gantt-summary-plan-table">
            <colgroup>
              <col className="col-code" />
              <col className="col-code" />
              <col className="col-key" />
              <col className="col-key" />
              <col className="col-num" />
              <col className="col-num" />
              <col className="col-num" />
            </colgroup>
            <thead>
              <tr>
                <SortableTh label="제품" sortKey="prod" currentKey={achTable.sort.key} currentDir={achTable.sort.dir} onSort={achTable.toggleSort} className="col-code" />
                <SortableTh label="공정" sortKey="oper" currentKey={achTable.sort.key} currentDir={achTable.sort.dir} onSort={achTable.toggleSort} className="col-code" />
                <SortableTh label="PPK" sortKey="prod" currentKey={achTable.sort.key} currentDir={achTable.sort.dir} onSort={achTable.toggleSort} className="col-key" />
                <SortableTh label="OPER" sortKey="oper" currentKey={achTable.sort.key} currentDir={achTable.sort.dir} onSort={achTable.toggleSort} className="col-key" />
                <SortableTh label="계획량" sortKey="planQty" currentKey={achTable.sort.key} currentDir={achTable.sort.dir} onSort={achTable.toggleSort} className="num col-num" />
                <SortableTh label="누적 실적" sortKey="doneQty" currentKey={achTable.sort.key} currentDir={achTable.sort.dir} onSort={achTable.toggleSort} className="num col-num" />
                <SortableTh label="달성률" sortKey="pct" currentKey={achTable.sort.key} currentDir={achTable.sort.dir} onSort={achTable.toggleSort} className="num col-num" />
              </tr>
            </thead>
            <tbody>
              {achTable.filtered.map((row) => {
                const pCode = achCodeLookup.prod.get(row.prod) ?? row.prod;
                const oCode = achCodeLookup.oper.get(row.oper) ?? row.oper;
                const pct = Math.min(row.pct, 100);
                return (
                  <tr key={row.key}>
                    <td className="mono col-code"><span className="code-text">{pCode}</span></td>
                    <td className="mono col-code"><span className="code-text">{oCode}</span></td>
                    <td className="cell-key col-key">{row.prod}</td>
                    <td className="cell-key col-key">{row.oper}</td>
                    <td className="mono num col-num">{row.planQty.toLocaleString()}매</td>
                    <td className="mono num col-num">{row.doneQty.toLocaleString()}매</td>
                    <td className={`mono num col-num ${achClass(pct)}`}>{row.pct}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {achTable.filtered.length === 0 && (
            <p className="gantt-table-empty">검색 결과가 없습니다.</p>
          )}
        </div>
      </div>
    </div>
  );
}
