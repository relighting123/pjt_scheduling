import { useMemo } from "react";
import {
  buildCapacityMatrix,
  capacityReasonLabel,
  type CapacityBucketRow,
} from "../lib/metrics";
import {
  compareNumbers,
  compareStrings,
  useTableFilterSort,
} from "../hooks/useTableFilterSort";
import type { CapacityPlanRow } from "../types";

interface Props {
  rows: CapacityPlanRow[];
}

function reasonBadgeClass(code: string): string {
  switch (code) {
    case "OK":
    case "DONE": return "badge-ok";
    case "WIP": return "badge-warn";
    case "EQP": return "badge-err";
    case "NO_WIP":
    case "NO_EQP": return "badge-info";
    default: return "badge-accent";
  }
}

function num(n: number): string {
  return n.toLocaleString();
}

function TableToolbar({
  query, onQueryChange, shown, total, placeholder,
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
  label, sortKey, currentKey, currentDir, onSort, className = "",
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

function filterBucket(row: CapacityBucketRow, query: string): boolean {
  const blob = [
    row.ppk, row.oper, row.reasonCd, row.allocMode,
    row.planQty, row.remainQty, row.reqTotal, row.planTotal, row.allocTotal,
  ].join(" ").toLowerCase();
  return blob.includes(query);
}

function sortBucket(a: CapacityBucketRow, b: CapacityBucketRow, key: string, dir: "asc" | "desc") {
  switch (key) {
    case "ppk": return compareStrings(a.ppk, b.ppk, dir);
    case "oper": return compareStrings(a.oper, b.oper, dir);
    case "planQty": return compareNumbers(a.planQty, b.planQty, dir);
    case "capaQty": return compareNumbers(a.capaQty, b.capaQty, dir);
    case "remainQty": return compareNumbers(a.remainQty, b.remainQty, dir);
    case "reqTotal": return compareNumbers(a.reqTotal, b.reqTotal, dir);
    case "planTotal": return compareNumbers(a.planTotal, b.planTotal, dir);
    case "allocTotal": return compareNumbers(a.allocTotal, b.allocTotal, dir);
    case "shortfallRatio": return compareNumbers(a.shortfallRatio, b.shortfallRatio, dir);
    case "reasonCd": return compareStrings(a.reasonCd, b.reasonCd, dir);
    default: return compareStrings(a.ppk, b.ppk, dir) || compareStrings(a.oper, b.oper, dir);
  }
}

export default function CapacityPanel({ rows }: Props) {
  const matrix = useMemo(() => buildCapacityMatrix(rows), [rows]);
  const { models, buckets } = matrix;

  const table = useTableFilterSort(buckets, filterBucket, sortBucket, { key: "ppk", dir: "asc" });

  const totals = useMemo(() => {
    const planQty = buckets.reduce((s, b) => s + b.planQty, 0);
    const capaQty = buckets.reduce((s, b) => s + b.capaQty, 0);
    const req = buckets.reduce((s, b) => s + b.reqTotal, 0);
    const plan = buckets.reduce((s, b) => s + b.planTotal, 0);
    const alloc = buckets.reduce((s, b) => s + b.allocTotal, 0);
    const shortBuckets = buckets.filter((b) => b.capaQty < b.planQty - 1e-6).length;
    const maskOn = buckets.some((b) => b.maskApplied);
    const mode = buckets[0]?.allocMode ?? "CAP";
    return { planQty, capaQty, req, plan, alloc, shortBuckets, maskOn, mode };
  }, [buckets]);

  if (buckets.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">◌</div>
        <p>이번 결과에는 적정 장비 대수 산출 데이터가 없습니다.</p>
      </div>
    );
  }

  return (
    <div className="gantt-summary">
      <div className="gantt-summary-kpi card">
        <div className="gantt-summary-kpi-grid">
          <div><span className="gantt-summary-kpi-label">마스킹</span><strong className={totals.maskOn ? "ok" : "bad"}>{totals.maskOn ? "ON" : "OFF"}</strong></div>
          <div><span className="gantt-summary-kpi-label">모드</span><strong>{totals.mode === "ALLOC" ? "allocate" : "cap"}</strong></div>
          <div><span className="gantt-summary-kpi-label">계획 합계</span><strong>{num(totals.planQty)}매</strong></div>
          <div><span className="gantt-summary-kpi-label">Capa 합계</span><strong className={totals.capaQty >= totals.planQty - 1e-6 ? "ok" : "bad"}>{num(Math.round(totals.capaQty))}매</strong></div>
          <div><span className="gantt-summary-kpi-label">필요 대수</span><strong>{totals.req}대</strong></div>
          <div><span className="gantt-summary-kpi-label">정원/배분 대수</span><strong>{totals.plan}대</strong></div>
          <div><span className="gantt-summary-kpi-label">실배치 대수</span><strong>{totals.alloc}대</strong></div>
          <div><span className="gantt-summary-kpi-label">계획 미달 버킷</span><strong className={totals.shortBuckets > 0 ? "warn" : "ok"}>{totals.shortBuckets}개</strong></div>
        </div>
      </div>

      <div className="gantt-summary-eqp card">
        <div className="gantt-summary-section-title">매트릭스 — 계획 · Capa · 모델별 정원/배분 대수</div>
        <div className="table-wrap gantt-summary-scroll">
          <table className="gantt-summary-table capacity-matrix-table">
            <thead>
              <tr>
                <th>PPK / OPER</th>
                <th className="num">계획</th>
                <th className="num">Capa</th>
                {models.map((m) => <th key={m} className="num">{m}</th>)}
                <th>사유</th>
              </tr>
            </thead>
            <tbody>
              {buckets.map((b) => {
                const full = b.capaQty >= b.planQty - 1e-6;
                return (
                  <tr key={b.key}>
                    <td className="mono cell-key">{b.ppk}/{b.oper}</td>
                    <td className="mono num">{num(b.planQty)}</td>
                    <td className={`mono num ${full ? "ok" : "bad"}`}>{num(Math.round(b.capaQty))}</td>
                    {models.map((m) => {
                      const cnt = b.perModel[m]?.alloc ?? 0;
                      return (
                        <td key={m} className="mono num">
                          {cnt > 0 ? cnt : <span className="cell-dim">–</span>}
                        </td>
                      );
                    })}
                    <td>
                      <span className={`badge ${reasonBadgeClass(b.reasonCd)}`} title={capacityReasonLabel(b.reasonCd)}>
                        {b.reasonCd}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
            <tfoot>
              <tr className="capacity-matrix-total">
                <td className="mono cell-key">합계</td>
                <td className="mono num">{num(totals.planQty)}</td>
                <td className="mono num">{num(Math.round(totals.capaQty))}</td>
                {models.map((m) => (
                  <td key={m} className="mono num">
                    {buckets.reduce((s, b) => s + (b.perModel[m]?.alloc ?? 0), 0)}
                  </td>
                ))}
                <td />
              </tr>
            </tfoot>
          </table>
        </div>
        <p className="hint" style={{ marginTop: "0.5rem" }}>
          Capa = 실적 + 잔여량 × (1 − 부족률) — 지금 정원/배분으로 마감까지 실제 만들 수 있는 매수.
          계획보다 낮으면 붉게 표시됩니다. 모델 열은 그 버킷에 실제 배치된 대수입니다.
        </p>
      </div>

      <div className="gantt-summary-plan card">
        <div className="gantt-summary-section-title">상세 — 산출 근거 (RTS_EQPCAPA_INF 대응)</div>
        <TableToolbar
          query={table.query}
          onQueryChange={table.setQuery}
          shown={table.filtered.length}
          total={buckets.length}
          placeholder="PPK·OPER·사유 검색"
        />
        <div className="table-wrap gantt-summary-scroll">
          <table className="gantt-summary-table">
            <thead>
              <tr>
                <SortableTh label="PPK" sortKey="ppk" currentKey={table.sort.key} currentDir={table.sort.dir} onSort={table.toggleSort} />
                <SortableTh label="OPER" sortKey="oper" currentKey={table.sort.key} currentDir={table.sort.dir} onSort={table.toggleSort} />
                <SortableTh label="계획" sortKey="planQty" currentKey={table.sort.key} currentDir={table.sort.dir} onSort={table.toggleSort} className="num" />
                <SortableTh label="잔여량" sortKey="remainQty" currentKey={table.sort.key} currentDir={table.sort.dir} onSort={table.toggleSort} className="num" />
                <SortableTh label="Capa" sortKey="capaQty" currentKey={table.sort.key} currentDir={table.sort.dir} onSort={table.toggleSort} className="num" />
                <th className="num">재공</th>
                <SortableTh label="필요대수" sortKey="reqTotal" currentKey={table.sort.key} currentDir={table.sort.dir} onSort={table.toggleSort} className="num" />
                <SortableTh label="정원/배분" sortKey="planTotal" currentKey={table.sort.key} currentDir={table.sort.dir} onSort={table.toggleSort} className="num" />
                <SortableTh label="실배치" sortKey="allocTotal" currentKey={table.sort.key} currentDir={table.sort.dir} onSort={table.toggleSort} className="num" />
                <SortableTh label="부족률" sortKey="shortfallRatio" currentKey={table.sort.key} currentDir={table.sort.dir} onSort={table.toggleSort} className="num" />
                <SortableTh label="사유" sortKey="reasonCd" currentKey={table.sort.key} currentDir={table.sort.dir} onSort={table.toggleSort} />
                <th>구동가능</th>
              </tr>
            </thead>
            <tbody>
              {table.filtered.map((b) => (
                <tr key={b.key}>
                  <td className="cell-key" title={b.ppk}>{b.ppk}</td>
                  <td className="cell-key" title={b.oper}>{b.oper}</td>
                  <td className="mono num">{num(b.planQty)}매</td>
                  <td className="mono num">{num(b.remainQty)}매</td>
                  <td className="mono num">{num(Math.round(b.capaQty))}매</td>
                  <td className="mono num">{num(b.wipQty)}매/{b.wipCarrierCnt}c</td>
                  <td className="mono num">{b.reqTotal}대</td>
                  <td className="mono num">{b.planTotal}대</td>
                  <td className="mono num">{b.allocTotal}대</td>
                  <td className={`mono num ${b.shortfallRatio > 0 ? "warn" : "ok"}`}>{(b.shortfallRatio * 100).toFixed(1)}%</td>
                  <td>
                    <span className={`badge ${reasonBadgeClass(b.reasonCd)}`} title={capacityReasonLabel(b.reasonCd)}>
                      {b.reasonCd}
                    </span>
                  </td>
                  <td className={b.runnable ? "ok" : "bad"}>{b.runnable ? "Y" : "N"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {table.filtered.length === 0 && (
            <p className="gantt-table-empty">검색 결과가 없습니다.</p>
          )}
        </div>
      </div>
    </div>
  );
}
