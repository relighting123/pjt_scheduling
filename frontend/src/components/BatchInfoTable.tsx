import type { BatchInfoRecord } from "../types";

interface BatchInfoTableProps {
  rows: BatchInfoRecord[];
  compact?: boolean;
}

export default function BatchInfoTable({ rows, compact = false }: BatchInfoTableProps) {
  const sorted = [...rows].sort(
    (a, b) =>
      a.PLAN_PROD_ATTR_VAL.localeCompare(b.PLAN_PROD_ATTR_VAL) || a.oper_id.localeCompare(b.oper_id),
  );

  if (!sorted.length) {
    return <p className="hint">batch_info 없음 — PPK/OPER별 LOT_CD·TEMP는 lot_master 또는 PPK 추정값을 사용합니다.</p>;
  }

  return (
    <div className={`arrange-wrap batch-info${compact ? " batch-info-compact" : ""}`}>
      {!compact && (
        <p className="arrange-meta">
          (PPK, OPER) 배치 레시피 {sorted.length}건
          <span className="arrange-hint-inline"> · conversion / tool cap lookup</span>
        </p>
      )}
      <table className="arrange-table batch-info-table">
        <thead>
          <tr>
            {["PPK", "OPER", "LOT_CD", "TEMP"].map((h) => (
              <th key={h}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr key={`${row.PLAN_PROD_ATTR_VAL}|${row.oper_id}`}>
              <td>{row.PLAN_PROD_ATTR_VAL}</td>
              <td>{row.oper_id}</td>
              <td><code>{row.lot_cd}</code></td>
              <td><code>{row.temp}</code></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
