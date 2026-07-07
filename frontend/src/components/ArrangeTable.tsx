import type { ArrangeRow, AssignedLot } from "../types";

interface ArrangeTableProps {
  rows: ArrangeRow[];
  assigned: AssignedLot | null | undefined;
  step: number;
  title?: string;
}

function procTime(row: ArrangeRow): number {
  return row.proc_time ?? row.st;
}

export default function ArrangeTable({
  rows,
  assigned,
  step,
  title = "Actual",
}: ArrangeTableProps) {
  const sorted = [...rows].sort(
    (a, b) => a.lot_id.localeCompare(b.lot_id) || a.eqp_id.localeCompare(b.eqp_id),
  );

  const showAssigned = assigned?.kind === "actual" || !assigned?.kind;

  return (
    <div className="arrange-wrap arrange-actual">
      <p className="arrange-meta">
        <span className="arrange-type-badge">{title}</span>
        {" "}스텝 {step} · 구체 조합 {sorted.length}건
        {showAssigned && assigned && (
          <span className="arrange-selected">
            {" "}· 선택: {assigned.lot_id} → {assigned.eqp_id}
            {assigned.oper_id ? ` · ${assigned.oper_id}` : ""}
            {assigned.lot_cd ? ` · ${assigned.lot_cd}/${assigned.temp ?? ""}` : ""}
            {assigned.conversion ? " · Conversion" : ""}
            {assigned.eqp_model != null ? ` (MODEL ${assigned.eqp_model}` : " ("}
            {assigned.st != null ? `, ST ${assigned.st}분` : ""}
            {assigned.wf_qty != null ? `, ${assigned.wf_qty}매` : ""}, 투입 {assigned.start_tm}분)
          </span>
        )}
      </p>
      {sorted.length === 0 ? (
        <p className="hint">투입 가능한 arrange 조합이 없습니다.</p>
      ) : (
        <table className="arrange-table" key={step}>
          <thead>
            <tr>
              {["EQP_ID", "LOT_ID", "OPER", "LOT_STAT_CD", "LOT_CD", "TEMP", "PLAN_PROD_ATTR_VAL", "EQP MODEL", "ST(분)", "START_TM(분)", "WF_QTY"].map((h) => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((row) => {
              const isSelected =
                assigned &&
                assigned.lot_id === row.lot_id &&
                assigned.eqp_id === row.eqp_id;
              const isSameLotSelected =
                assigned && assigned.lot_id === row.lot_id && !isSelected;

              return (
                <tr
                  key={`${row.eqp_id}-${row.lot_id}-${row.oper_id ?? ""}`}
                  className={
                    isSelected
                      ? "arrange-row-selected"
                      : isSameLotSelected
                        ? "arrange-row-removed"
                        : ""
                  }
                >
                  <td>{row.eqp_id}</td>
                  <td>{row.lot_id}</td>
                  <td>{row.oper_id ?? "-"}</td>
                  <td>{row.lot_stat_cd ?? "WAIT"}</td>
                  <td>{row.lot_cd ?? "-"}</td>
                  <td>{row.temp ?? "-"}</td>
                  <td>{row.PLAN_PROD_ATTR_VAL}</td>
                  <td>{row.eqp_model}</td>
                  <td className="num">{procTime(row)}</td>
                  <td className="num">{row.initial_start_tm ?? "-"}</td>
                  <td>{row.wf_qty}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <p className="arrange-hint">
        초기 재공의 구체적 EQP×LOT 조합입니다. ST는 해당 장비 투입 시 소요시간(분),
        EQP MODEL은 장비 모델, START_TM은 초기 스케줄 시작 시각(분)입니다.
        LOT 투입 시 해당 LOT의 모든 EQP 조합이 제거됩니다.
        LOT_STAT_CD가 WAIT인 재공만 알고리즘이 자유롭게 배정하며, PROC/LOAD/SELE/RESV는
        지정된 EQP_ID에 입력된 순서대로 강제 배정됩니다.
      </p>
    </div>
  );
}
