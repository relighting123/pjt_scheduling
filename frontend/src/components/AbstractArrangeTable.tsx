import type { AbstractArrangeRow, AssignedLot } from "../types";



interface AbstractArrangeTableProps {

  rows: AbstractArrangeRow[];

  assigned: AssignedLot | null | undefined;

  step: number;

  simTime: number;

}



function formatLotUnits(row: AbstractArrangeRow): string {

  const units = row.lot_units ?? [];

  if (!units.length) return "-";

  return units

    .filter((u) => !u.assigned)

    .map((u) => `${u.lot_id}@${u.oper_in_time}`)

    .join(", ") || "(배정됨)";

}



export default function AbstractArrangeTable({

  rows,

  assigned,

  step,

  simTime,

}: AbstractArrangeTableProps) {

  const sorted = [...rows].sort(

    (a, b) =>

      a.min_inject_time - b.min_inject_time

      || a.plan_prod_key.localeCompare(b.plan_prod_key)

      || a.eqp_model.localeCompare(b.eqp_model),

  );



  const selectedLot = assigned?.kind === "abstract" ? assigned.lot_id : undefined;



  return (

    <div className="arrange-wrap arrange-abstract">

      <p className="arrange-meta">

        스텝 {step} · 추상 유입 조합 {sorted.length}건

        <span className="arrange-time"> (시뮬 시각 {simTime}분)</span>

        {assigned?.kind === "abstract" && (

          <span className="arrange-selected">

            {" "}· 선택: {assigned.lot_id} → {assigned.eqp_id} × EQP MODEL {assigned.eqp_model}
            {assigned.st != null ? `, ST ${assigned.st}분` : ""}

            {assigned.oper_in_time != null ? ` (OPER IN ${assigned.oper_in_time}분` : " ("}

            , 투입 {assigned.start_tm}분)

          </span>

        )}

      </p>

      {sorted.length === 0 ? (

        <p className="hint">유입 재공(추상 조합)이 없습니다.</p>

      ) : (

        <table className="arrange-table abstract-table" key={step}>

          <thead>

            <tr>

              {["PPK", "EQP MODEL", "재공수", "처리시간", "최소투입", "OPER", "WF_QTY", "대기 LOT (OPER IN)", "전공정"].map((h) => (

                <th key={h}>{h}</th>

              ))}

            </tr>

          </thead>

          <tbody>

            {sorted.map((row) => {

              const arrived = simTime >= row.min_inject_time;

              const isSelected =

                selectedLot != null

                && row.lot_units?.some((u) => u.lot_id === selectedLot);

              const isPending = arrived && row.wip_qty > 0 && !isSelected;



              return (

                <tr

                  key={row.abs_key}

                  className={

                    isSelected

                      ? "arrange-row-selected"

                      : isPending

                        ? "arrange-row-pending"

                        : arrived && row.wip_qty === 0

                          ? "arrange-row-consumed"

                          : ""

                  }

                >

                  <td>{row.plan_prod_key}</td>

                  <td>{row.eqp_model}</td>

                  <td className="num">{row.wip_qty}</td>

                  <td className="num">{row.proc_time}</td>

                  <td className="num">{row.min_inject_time}</td>

                  <td>{row.oper_id}</td>

                  <td className="num">{row.wf_qty}</td>

                  <td className="lot-units-cell">{formatLotUnits(row)}</td>

                  <td>{row.from_oper ?? "-"}</td>

                </tr>

              );

            })}

          </tbody>

        </table>

      )}

      <p className="arrange-hint">

        전 공정 END_TM 완료 시 다음 공정으로 유입됩니다. 각 추상 조합은 실 LOT ID와 OPER IN TIME을 lot_units로 관리하며,

        배정 시 ABS가 아닌 실제 LOT ID가 스케줄에 기록됩니다.

      </p>

    </div>

  );

}


