import type { DecisionLogEntry } from "../types";

const STATUS_LABELS: Record<string, string> = {
  assigned: "배정",
  action_corrected: "보정 배정",
  assign_failed: "배정 실패",
  no_feasible: "feasible 없음",
  no_idle_eqp: "EQP 대기",
  eqp_not_idle: "EQP 비idle",
  sim_done: "종료",
};

interface DecisionLogTableProps {
  entries: DecisionLogEntry[];
  highlightStep?: number;
  maxHeight?: number;
  title?: string;
}

function formatFeasible(entry: DecisionLogEntry): string {
  if (!entry.feasible_options?.length) return "—";
  return entry.feasible_options
    .map((o) => `${o.ppk}/${o.oper_id}${o.lot_id ? `(${o.lot_id})` : ""}`)
    .join(", ");
}

function formatBlocked(entry: DecisionLogEntry): string {
  if (!entry.blocked_buckets?.length) return "—";
  return entry.blocked_buckets
    .map((b) => `${b.ppk}/${b.oper_id}: ${b.detail}`)
    .join(" · ");
}

export default function DecisionLogTable({
  entries,
  highlightStep,
  maxHeight = 360,
  title = "결정 로그 (step별 EQP / PPK / OPER)",
}: DecisionLogTableProps) {
  if (!entries.length) {
    return (
      <section className="card decision-log">
        <h3>{title}</h3>
        <p className="hint">결정 로그가 없습니다. 추론 시 「결정 로그」 옵션을 켜고 실행하세요.</p>
      </section>
    );
  }

  return (
    <section className="card decision-log">
      <h3>{title}</h3>
      <p className="hint page-lead">
        각 RL/휴리스틱 step에서 대상 EQP, 선택 PPK/OPER, feasible 후보, 미할당 사유를 기록합니다.
      </p>
      <div className="decision-log-scroll" style={{ maxHeight }}>
        <table className="decision-log-table">
          <thead>
            <tr>
              <th>step</th>
              <th>시각</th>
              <th>EQP</th>
              <th>요청 PPK/OPER</th>
              <th>선택 PPK/OPER</th>
              <th>LOT</th>
              <th>상태</th>
              <th>사유 / feasible</th>
              <th>차단 bucket</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((row) => {
              const highlighted = highlightStep != null && row.step === highlightStep;
              return (
                <tr
                  key={`decision-${row.step}`}
                  className={highlighted ? "decision-log-row-active" : undefined}
                >
                  <td>{row.step}</td>
                  <td>
                    {row.sim_time}
                    {row.time_advanced ? `→${row.sim_time_after}` : ""}
                  </td>
                  <td>{row.eqp_id ?? "—"}</td>
                  <td>
                    {row.action_requested_ppk && row.action_requested_oper
                      ? `${row.action_requested_ppk}/${row.action_requested_oper}`
                      : "—"}
                  </td>
                  <td>
                    {row.resolved_ppk && row.resolved_oper
                      ? `${row.resolved_ppk}/${row.resolved_oper}`
                      : "—"}
                  </td>
                  <td>{row.assigned_lot_id ?? "—"}</td>
                  <td>
                    <span className={`decision-status decision-status-${row.status}`}>
                      {STATUS_LABELS[row.status] ?? row.status}
                    </span>
                  </td>
                  <td className="decision-reason">
                    <div>{row.reason}</div>
                    {row.feasible_options?.length ? (
                      <div className="decision-sub">feasible: {formatFeasible(row)}</div>
                    ) : null}
                  </td>
                  <td className="decision-blocked">{formatBlocked(row)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function decisionLogForStep(entries: DecisionLogEntry[], step: number): DecisionLogEntry | undefined {
  return entries.find((row) => row.step === step);
}
