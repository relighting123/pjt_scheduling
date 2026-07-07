import { useEffect, useState } from "react";
import type { SimEvent, SimEventKind } from "../types";
import {
  formatSimEventExtra,
  normalizeSimEventKind,
  simEventIcon,
  simEventLabel,
  SIM_EVENT_CLASS,
} from "../lib/simEvents";

interface EventTimelineProps {
  events: SimEvent[];
  highlightKinds?: Set<string>;
  title?: string;
}

const ROWS = 200;

const COLS = [
  { key: "time", label: "TIME", num: true },
  { key: "event", label: "EVENT" },
  { key: "eqp_id", label: "EQP_ID" },
  { key: "lot_id", label: "LOT_ID" },
  { key: "plan", label: "PLAN_PROD_ATTR_VAL" },
  { key: "oper_id", label: "OPER_ID" },
  { key: "extra", label: "DETAIL" },
] as const;

export function EventTimeline({
  events,
  highlightKinds,
  title = "시뮬레이션 이벤트",
}: EventTimelineProps) {
  const [page, setPage] = useState(0);
  const total = events.length;
  const pages = Math.max(1, Math.ceil(total / ROWS));
  const visible = events.slice(page * ROWS, (page + 1) * ROWS);

  useEffect(() => {
    setPage(0);
  }, [events]);

  if (!events.length) {
    return (
      <section className="card event-timeline">
        {title ? <h3>{title}</h3> : null}
        <p className="hint">이 단계에 기록된 이벤트가 없습니다.</p>
      </section>
    );
  }

  return (
    <>
      <div className="vtable-header">
        <span className="vtable-count">총 {total.toLocaleString()}건</span>
        {pages > 1 && (
          <div className="vtable-pagination">
            <button type="button" className="btn btn-ghost btn-xs" disabled={page === 0} onClick={() => setPage(0)}>«</button>
            <button type="button" className="btn btn-ghost btn-xs" disabled={page === 0} onClick={() => setPage(p => p - 1)}>‹</button>
            <span className="page-info">{page + 1} / {pages}</span>
            <button type="button" className="btn btn-ghost btn-xs" disabled={page >= pages - 1} onClick={() => setPage(p => p + 1)}>›</button>
            <button type="button" className="btn btn-ghost btn-xs" disabled={page >= pages - 1} onClick={() => setPage(pages - 1)}>»</button>
          </div>
        )}
      </div>
      <div className="table-wrap">
        <table className="data-table event-data-table">
          <thead>
            <tr>
              {COLS.map(col => (
                <th key={col.key} className={"num" in col && col.num ? "num" : undefined}>{col.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visible.map((ev, i) => {
              const kind = normalizeSimEventKind(ev.kind) as SimEventKind;
              const label = simEventLabel(kind, true);
              const icon = simEventIcon(kind);
              const rowClass = [
                SIM_EVENT_CLASS[kind] ?? "",
                highlightKinds?.has(kind) ? "evt-highlight" : "",
              ]
                .filter(Boolean)
                .join(" ");
              return (
                <tr key={`${ev.time}-${ev.kind}-${ev.eqp_id}-${i}`} className={rowClass}>
                  <td className="num">{ev.time}</td>
                  <td>
                    <span className={`evt-cell ${SIM_EVENT_CLASS[kind] ?? ""}`}>
                      <span className="evt-icon" aria-hidden="true">{icon}</span>
                      <span className="evt-label">{label}</span>
                    </span>
                  </td>
                  <td>{ev.eqp_id || "—"}</td>
                  <td>{ev.lot_id || "—"}</td>
                  <td>{ev.PLAN_PROD_ATTR_VAL || "—"}</td>
                  <td>{ev.oper_id || "—"}</td>
                  <td className="evt-extra">{formatSimEventExtra(ev)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}
