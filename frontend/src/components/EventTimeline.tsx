import type { SimEvent, SimEventKind } from "../types";
import {
  formatSimEventDetail,
  normalizeSimEventKind,
  simEventLabel,
  SIM_EVENT_CLASS,
} from "../lib/simEvents";

interface EventTimelineProps {
  events: SimEvent[];
  highlightKinds?: Set<string>;
  maxHeight?: number;
  title?: string;
}

export function EventTimeline({
  events,
  highlightKinds,
  maxHeight = 320,
  title = "시뮬레이션 이벤트",
}: EventTimelineProps) {
  if (!events.length) {
    return (
      <section className="card event-timeline">
        <h3>{title}</h3>
        <p className="hint">이 단계에 기록된 이벤트가 없습니다.</p>
      </section>
    );
  }

  return (
    <section className="card event-timeline">
      <h3>{title}</h3>
      <div className="event-timeline-scroll" style={{ maxHeight }}>
        <table className="event-table">
          <thead>
            <tr>
              <th>시각(분)</th>
              <th>이벤트</th>
              <th>장비</th>
              <th>상세</th>
            </tr>
          </thead>
          <tbody>
            {events.map((ev, i) => {
              const kind = normalizeSimEventKind(ev.kind) as SimEventKind;
              const label = simEventLabel(kind, true);
              const rowClass = [
                SIM_EVENT_CLASS[kind] ?? "",
                highlightKinds?.has(kind) ? "evt-highlight" : "",
              ]
                .filter(Boolean)
                .join(" ");
              return (
                <tr key={`${ev.time}-${ev.kind}-${ev.eqp_id}-${i}`} className={rowClass}>
                  <td className="evt-time">{ev.time}</td>
                  <td>
                    <span className={`evt-badge ${SIM_EVENT_CLASS[kind] ?? ""}`}>
                      <code className="evt-code">{simEventLabel(kind)}</code>
                      <span className="evt-label-ko">{label}</span>
                    </span>
                  </td>
                  <td>{ev.eqp_id || "—"}</td>
                  <td className="evt-detail">{formatSimEventDetail(ev) || "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
