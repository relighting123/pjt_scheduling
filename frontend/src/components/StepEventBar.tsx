import type { SimEvent } from "../types";
import {
  formatSimEventDetail,
  normalizeSimEventKind,
  simEventLabel,
  SIM_EVENT_CLASS,
} from "../lib/simEvents";

interface StepEventBarProps {
  step: number;
  simTime?: number;
  events: SimEvent[];
}

export function StepEventBar({ step, simTime, events }: StepEventBarProps) {
  return (
    <div className="step-event-bar" aria-live="polite">
      <div className="step-event-bar-meta">
        <span className="step-event-bar-title">이 스텝 이벤트</span>
        <span className="step-event-bar-time">
          step {step}
          {simTime != null ? ` · ${simTime}분` : ""}
        </span>
      </div>
      {events.length === 0 ? (
        <p className="step-event-bar-empty hint">이 스텝에 기록된 이벤트 없음</p>
      ) : (
        <ul className="step-event-bar-list">
          {events.map((ev, i) => {
            const kind = normalizeSimEventKind(ev.kind);
            const cls = SIM_EVENT_CLASS[kind as SimEventKind] ?? "";
            return (
              <li key={`${ev.time}-${kind}-${ev.eqp_id}-${i}`} className="step-event-bar-item">
                <span className={`evt-badge ${cls}`} title={formatSimEventDetail(ev)}>
                  <code className="evt-code">{simEventLabel(kind)}</code>
                </span>
                {ev.eqp_id ? <span className="step-event-bar-eqp">{ev.eqp_id}</span> : null}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
