import { useMemo, useState } from "react";

interface Props {
  errors?: string[];
  warnings?: string[];
}

type MsgGroup = { source: string; messages: string[] };

function parseGroups(msgs: string[]): MsgGroup[] {
  const grouped: Record<string, string[]> = {};
  for (const msg of msgs) {
    // Format: "discrete_arrange[3]: 필드 누락 – {'EQP_MODEL_CD'}"
    const colonIdx = msg.indexOf(":");
    const source = colonIdx > 0 ? msg.slice(0, colonIdx).replace(/\[\d+\]$/, "") : "데이터";
    (grouped[source] ??= []).push(msg);
  }
  return Object.entries(grouped).map(([source, messages]) => ({ source, messages }));
}

export default function ErrorLogPanel({ errors = [], warnings = [] }: Props) {
  const [open, setOpen] = useState(false);
  const errGroups  = useMemo(() => parseGroups(errors),   [errors]);
  const warnGroups = useMemo(() => parseGroups(warnings),  [warnings]);

  const hasErrors   = errors.length > 0;
  const hasWarnings = warnings.length > 0;
  if (!hasErrors && !hasWarnings) return null;

  const totalCount = errors.length + warnings.length;

  return (
    <div className={`error-log-panel${open ? " open" : ""}${hasErrors ? " has-errors" : " has-warnings"}`}>
      <button
        type="button"
        className="error-log-toggle"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="error-log-icon">{hasErrors ? "✕" : "⚠"}</span>
        <span className="error-log-summary">
          {hasErrors && <><strong>{errors.length}개 오류</strong>{hasWarnings ? ` · ${warnings.length}개 경고` : ""}</>}
          {!hasErrors && hasWarnings && <><strong>{warnings.length}개 경고</strong> — 데이터 필드 일부 누락 (자동 보정됨)</>}
        </span>
        <span className="error-log-chevron">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="error-log-body">
          {errGroups.map(({ source, messages }) => (
            <div key={source} className="error-log-group">
              <div className="error-log-group-title error">
                {source}
                <span className="error-log-count">{messages.length}건</span>
              </div>
              <ul className="error-log-list">
                {messages.slice(0, 5).map((m, i) => (
                  <li key={i} className="error-log-item error">{m}</li>
                ))}
                {messages.length > 5 && (
                  <li className="error-log-more">…외 {messages.length - 5}건 더</li>
                )}
              </ul>
            </div>
          ))}

          {warnGroups.map(({ source, messages }) => (
            <div key={source} className="error-log-group">
              <div className="error-log-group-title warn">
                {source}
                <span className="error-log-count">{messages.length}건</span>
                <span className="error-log-fixed">자동 보정</span>
              </div>
              <ul className="error-log-list">
                {messages.slice(0, 3).map((m, i) => (
                  <li key={i} className="error-log-item warn">{m}</li>
                ))}
                {messages.length > 3 && (
                  <li className="error-log-more">…외 {messages.length - 3}건 더</li>
                )}
              </ul>
            </div>
          ))}

          <p className="error-log-hint">
            {hasErrors
              ? "오류를 수정한 후 백엔드를 재시작하세요."
              : `경고 항목은 기본값으로 자동 보정됩니다. 총 ${totalCount}건.`
            }
          </p>
        </div>
      )}
    </div>
  );
}
