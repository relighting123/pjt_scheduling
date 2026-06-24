import { useEffect, useMemo, useState } from "react";

interface Props {
  errors?: string[];
  warnings?: string[];
}

type MsgGroup = { source: string; messages: string[] };

function expandMessages(msgs: string[]): string[] {
  return msgs.flatMap((msg) =>
    msg.split("\n").map((line) => line.trim()).filter(Boolean),
  );
}

function parseGroups(msgs: string[]): MsgGroup[] {
  const expanded = expandMessages(msgs);
  const grouped: Record<string, string[]> = {};
  for (const msg of expanded) {
    const colonIdx = msg.indexOf(":");
    const source = colonIdx > 0 ? msg.slice(0, colonIdx).replace(/\[\d+\]$/, "") : "데이터";
    (grouped[source] ??= []).push(msg);
  }
  return Object.entries(grouped).map(([source, messages]) => ({ source, messages }));
}

function previewText(msgs: string[]): string {
  const first = expandMessages(msgs)[0];
  if (!first) return "";
  return first.length > 100 ? `${first.slice(0, 100)}…` : first;
}

export default function ErrorLogPanel({ errors = [], warnings = [] }: Props) {
  const [open, setOpen] = useState(false);
  const expandedErrors = useMemo(() => expandMessages(errors), [errors]);
  const expandedWarnings = useMemo(() => expandMessages(warnings), [warnings]);
  const errGroups = useMemo(() => parseGroups(errors), [errors]);
  const warnGroups = useMemo(() => parseGroups(warnings), [warnings]);

  const hasErrors = expandedErrors.length > 0;
  const hasWarnings = expandedWarnings.length > 0;

  useEffect(() => {
    if (hasErrors) setOpen(true);
  }, [hasErrors, errors]);

  if (!hasErrors && !hasWarnings) return null;

  const totalCount = expandedErrors.length + expandedWarnings.length;

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
          {hasErrors && (
            <>
              <strong>{expandedErrors.length}개 오류</strong>
              {hasWarnings ? ` · ${expandedWarnings.length}개 경고` : ""}
            </>
          )}
          {!hasErrors && hasWarnings && (
            <>
              <strong>{expandedWarnings.length}개 경고</strong> — 데이터 필드 일부 누락 (자동 보정됨)
            </>
          )}
          {!open && (
            <span className="error-log-preview">
              {previewText(hasErrors ? errors : warnings)}
            </span>
          )}
        </span>
        <span className="error-log-chevron" title={open ? "접기" : "클릭하여 상세 보기"}>
          {open ? "▲" : "▼"}
        </span>
      </button>

      {!open && (
        <div className="error-log-collapsed-hint">클릭하여 상세 보기</div>
      )}

      {open && (
        <div className="error-log-body">
          {errGroups.map(({ source, messages }) => (
            <div key={source} className="error-log-group">
              <div className="error-log-group-title error">
                {source}
                <span className="error-log-count">{messages.length}건</span>
              </div>
              <ul className="error-log-list">
                {messages.map((m, i) => (
                  <li key={i} className="error-log-item error">{m}</li>
                ))}
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
                {messages.map((m, i) => (
                  <li key={i} className="error-log-item warn">{m}</li>
                ))}
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
