import { useEffect, useMemo, useState } from "react";
import type { DecisionLogEntry } from "../types";

const STATUS_LABELS: Record<string, string> = {
  assigned: "배정 완료",
  action_corrected: "보정 배정",
  assign_failed: "배정 실패",
  no_feasible: "feasible 없음",
  no_idle_eqp: "EQP 대기 (시간 전진)",
  eqp_not_idle: "EQP 비idle",
  sim_done: "시뮬 종료",
};

/** 리워드 항목 한글 라벨 + 부호 의미 */
const TERM_LABELS: Record<string, string> = {
  same_setup: "동일 셋업 연속",
  pacing: "페이싱(takt 추종)",
  plan_hit: "계획 달성 진척",
  flow_balance: "Flow balance",
  conversion: "전환 패널티",
  avoidable_conversion: "회피가능 전환",
  idle: "Idle 패널티",
  bulk_block_bonus: "벌크 블록 보너스",
  dedication_misuse: "전용 오용",
  redundant_cover: "중복 커버",
};

interface Props {
  entries: DecisionLogEntry[];
  /** 현재 스텝 변경 시 부모에 통지 (간트 동기화용) */
  onStepChange?: (entry: DecisionLogEntry | null) => void;
}

export default function StepDebugger({ entries, onStepChange }: Props) {
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [onlyAssigned, setOnlyAssigned] = useState(false);

  const view = useMemo(
    () => (onlyAssigned ? entries.filter((e) => e.status === "assigned" || e.status === "action_corrected") : entries),
    [entries, onlyAssigned],
  );

  useEffect(() => { setIdx(0); }, [onlyAssigned, entries]);

  const clampedIdx = Math.min(idx, Math.max(view.length - 1, 0));
  const cur: DecisionLogEntry | undefined = view[clampedIdx];

  useEffect(() => { onStepChange?.(cur ?? null); }, [cur, onStepChange]);

  useEffect(() => {
    if (!playing) return;
    if (clampedIdx >= view.length - 1) { setPlaying(false); return; }
    const t = setTimeout(() => setIdx((i) => Math.min(i + 1, view.length - 1)), 700);
    return () => clearTimeout(t);
  }, [playing, clampedIdx, view.length]);

  if (!entries.length) {
    return (
      <div className="card">
        <div className="card-title">스텝 디버거</div>
        <p className="hint">결정 로그가 없습니다. 좌측에서 「결정 로그 포함」을 켜고 추론을 실행하세요.</p>
      </div>
    );
  }

  const bd = cur?.reward_breakdown ?? {};
  const terms = Object.entries(bd);
  const bdSum = terms.reduce((a, [, v]) => a + v, 0);
  const maxAbs = Math.max(0.01, ...terms.map(([, v]) => Math.abs(v)));

  const go = (n: number) => { setPlaying(false); setIdx(Math.max(0, Math.min(n, view.length - 1))); };

  return (
    <div className="step-debugger">
      {/* 네비게이션 */}
      <div className="card stepdbg-nav">
        <div className="stepdbg-nav-row">
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => go(0)} disabled={clampedIdx === 0}>⏮</button>
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => go(clampedIdx - 1)} disabled={clampedIdx === 0}>◀</button>
          <button type="button" className={`btn btn-sm ${playing ? "btn-accent" : "btn-primary"}`} onClick={() => setPlaying((p) => !p)}>
            {playing ? "⏸ 일시정지" : "▶ 재생"}
          </button>
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => go(clampedIdx + 1)} disabled={clampedIdx >= view.length - 1}>▶</button>
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => go(view.length - 1)} disabled={clampedIdx >= view.length - 1}>⏭</button>
          <span className="stepdbg-counter">{clampedIdx + 1} / {view.length}</span>
          <label className="check-label stepdbg-filter">
            <input type="checkbox" checked={onlyAssigned} onChange={(e) => setOnlyAssigned(e.target.checked)} />
            배정만
          </label>
        </div>
        <input
          type="range" className="stepdbg-slider" min={0} max={Math.max(view.length - 1, 0)}
          value={clampedIdx} onChange={(e) => go(Number(e.target.value))}
        />
      </div>

      {cur && (
        <div className="grid-2 stepdbg-body">
          {/* 결정 상세 */}
          <div className="card">
            <div className="card-title">결정 상세 · step {cur.step}</div>
            <table className="kv-table">
              <tbody>
                <tr><th>시각(분)</th><td>{cur.sim_time}{cur.time_advanced ? ` → ${cur.sim_time_after}` : ""}</td></tr>
                <tr><th>대상 EQP</th><td><b>{cur.selected_eqp_id ?? cur.eqp_id ?? "—"}</b></td></tr>
                <tr>
                  <th>요청 → 선택 PPK/OPER</th>
                  <td>
                    {cur.action_requested_ppk ? `${cur.action_requested_ppk}/${cur.action_requested_oper}` : "—"}
                    {"  →  "}
                    <b>{(cur.selected_ppk ?? cur.resolved_ppk) ? `${cur.selected_ppk ?? cur.resolved_ppk}/${cur.selected_oper_id ?? cur.resolved_oper}` : "—"}</b>
                    {cur.action_corrected ? <span className="stepdbg-tag stepdbg-tag-warn">보정</span> : null}
                  </td>
                </tr>
                <tr><th>선택 LOT</th><td><b>{cur.selected_lot_id ?? cur.assigned_lot_id ?? "—"}</b></td></tr>
                <tr>
                  <th>블록</th>
                  <td>
                    {cur.block_start
                      ? <span className="stepdbg-tag stepdbg-tag-accent">블록 시작 · N={cur.block_size ?? "?"} (lv {cur.size_level ?? 0})</span>
                      : cur.status === "assigned" ? <span className="stepdbg-tag">블록 연속</span> : "—"}
                  </td>
                </tr>
                <tr>
                  <th>상태</th>
                  <td><span className={`decision-status decision-status-${cur.status}`}>{STATUS_LABELS[cur.status] ?? cur.status}</span></td>
                </tr>
                <tr><th>사유</th><td>{cur.selection_reason ?? cur.reason}</td></tr>
              </tbody>
            </table>

            {cur.feasible_options?.length ? (
              <div className="stepdbg-list">
                <div className="stepdbg-list-title">feasible 후보 ({cur.feasible_options.length})</div>
                <div className="stepdbg-chips">
                  {cur.feasible_options.map((o, i) => (
                    <span key={i} className={`stepdbg-chip${(o.ppk === (cur.selected_ppk ?? cur.resolved_ppk) && o.oper_id === (cur.selected_oper_id ?? cur.resolved_oper)) ? " stepdbg-chip-sel" : ""}`}>
                      {o.ppk}/{o.oper_id}{o.lot_id ? ` · ${o.lot_id}` : ""}
                    </span>
                  ))}
                </div>
              </div>
            ) : null}

            {cur.blocked_buckets?.length ? (
              <div className="stepdbg-list">
                <div className="stepdbg-list-title">차단된 bucket ({cur.blocked_buckets.length})</div>
                <div className="stepdbg-blocked">
                  {cur.blocked_buckets.slice(0, 6).map((b, i) => (
                    <div key={i}><b>{b.ppk}/{b.oper_id}</b> — {b.detail}</div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>

          {/* 리워드 분해 */}
          <div className="card">
            <div className="card-title">리워드 항목별 분해</div>
            {terms.length ? (
              <>
                <table className="reward-table">
                  <tbody>
                    {terms.map(([k, v]) => (
                      <tr key={k}>
                        <td className="reward-term">{TERM_LABELS[k] ?? k}</td>
                        <td className="reward-bar-cell">
                          <div className="reward-bar-track">
                            <div
                              className={`reward-bar ${v >= 0 ? "reward-bar-pos" : "reward-bar-neg"}`}
                              style={{ width: `${(Math.abs(v) / maxAbs) * 100}%` }}
                            />
                          </div>
                        </td>
                        <td className={`reward-val ${v >= 0 ? "reward-pos" : "reward-neg"}`}>{v >= 0 ? "+" : ""}{v.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                  <tfoot>
                    <tr>
                      <td className="reward-term"><b>합계 (클립 전)</b></td>
                      <td />
                      <td className={`reward-val ${bdSum >= 0 ? "reward-pos" : "reward-neg"}`}><b>{bdSum >= 0 ? "+" : ""}{bdSum.toFixed(3)}</b></td>
                    </tr>
                  </tfoot>
                </table>
                <p className="hint reward-note">
                  최종 스텝 리워드: <b>{cur.reward?.toFixed(3)}</b>
                  {Math.abs((cur.reward ?? 0) - bdSum) > 0.01 ? "  (※ ±10 클립 적용 전 합계와 차이)" : ""}
                </p>
              </>
            ) : (
              <p className="hint">이 스텝은 배정이 없어 리워드 분해가 없습니다 (상태: {STATUS_LABELS[cur.status] ?? cur.status}).</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
