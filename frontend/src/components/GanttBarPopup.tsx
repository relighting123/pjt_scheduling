import { useEffect } from "react";
import { createPortal } from "react-dom";
import type { GanttBarClickInfo } from "../lib/charts";

const KIND_LABELS: Record<GanttBarClickInfo["kind"], string> = {
  assign: "가공 스케줄",
  conversion: "Conversion (전환)",
  downtime: "Downtime (PM/개조)",
};

/**
 * 간트 바 hover 시 커서 옆에 뜨는 짧은 요약. Plotly 내장 hoverlayer는 스크롤
 * 컨테이너 클리핑과 얽혀 깨지는 버그가 있어(PlotChart.tsx 참고) 대신
 * customdata 기반으로 직접 그리는 가벼운 요약이다 — 전체 상세는 클릭 팝업에서.
 */
export function GanttHoverSummary({ info }: { info: GanttBarClickInfo }) {
  const duration = Math.max(info.end_min - info.start_min, 0);
  const records = info.records ?? [];
  const totalWf = records.reduce((s, r) => s + (r.wf_qty ?? 0), 0);

  return (
    <div className="gantt-hover-tip">
      <div className="gantt-hover-tip-head">
        <span className={`gantt-popup-kind gantt-popup-kind-${info.kind}`}>{KIND_LABELS[info.kind]}</span>
        <b>{info.eqp_label || info.eqp_id}</b>
      </div>
      <div className="gantt-hover-tip-row">
        {info.start_label} → {info.end_label}
        {info.kind === "downtime" && info.unbounded ? " · 무제한" : ` · ${duration}분`}
      </div>
      {info.kind === "assign" && info.prod ? (
        <div className="gantt-hover-tip-row">
          {info.prod_code ? `${info.prod_code} ` : ""}({info.prod}) / {info.oper_code ? `${info.oper_code} ` : ""}({info.oper})
        </div>
      ) : null}
      {info.kind === "assign" && records.length ? (
        <div className="gantt-hover-tip-row">{records.length}건 · 총 {totalWf}매</div>
      ) : null}
      {info.kind === "conversion" && info.transition ? (
        <div className="gantt-hover-tip-row">{info.transition}</div>
      ) : null}
      <div className="gantt-hover-tip-hint">클릭하면 상세 정보</div>
    </div>
  );
}

interface Props {
  info: GanttBarClickInfo;
  onClose: () => void;
}

/** 간트 바 클릭 시 해당 구간 상세를 보여주는 팝업 (hover 툴팁 대체) */
export default function GanttBarPopup({ info, onClose }: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const duration = Math.max(info.end_min - info.start_min, 0);
  const records = info.records ?? [];
  const totalWf = records.reduce((s, r) => s + (r.wf_qty ?? 0), 0);

  // 페이지 전환 애니메이션 등 transform이 걸린 조상 밑에서는 position:fixed가
  // 뷰포트 기준이 아니게 되므로, body에 portal로 렌더해 화면 중앙 고정을 보장한다.
  return createPortal(
    <div className="gantt-popup-overlay" onClick={onClose}>
      <div
        className="gantt-popup card"
        role="dialog"
        aria-modal="true"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="gantt-popup-head">
          <div className="gantt-popup-title">
            <span className={`gantt-popup-kind gantt-popup-kind-${info.kind}`}>
              {KIND_LABELS[info.kind]}
            </span>
            <b>{info.eqp_label || info.eqp_id}</b>
          </div>
          <button type="button" className="btn btn-ghost btn-sm" onClick={onClose} aria-label="닫기">
            ✕
          </button>
        </div>

        <table className="kv-table">
          <tbody>
            <tr>
              <th>구간</th>
              <td>
                {info.start_label} → {info.end_label}
                {info.kind === "downtime" && info.unbounded ? "" : ` · ${duration}분`}
              </td>
            </tr>
            {info.kind === "assign" && info.prod ? (
              <tr>
                <th>제품 / 공정</th>
                <td>
                  {info.prod_code ? `${info.prod_code} ` : ""}({info.prod}) / {info.oper_code ? `${info.oper_code} ` : ""}({info.oper})
                </td>
              </tr>
            ) : null}
            {info.kind === "assign" && records.length ? (
              <tr>
                <th>LOT / 수량</th>
                <td>{records.length}건 · 총 {totalWf}매</td>
              </tr>
            ) : null}
            {info.kind === "conversion" && info.transition ? (
              <tr><th>전환</th><td>{info.transition}</td></tr>
            ) : null}
            {info.kind === "downtime" ? (
              <tr><th>종료</th><td>{info.unbounded ? "무제한 (DOWN_END_TM 없음)" : info.end_label}</td></tr>
            ) : null}
          </tbody>
        </table>

        {records.length ? (
          <div className="gantt-popup-lots">
            <table className="data-table gantt-popup-lot-table">
              <thead>
                <tr>
                  <th>LOT</th>
                  <th>CAR</th>
                  <th>LOT_CD/TEMP</th>
                  <th>시작 → 종료</th>
                  <th>수량</th>
                  <th>비고</th>
                </tr>
              </thead>
              <tbody>
                {records.map((r, i) => (
                  <tr key={i}>
                    <td><b>{r.lot_id}</b></td>
                    <td>{r.carrier_id ?? "—"}</td>
                    <td>{r.lot_cd ?? "—"}{r.temp ? ` / ${r.temp}` : ""}</td>
                    <td>{r.start_label} → {r.end_label}</td>
                    <td>{r.wf_qty ?? "—"}</td>
                    <td>
                      {[
                        r.lot_stat_cd && r.lot_stat_cd !== "WAIT" ? r.lot_stat_cd : null,
                        r.inflow ? "유입 재공" : null,
                      ].filter(Boolean).join(" · ") || "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}

        <p className="hint gantt-popup-hint">바깥 영역 클릭 또는 ESC로 닫기</p>
      </div>
    </div>,
    document.body,
  );
}
