import { useEffect } from "react";
import { createPortal } from "react-dom";
import type { GanttBarClickInfo } from "../lib/charts";

const KIND_LABELS: Record<GanttBarClickInfo["kind"], string> = {
  assign: "가공 스케줄",
  conversion: "Conversion (전환)",
  downtime: "Downtime (PM/개조)",
};

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
