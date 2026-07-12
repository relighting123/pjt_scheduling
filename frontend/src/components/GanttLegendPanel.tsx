import { useMemo, useState, type CSSProperties } from "react";
import type { GanttLegendItem } from "../lib/charts";

type LegendSortKey = "label" | "prod" | "oper" | "color";

interface Props {
  items: GanttLegendItem[];
  hiddenKeys: ReadonlySet<string>;
  onToggle: (pairKey: string) => void;
  onShowAll: () => void;
  onHideAll: () => void;
  showConversion?: boolean;
  conversionHidden?: boolean;
  onToggleConversion?: () => void;
  showDowntime?: boolean;
  downtimeHidden?: boolean;
  onToggleDowntime?: () => void;
}

export default function GanttLegendPanel({
  items,
  hiddenKeys,
  onToggle,
  onShowAll,
  onHideAll,
  showConversion = false,
  conversionHidden = false,
  onToggleConversion,
  showDowntime = false,
  downtimeHidden = false,
  onToggleDowntime,
}: Props) {
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<LegendSortKey>("label");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

  const visibleCount = items.filter((item) => !hiddenKeys.has(item.pairKey)).length
    + (showConversion && !conversionHidden ? 1 : 0)
    + (showDowntime && !downtimeHidden ? 1 : 0);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    let list = q
      ? items.filter((item) => (
        item.label.toLowerCase().includes(q)
        || item.prodKey.toLowerCase().includes(q)
        || item.operId.toLowerCase().includes(q)
        || item.color.toLowerCase().includes(q)
      ))
      : [...items];

    list.sort((a, b) => {
      const pick = (item: GanttLegendItem) => {
        switch (sortKey) {
          case "prod": return item.prodKey;
          case "oper": return item.operId;
          case "color": return item.color;
          default: return item.label;
        }
      };
      const cmp = pick(a).localeCompare(pick(b), undefined, { numeric: true, sensitivity: "base" });
      return sortDir === "asc" ? cmp : -cmp;
    });
    return list;
  }, [items, query, sortKey, sortDir]);

  if (!items.length && !showConversion && !showDowntime) return null;

  return (
    <div className="card gantt-legend-panel">
      <div className="gantt-legend-head">
        <div className="gantt-summary-section-title">제품×공정 범례</div>
        <span className="gantt-table-count">
          {visibleCount}/{items.length + (showConversion ? 1 : 0) + (showDowntime ? 1 : 0)} 표시
        </span>
      </div>

      <div className="gantt-table-toolbar">
        <input
          type="search"
          className="gantt-table-search"
          placeholder="코드·PPK·OPER·색상 검색"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <label className="gantt-table-sort">
          <span>정렬</span>
          <select
            value={`${sortKey}:${sortDir}`}
            onChange={(e) => {
              const [key, dir] = e.target.value.split(":") as [LegendSortKey, "asc" | "desc"];
              setSortKey(key);
              setSortDir(dir);
            }}
          >
            <option value="label:asc">조합 ↑</option>
            <option value="label:desc">조합 ↓</option>
            <option value="prod:asc">PPK ↑</option>
            <option value="prod:desc">PPK ↓</option>
            <option value="oper:asc">OPER ↑</option>
            <option value="oper:desc">OPER ↓</option>
            <option value="color:asc">색상 ↑</option>
            <option value="color:desc">색상 ↓</option>
          </select>
        </label>
        <div className="gantt-legend-actions">
          <button type="button" className="btn btn-secondary btn-sm" onClick={onShowAll}>전체 표시</button>
          <button type="button" className="btn btn-secondary btn-sm" onClick={onHideAll}>전체 숨김</button>
        </div>
      </div>

      <div className="gantt-legend-tags" role="list">
        {filtered.map((item) => {
          const hidden = hiddenKeys.has(item.pairKey);
          return (
            <button
              key={item.pairKey}
              type="button"
              role="listitem"
              className={`gantt-legend-tag${hidden ? " hidden" : ""}${!item.inSchedule ? " unused" : ""}`}
              style={{ "--legend-color": item.color } as CSSProperties}
              onClick={() => onToggle(item.pairKey)}
              title={`${item.label} · ${item.prodKey} / ${item.operId}`}
            >
              <span className="gantt-legend-tag-label">{item.label}</span>
              <span className="gantt-legend-tag-meta">{item.color}</span>
            </button>
          );
        })}
        {showConversion && onToggleConversion && (
          <button
            type="button"
            role="listitem"
            className={`gantt-legend-tag conversion${conversionHidden ? " hidden" : ""}`}
            onClick={onToggleConversion}
            title="Conversion"
          >
            <span className="gantt-legend-tag-label">CONV</span>
            <span className="gantt-legend-tag-meta">Conversion</span>
          </button>
        )}
        {showDowntime && onToggleDowntime && (
          <button
            type="button"
            role="listitem"
            className={`gantt-legend-tag downtime${downtimeHidden ? " hidden" : ""}`}
            onClick={onToggleDowntime}
            title="Downtime"
          >
            <span className="gantt-legend-tag-label">DOWN</span>
            <span className="gantt-legend-tag-meta">Downtime</span>
          </button>
        )}
      </div>

      {filtered.length === 0 && (
        <p className="gantt-table-empty">검색 결과가 없습니다.</p>
      )}
    </div>
  );
}
