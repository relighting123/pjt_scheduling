import type { ChartVisibilityItem } from "../hooks/useChartVisibility";

interface ChartVisibilityPanelProps {
  title?: string;
  charts: ChartVisibilityItem[];
  visibility: Record<string, boolean>;
  onToggle: (id: string) => void;
  onShowAll: () => void;
  onHideAll: () => void;
}

export default function ChartVisibilityPanel({
  title = "차트 표시",
  charts,
  visibility,
  onToggle,
  onShowAll,
  onHideAll,
}: ChartVisibilityPanelProps) {
  if (charts.length === 0) return null;

  const hiddenCount = charts.filter((c) => visibility[c.id] === false).length;

  return (
    <details className="chart-visibility-panel card">
      <summary className="chart-visibility-summary">
        {title}
        {hiddenCount > 0 && (
          <span className="chart-visibility-badge">{hiddenCount}개 숨김</span>
        )}
      </summary>
      <div className="chart-visibility-body">
        <div className="chart-visibility-actions">
          <button type="button" className="btn btn-secondary btn-sm" onClick={onShowAll}>
            전체 표시
          </button>
          <button type="button" className="btn btn-secondary btn-sm" onClick={onHideAll}>
            전체 숨김
          </button>
        </div>
        <ul className="chart-visibility-list">
          {charts.map((chart) => (
            <li key={chart.id}>
              <label className="chart-visibility-check">
                <input
                  type="checkbox"
                  checked={visibility[chart.id] !== false}
                  onChange={() => onToggle(chart.id)}
                />
                <span>{chart.title}</span>
              </label>
            </li>
          ))}
        </ul>
        <p className="hint chart-visibility-hint">
          차트 제목 바를 드래그하면 별도 창으로 분리할 수 있습니다.
        </p>
      </div>
    </details>
  );
}
