interface ChartSettingsPanelProps {
  dataTimeEnd: number;
  ganttTimeFixed: boolean;
  ganttTimeStart: number;
  ganttTimeEnd: number;
  onGanttFixedChange: (fixed: boolean) => void;
  onGanttStartChange: (minutes: number) => void;
  onGanttEndChange: (minutes: number) => void;
  showReplay?: boolean;
  step?: number;
  maxStep?: number;
  stepBump?: boolean;
  onStepChange?: (step: number) => void;
}

export default function ChartSettingsPanel({
  dataTimeEnd,
  ganttTimeFixed,
  ganttTimeStart,
  ganttTimeEnd,
  onGanttFixedChange,
  onGanttStartChange,
  onGanttEndChange,
  showReplay = false,
  step = 0,
  maxStep = 0,
  stepBump = false,
  onStepChange,
}: ChartSettingsPanelProps) {
  return (
    <details className="chart-settings card" open>
      <summary className="chart-settings-summary">차트 설정</summary>

      <div className="chart-settings-body">
        <section className="chart-settings-section">
          <h4>X축 시간</h4>
          <p className="hint">
            고정을 켜면 간트·생산량 차트 X축이 지정 구간으로 고정됩니다.
            {dataTimeEnd > 0 && <> 시뮬 종료: <strong>{dataTimeEnd}분</strong></>}
          </p>
          <label className="chart-settings-check">
            <input
              type="checkbox"
              checked={ganttTimeFixed}
              onChange={(e) => onGanttFixedChange(e.target.checked)}
            />
            X축 시간 고정
          </label>
          <div className="chart-settings-row">
            <label>
              시작 (분)
              <input
                type="number"
                min={0}
                step={1}
                value={ganttTimeStart}
                disabled={!ganttTimeFixed}
                onChange={(e) => onGanttStartChange(Math.max(0, Number(e.target.value) || 0))}
              />
            </label>
            <label>
              종료 (분)
              <input
                type="number"
                min={1}
                step={1}
                value={ganttTimeEnd}
                disabled={!ganttTimeFixed}
                onChange={(e) => onGanttEndChange(Math.max(1, Number(e.target.value) || 1))}
              />
            </label>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              disabled={!ganttTimeFixed || dataTimeEnd <= 0}
              onClick={() => {
                onGanttStartChange(0);
                onGanttEndChange(dataTimeEnd);
              }}
            >
              시뮬 종료 시각 적용
            </button>
          </div>
        </section>

        {showReplay && onStepChange && (
          <section className="chart-settings-section">
            <h4>시뮬레이션 재생</h4>
            <p className="hint">스텝 슬라이더로 시점별 차트를 탐색합니다.</p>
            <label className="slider-label chart-replay-slider">
              스텝
              <span className={`step-badge${stepBump ? " bump" : ""}`}>
                {step} / {maxStep}
              </span>
              <input
                type="range"
                min={0}
                max={maxStep}
                value={step}
                onChange={(e) => onStepChange(Number(e.target.value))}
              />
            </label>
            <div className="chart-replay-quick">
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                disabled={step <= 0}
                onClick={() => onStepChange(0)}
              >
                처음
              </button>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                disabled={step >= maxStep}
                onClick={() => onStepChange(maxStep)}
              >
                마지막
              </button>
            </div>
          </section>
        )}
      </div>
    </details>
  );
}
