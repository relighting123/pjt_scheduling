import type { GeneratorConfig, SampleScenario } from "../types";
import type { CreateSampleOpts } from "./DatasetEmptyPanel";

interface DatasetGeneratorFormProps {
  facId: string;
  onFacIdChange: (value: string) => void;
  scenario: string;
  onScenarioChange: (value: string) => void;
  scenarios: SampleScenario[];
  genConfig: GeneratorConfig | null;
  onGenConfigChange: (patch: Partial<GeneratorConfig>) => void;
  sampleLoading: boolean;
  onCreateSample: (opts: CreateSampleOpts) => void;
}

function numField(
  label: string,
  value: number,
  onChange: (v: number) => void,
  opts?: { min?: number; max?: number; step?: number },
) {
  return (
    <label className="gen-field page-gen-field">
      <span>{label}</span>
      <input
        type="number"
        value={value}
        min={opts?.min}
        max={opts?.max}
        step={opts?.step ?? 1}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

export default function DatasetGeneratorForm({
  facId,
  onFacIdChange,
  scenario,
  onScenarioChange,
  scenarios,
  genConfig,
  onGenConfigChange,
  sampleLoading,
  onCreateSample,
}: DatasetGeneratorFormProps) {
  const selectedScenario = scenarios.find((s) => s.id === scenario);
  const showGenPanel = scenario === "random" && genConfig;

  const sampleOpts = (): CreateSampleOpts => ({
    fac_id: facId,
    scenario,
    generator_config: genConfig ?? undefined,
  });

  const scenarioOptions =
    scenarios.length > 0
      ? scenarios
      : [
          { id: "random", name: "랜덤 (파라미터)", description: "" },
          { id: "default", name: "기본 (3제품)", description: "" },
        ];

  return (
    <div className="dataset-generator">
      <label className="field-label page-field-label" htmlFor="dataset-fac-id">
        FAC_ID
      </label>
      <input
        id="dataset-fac-id"
        type="text"
        className="input-text page-input"
        value={facId}
        onChange={(e) => onFacIdChange(e.target.value)}
        disabled={sampleLoading}
      />

      <label className="field-label page-field-label" htmlFor="dataset-scenario">
        샘플 시나리오
      </label>
      <select
        id="dataset-scenario"
        className="input-select page-input"
        value={scenario}
        onChange={(e) => onScenarioChange(e.target.value)}
        disabled={sampleLoading}
      >
        {scenarioOptions.map((s) => (
          <option key={s.id} value={s.id}>
            {s.name}
          </option>
        ))}
      </select>
      {selectedScenario?.description && (
        <p className="hint">{selectedScenario.description}</p>
      )}

      {showGenPanel && genConfig && (
        <details className="gen-panel page-gen-panel" open>
          <summary>랜덤 생성 파라미터</summary>

          <p className="gen-section-title">규모</p>
          <div className="gen-grid">
            {numField("제품 수", genConfig.n_products, (v) => onGenConfigChange({ n_products: v }), { min: 1, max: 20 })}
            {numField("설비 수", genConfig.n_eqps, (v) => onGenConfigChange({ n_eqps: v }), { min: 1, max: 20 })}
            {numField("공정 수", genConfig.n_opers, (v) => onGenConfigChange({ n_opers: v }), { min: 1, max: 10 })}
            {numField("공정당 LOT", genConfig.lots_per_oper, (v) => onGenConfigChange({ lots_per_oper: v }), { min: 1, max: 30 })}
            {numField("WF_QTY", genConfig.wf_qty, (v) => onGenConfigChange({ wf_qty: v }), { min: 1, max: 500 })}
          </div>

          <p className="gen-section-title">ST (분) — min / max / std</p>
          <div className="gen-grid">
            {numField("ST min", genConfig.st_min, (v) => onGenConfigChange({ st_min: v }), { min: 1 })}
            {numField("ST max", genConfig.st_max, (v) => onGenConfigChange({ st_max: v }), { min: 1 })}
            {numField("ST std", genConfig.st_std, (v) => onGenConfigChange({ st_std: v }), { min: 0, step: 0.1 })}
          </div>

          <p className="gen-section-title">Eligibility (0=전용, 1=전체)</p>
          <label className="gen-slider page-gen-slider">
            <span>{genConfig.eligibility.toFixed(2)}</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={genConfig.eligibility}
              onChange={(e) => onGenConfigChange({ eligibility: Number(e.target.value) })}
            />
          </label>

          <p className="gen-section-title">계획량 (D0/D1 랜덤)</p>
          <div className="gen-grid">
            {numField("min", genConfig.plan_qty_min, (v) => onGenConfigChange({ plan_qty_min: v }), { min: 0 })}
            {numField("max", genConfig.plan_qty_max, (v) => onGenConfigChange({ plan_qty_max: v }), { min: 1 })}
            {numField("priority", genConfig.plan_priority, (v) => onGenConfigChange({ plan_priority: v }), { min: 1, max: 9 })}
          </div>

          <p className="gen-section-title">Wafer Split (PPK×MODEL, 장)</p>
          <div className="gen-grid">
            {numField("SPLIT_QTY", genConfig.split_qty, (v) => onGenConfigChange({ split_qty: v }), { min: 1, max: 100 })}
          </div>

          <p className="gen-section-title">데이터셋 크기 (RULE_TIMEKEY 폴더 수)</p>
          <div className="gen-grid">
            {numField("train", genConfig.train_period_count, (v) => onGenConfigChange({ train_period_count: v }), { min: 1, max: 365 })}
            {numField("test", genConfig.test_period_count, (v) => onGenConfigChange({ test_period_count: v }), { min: 1, max: 365 })}
          </div>

          <label className="gen-field page-gen-field">
            <span>seed (선택)</span>
            <input
              type="number"
              placeholder="비우면 매번 다름"
              value={genConfig.seed ?? ""}
              onChange={(e) => {
                const raw = e.target.value.trim();
                onGenConfigChange({
                  seed: raw === "" ? null : Number(raw),
                });
              }}
            />
          </label>
        </details>
      )}

      <div className="dataset-actions">
        <button
          type="button"
          className={`btn btn-primary${sampleLoading ? " is-loading" : ""}`}
          onClick={() => onCreateSample({ ...sampleOpts(), bootstrap: true })}
          disabled={sampleLoading}
        >
          {sampleLoading ? "생성 중..." : "FAC bootstrap (train/test/infer)"}
        </button>
        <button
          type="button"
          className={`btn btn-secondary${sampleLoading ? " is-loading" : ""}`}
          onClick={() => onCreateSample({ ...sampleOpts(), split: "train", use_period_count: true })}
          disabled={sampleLoading || scenario !== "random"}
        >
          train 기간 일괄 생성
        </button>
        <button
          type="button"
          className={`btn btn-secondary${sampleLoading ? " is-loading" : ""}`}
          onClick={() => onCreateSample({ ...sampleOpts(), split: "test", use_period_count: true })}
          disabled={sampleLoading || scenario !== "random"}
        >
          test 기간 일괄 생성
        </button>
        <button
          type="button"
          className={`btn btn-secondary${sampleLoading ? " is-loading" : ""}`}
          onClick={() => onCreateSample({ ...sampleOpts(), split: "train" })}
          disabled={sampleLoading}
        >
          단일 train 스냅샷 생성
        </button>
      </div>
    </div>
  );
}
