import type { GeneratorConfig, SampleScenario } from "../types";

export interface CreateSampleOpts {
  fac_id: string;
  scenario: string;
  bootstrap?: boolean;
  split?: string;
  use_period_count?: boolean;
  generator_config?: GeneratorConfig;
}

interface DatasetEmptyPanelProps {
  facId: string;
  scenario: string;
  scenarios: SampleScenario[];
  sampleLoading: boolean;
  loadError: string | null;
  onCreateSample: (opts: CreateSampleOpts) => void;
  sampleOpts: () => CreateSampleOpts;
  onGoToDataset: () => void;
}

export default function DatasetEmptyPanel({
  facId,
  scenario,
  scenarios,
  sampleLoading,
  loadError,
  onCreateSample,
  sampleOpts,
  onGoToDataset,
}: DatasetEmptyPanelProps) {
  const scenarioName =
    scenarios.find((s) => s.id === scenario)?.name ?? scenario;

  return (
    <section className="empty-dataset card card-stagger">
      <h2>데이터셋이 없습니다</h2>
      <p className="empty-dataset-lead">
        dataset 폴더가 비어 있거나 JSON이 없습니다.
        데이터셋 페이지에서 샘플을 생성하세요.
      </p>
      {loadError && <p className="empty-dataset-error">{loadError}</p>}
      <p className="empty-dataset-meta">
        FAC_ID: <strong>{facId}</strong> · 시나리오: <strong>{scenarioName}</strong>
      </p>
      <div className="empty-dataset-actions">
        <button type="button" className="btn btn-primary" onClick={onGoToDataset}>
          데이터셋 생성 페이지로 이동
        </button>
        <button
          type="button"
          className={`btn btn-secondary${sampleLoading ? " is-loading" : ""}`}
          disabled={sampleLoading}
          onClick={() => onCreateSample({ ...sampleOpts(), bootstrap: true })}
        >
          {sampleLoading ? "생성 중..." : "여기서 바로 bootstrap"}
        </button>
      </div>
    </section>
  );
}
