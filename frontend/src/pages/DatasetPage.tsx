import DatasetGeneratorForm from "../components/DatasetGeneratorForm";
import type { CreateSampleOpts } from "../components/DatasetEmptyPanel";
import type { AppConfig, DataSummary, GeneratorConfig, SampleScenario } from "../types";

interface DatasetPageProps {
  config: AppConfig | null;
  summary: DataSummary | null;
  facId: string;
  onFacIdChange: (value: string) => void;
  scenario: string;
  onScenarioChange: (value: string) => void;
  scenarios: SampleScenario[];
  genConfig: GeneratorConfig | null;
  onGenConfigChange: (patch: Partial<GeneratorConfig>) => void;
  sampleLoading: boolean;
  loadError: string | null;
  onCreateSample: (opts: CreateSampleOpts) => void;
}

export default function DatasetPage({
  config,
  summary,
  facId,
  onFacIdChange,
  scenario,
  onScenarioChange,
  scenarios,
  genConfig,
  onGenConfigChange,
  sampleLoading,
  loadError,
  onCreateSample,
}: DatasetPageProps) {
  return (
    <div className="page">
      <h2>데이터셋 생성</h2>
      <p className="hint page-lead">
        샘플 JSON을 생성합니다. 생성 후 사이드바에서 dataset 경로를 선택해 학습·추론에 사용하세요.
      </p>

      {loadError && <div className="banner banner-warn">{loadError}</div>}

      <div className="card-stagger">
        {config && (
          <section className="card">
            <h3>현재 선택</h3>
            <p className="hint dataset-path-hint">
              경로: <code>{config.input_folder}</code>
              {config.input_dir && <> · 실제 폴더: <code>{config.input_dir}</code></>}
            </p>
            {summary ? (
              <p className="hint">
                EQP {summary.eqp_count} · LOT {summary.lot_count} · 제품 {summary.prod_count} · 공정 {summary.oper_count}
              </p>
            ) : (
              <p className="hint">선택한 경로에 데이터가 없습니다. 아래에서 생성하세요.</p>
            )}
          </section>
        )}

        <section className="card">
          <h3>생성 설정</h3>
          <DatasetGeneratorForm
            facId={facId}
            onFacIdChange={onFacIdChange}
            scenario={scenario}
            onScenarioChange={onScenarioChange}
            scenarios={scenarios}
            genConfig={genConfig}
            onGenConfigChange={onGenConfigChange}
            sampleLoading={sampleLoading}
            onCreateSample={onCreateSample}
          />
        </section>
      </div>
    </div>
  );
}
