import { useEffect, useMemo, useState } from "react";
import PlotChart from "../components/PlotChart";
import { api } from "../lib/api";
import {
  buildTrainRewardChart,
  buildAlgorithmKpiComparison,
  hasTrainChartData,
  type AlgoCompareEntry,
} from "../lib/charts";
import { buildEqpModelMap, computeInferenceKpi } from "../lib/metrics";
import type { AppMode, InferenceResult, TestBenchmarkResponse, TrainStatusResponse } from "../types";

const EMPTY_SERIES: TrainStatusResponse["series"] = {
  timesteps: [], ep_rew_mean: [], eval_timesteps: [], eval_reward: [],
  policy_loss: [], value_loss: [], explained_variance: [],
};

interface Props { onNavigate: (m: AppMode) => void; }

export default function DashboardPage({ onNavigate }: Props) {
  const [infer, setInfer] = useState<InferenceResult | null>(null);
  const [inferErr, setInferErr] = useState<string | null>(null);
  const [inferLoading, setInferLoading] = useState(true);

  const [testData, setTestData] = useState<TestBenchmarkResponse | null>(null);
  const [testLoading, setTestLoading] = useState(true);

  const [trainStatus, setTrainStatus] = useState<TrainStatusResponse | null>(null);
  const [trainLoading, setTrainLoading] = useState(true);

  useEffect(() => {
    api.getInferenceResult()
      .then(setInfer)
      .catch((e: unknown) => setInferErr(e instanceof Error ? e.message : "결과 없음"))
      .finally(() => setInferLoading(false));

    api.getSavedTestBenchmark().then(setTestData).catch(() => {}).finally(() => setTestLoading(false));
    api.getTrainingStatus().then(setTrainStatus).catch(() => {}).finally(() => setTrainLoading(false));
  }, []);

  const eqpModelMap = useMemo(() => buildEqpModelMap(infer?.event_log ?? []), [infer]);
  const kpi = useMemo(() => (infer ? computeInferenceKpi(infer, eqpModelMap) : null), [infer, eqpModelMap]);

  const testEntries = useMemo((): AlgoCompareEntry[] => {
    if (!testData?.datasets?.length) return [];
    const map: Record<string, InferenceResult> = {};
    testData.datasets.forEach((d) => d.results.forEach((r) => { map[r.algorithm ?? "rl"] = r; }));
    const labels: Record<string, string> = { rl: "PPO", minprogress: "MinProgress", earliest_st: "Earliest-ST" };
    return Object.entries(map).map(([algo, result]) => ({ algorithm: algo, label: labels[algo] ?? algo, result }));
  }, [testData]);

  const testChart = useMemo(
    () => (testEntries.length ? buildAlgorithmKpiComparison(testEntries) : null),
    [testEntries],
  );

  const trainSeries = trainStatus?.series ?? EMPTY_SERIES;
  const trainChart = useMemo(
    () => (hasTrainChartData(trainSeries) ? buildTrainRewardChart(trainSeries) : null),
    [trainSeries],
  );

  const now = new Date().toLocaleString("ko-KR", { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });

  return (
    <div className="dash-page">
      <div className="dash-header">
        <div>
          <h1 className="dash-title">AI Scheduling <span>Dashboard</span></h1>
          <p className="dash-subtitle">반도체 설비 AI 스케줄링 인텔리전스</p>
        </div>
        <div className="dash-time">{now}</div>
      </div>

      <div className="dash-grid">
        {/* ── Inference Card ── */}
        <div className="dash-card">
          <div className="dash-card-top inference" />
          <div className="dash-card-header">
            <div className="dash-card-icon inference">▶</div>
            <div className="dash-card-heading">
              <div className="dash-card-name">추론 결과</div>
              {infer?.algorithm && (
                <div className="dash-card-meta">
                  알고리즘 <span className="mono">{infer.algorithm}</span>
                </div>
              )}
            </div>
            {infer && <span className="badge badge-accent" style={{ marginLeft: "auto" }}>최신</span>}
          </div>
          <div className="dash-card-body">
            {inferLoading && (
              <div className="dash-loading"><div className="dash-spinner" /> 불러오는 중</div>
            )}
            {!inferLoading && inferErr && (
              <div className="dash-empty">
                <span className="dash-empty-icon">◌</span>
                <p>추론 결과 없음</p>
                <button type="button" className="btn btn-accent btn-sm" onClick={() => onNavigate("inference")}>추론 실행 →</button>
              </div>
            )}
            {!inferLoading && kpi && (
              <div className="dash-kpis dash-kpis-4">
                <div className="dash-kpi">
                  <div className="dash-kpi-label">Makespan</div>
                  <div className="dash-kpi-value">{kpi.makespan.toLocaleString()}<small className="dash-kpi-unit">분</small></div>
                </div>
                <div className="dash-kpi">
                  <div className="dash-kpi-label">평균 가동률</div>
                  <div className={`dash-kpi-value ${kpi.avgUtilPct >= 80 ? "good" : "warn"}`}>{kpi.avgUtilPct.toLocaleString()}<small className="dash-kpi-unit">%</small></div>
                </div>
                <div className="dash-kpi">
                  <div className="dash-kpi-label">계획 달성률</div>
                  <div className={`dash-kpi-value ${kpi.avgAchPct >= 90 ? "good" : kpi.avgAchPct >= 70 ? "warn" : "bad"}`}>{kpi.avgAchPct.toLocaleString()}<small className="dash-kpi-unit">%</small></div>
                </div>
                <div className="dash-kpi">
                  <div className="dash-kpi-label">타겟 달성률</div>
                  <div className={`dash-kpi-value ${kpi.avgTargetAchPct >= 90 ? "good" : kpi.avgTargetAchPct >= 70 ? "warn" : "bad"}`}>{kpi.avgTargetAchPct.toLocaleString()}<small className="dash-kpi-unit">%</small></div>
                </div>
              </div>
            )}
          </div>
          <div className="dash-card-footer">
            <span className="dash-meta">{infer ? `${infer.schedule.length}건` : "—"}</span>
            <button type="button" className="btn btn-accent btn-sm" onClick={() => onNavigate("inference")}>자세히 →</button>
          </div>
        </div>

        {/* ── Test Card ── */}
        <div className="dash-card">
          <div className="dash-card-top test" />
          <div className="dash-card-header">
            <div className="dash-card-icon test">⊞</div>
            <div>
              <div className="dash-card-name">테스트 셋 결과</div>
            </div>
            {testData?.datasets?.length ? (
              <span className="badge badge-info" style={{ marginLeft: "auto" }}>{testData.datasets.length}셋</span>
            ) : null}
          </div>
          <div className="dash-card-body">
            {testLoading && (
              <div className="dash-loading"><div className="dash-spinner" /> 불러오는 중</div>
            )}
            {!testLoading && !testData?.datasets?.length && (
              <div className="dash-empty">
                <span className="dash-empty-icon">◌</span>
                <p>테스트 결과 없음</p>
                <button type="button" className="btn btn-accent btn-sm" onClick={() => onNavigate("test")}>테스트 실행 →</button>
              </div>
            )}
            {!testLoading && testChart && (
              <>
                <PlotChart {...testChart} />
                <div className="dash-kpis">
                  {testEntries.slice(0, 4).map((e) => {
                    const s = e.result.schedule;
                    const ms = s.length ? Math.max(...s.map((r) => r.END_TM)) : 0;
                    return (
                      <div key={e.algorithm} className="dash-kpi">
                        <div className="dash-kpi-label">{e.label}</div>
                        <div className="dash-kpi-value">{ms.toLocaleString()}<small className="dash-kpi-unit">분</small></div>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>
          <div className="dash-card-footer">
            <span className="dash-meta">{testEntries.length ? `${testEntries.length}개 알고리즘` : "—"}</span>
            <button type="button" className="btn btn-accent btn-sm" onClick={() => onNavigate("test")}>자세히 →</button>
          </div>
        </div>

        {/* ── Training Card ── */}
        <div className="dash-card">
          <div className="dash-card-top train" />
          <div className="dash-card-header">
            <div className="dash-card-icon train">↑</div>
            <div>
              <div className="dash-card-name">학습 결과</div>
            </div>
            {trainStatus?.status && (
              <span className={`badge ${trainStatus.status === "completed" ? "badge-ok" : trainStatus.status === "running" ? "badge-warn" : "badge-accent"}`} style={{ marginLeft: "auto" }}>
                {trainStatus.status === "completed" ? "완료" : trainStatus.status === "running" ? "진행중" : trainStatus.status}
              </span>
            )}
          </div>
          <div className="dash-card-body">
            {trainLoading && (
              <div className="dash-loading"><div className="dash-spinner" /> 불러오는 중</div>
            )}
            {!trainLoading && !trainChart && (
              <div className="dash-empty">
                <span className="dash-empty-icon">◌</span>
                <p>학습 이력 없음</p>
                <button type="button" className="btn btn-accent btn-sm" onClick={() => onNavigate("train")}>학습 시작 →</button>
              </div>
            )}
            {!trainLoading && trainChart && (
              <>
                <PlotChart {...trainChart} />
                {trainStatus?.metrics && (
                  <div className="dash-kpis">
                    <div className="dash-kpi">
                      <div className="dash-kpi-label">평균 보상</div>
                      <div className="dash-kpi-value">{trainStatus.metrics.mean_reward.toFixed(2)}</div>
                    </div>
                    <div className="dash-kpi">
                      <div className="dash-kpi-label">완료율</div>
                      <div className="dash-kpi-value">{(trainStatus.metrics.mean_completion * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
          <div className="dash-card-footer">
            <span className="dash-meta">
              {trainStatus?.timesteps ? `${trainStatus.timesteps.toLocaleString()} steps` : "—"}
            </span>
            <button type="button" className="btn btn-accent btn-sm" onClick={() => onNavigate("train")}>자세히 →</button>
          </div>
        </div>
      </div>
    </div>
  );
}
