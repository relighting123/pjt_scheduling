import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import PlotChart from "../components/PlotChart";
import BatchInfoTable from "../components/BatchInfoTable";
import { api } from "../lib/api";
import {
  buildTrainExplainedVarChart,
  buildTrainLossChart,
  buildTrainRewardChart,
  hasTrainChartData,
} from "../lib/charts";
import type { AppConfig, DataSummary, TrainMetrics, TrainStatusResponse } from "../types";

interface TrainPageProps {
  config: AppConfig;
  summary: DataSummary | null;
  modelExists: boolean;
  onTrained: () => void;
  onRefresh: () => void;
}

const EMPTY_SERIES: TrainStatusResponse["series"] = {
  timesteps: [],
  ep_rew_mean: [],
  eval_timesteps: [],
  eval_reward: [],
  policy_loss: [],
  value_loss: [],
  explained_variance: [],
};

type DataRangeMode = "current" | "period" | "pick";
type TrainBudgetMode = "timesteps" | "episodes";

function periodLabel(folder: string): string {
  const parts = folder.split("/");
  return parts[parts.length - 1] ?? folder;
}
function formatLogTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString("ko-KR", { hour12: false });
  } catch {
    return iso;
  }
}

export default function TrainPage({
  config,
  summary,
  modelExists,
  onTrained,
  onRefresh,
}: TrainPageProps) {
  const [totalTs, setTotalTs] = useState(config.default_timesteps);
  const [nEpisodes, setNEpisodes] = useState(config.default_n_episodes ?? 100);
  const [budgetMode, setBudgetMode] = useState<TrainBudgetMode>("timesteps");
  const [lr, setLr] = useState(config.default_learning_rate);
  const [wSameOper, setWSameOper] = useState(config.default_w_same_oper);
  const [wIdle, setWIdle] = useState(config.default_w_idle_per_min);
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState<TrainMetrics | null>(null);
  const [status, setStatus] = useState<TrainStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rangeMode, setRangeMode] = useState<DataRangeMode>("current");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");
  const [pickedFolders, setPickedFolders] = useState<string[]>([]);
  const logRef = useRef<HTMLDivElement>(null);

  const trainFolders = useMemo(
    () =>
      (config.input_folders?.length ? config.input_folders : [config.input_folder]).filter(
        (f) => f.includes("/train/"),
      ),
    [config.input_folder, config.input_folders],
  );

  useEffect(() => {
    if (trainFolders.length === 0) return;
    setPickedFolders((prev) => {
      const filtered = prev.filter((f) => trainFolders.includes(f));
      if (filtered.length > 0) return filtered;
      return trainFolders.includes(config.input_folder)
        ? [config.input_folder]
        : [trainFolders[0]];
    });
    setFromDate((prev) => prev || periodLabel(trainFolders[0]));
    setToDate((prev) => prev || periodLabel(trainFolders[trainFolders.length - 1]));
  }, [trainFolders, config.input_folder]);

  const buildTrainBody = () => {
    const base = {
      total_timesteps: totalTs,
      learning_rate: lr,
      w_same_oper: wSameOper,
      w_idle_per_min: wIdle,
      train_budget_mode: budgetMode,
      ...(budgetMode === "episodes" ? { n_episodes: nEpisodes } : {}),
    };
    if (rangeMode === "period" && fromDate && toDate) {
      return { ...base, from_date: fromDate, to_date: toDate, fac_id: config.fac_id };
    }
    if (rangeMode === "pick" && pickedFolders.length > 0) {
      return { ...base, input_folders: pickedFolders };
    }
    return { ...base, input_folder: config.input_folder };
  };

  const rangeSummary =
    rangeMode === "current"
      ? config.input_folder
      : rangeMode === "period"
        ? `${fromDate || "?"} ~ ${toDate || "?"}`
        : `${pickedFolders.length}개 선택`;

  const toggleFolder = (folder: string) => {
    setPickedFolders((prev) =>
      prev.includes(folder) ? prev.filter((f) => f !== folder) : [...prev, folder],
    );
  };

  const applyStatus = useCallback((s: TrainStatusResponse) => {
    setStatus(s);
    if (s.status === "completed") {
      if (s.metrics) setMetrics(s.metrics);
      setLoading(false);
      onTrained();
    } else if (s.status === "failed") {
      setError(s.error ?? "학습 실패");
      setLoading(false);
    }
  }, [onTrained]);

  useEffect(() => {
    api.getTrainingStatus()
      .then((s) => {
        if (s.status === "running") {
          setLoading(true);
          setStatus(s);
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!loading) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const s = await api.getTrainingStatus();
        if (!cancelled) applyStatus(s);
      } catch {
        /* ignore transient poll errors */
      }
    };

    poll();
    const id = window.setInterval(poll, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [loading, applyStatus]);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [status?.logs]);

  const handleTrain = async () => {
    setLoading(true);
    setError(null);
    setMetrics(null);
    setStatus({
      status: "running",
      progress: 0,
      timesteps: 0,
      total_timesteps: totalTs,
      episodes: 0,
      total_episodes: nEpisodes,
      train_budget_mode: budgetMode,
      logs: [],
      series: EMPTY_SERIES,
      metrics: null,
      error: null,
    });
    try {
      await api.startTraining(buildTrainBody());
    } catch (e) {
      setError(e instanceof Error ? e.message : "학습 시작 실패");
      setLoading(false);
      setStatus(null);
    }
  };

  const series = status?.series ?? EMPTY_SERIES;
  const progressPct = Math.round((status?.progress ?? 0) * 100);
  const liveBudgetMode = status?.train_budget_mode ?? budgetMode;
  const progressLabel =
    liveBudgetMode === "episodes"
      ? `${(status?.episodes ?? 0).toLocaleString()} / ${(status?.total_episodes ?? nEpisodes).toLocaleString()} episodes · ${progressPct}%`
      : `${(status?.timesteps ?? 0).toLocaleString()} / ${(status?.total_timesteps ?? totalTs).toLocaleString()} steps · ${progressPct}%`;
  const showLive = loading || (status && status.status !== "idle");
  const showCharts = hasTrainChartData(series);

  return (
    <div className="page">
      <h2>모델 학습</h2>

      <div className="card-stagger">
      <div className="grid-2">
        <section className="card">
          <h3>학습 파라미터</h3>
          <fieldset className="mode-group train-range-group">
            <legend>학습량 기준</legend>
            <div className="mode-pills">
              <label className={`mode-pill${budgetMode === "timesteps" ? " active" : ""}`}>
                <input
                  type="radio"
                  name="train-budget"
                  checked={budgetMode === "timesteps"}
                  onChange={() => setBudgetMode("timesteps")}
                  disabled={loading}
                />
                Timesteps
              </label>
              <label className={`mode-pill${budgetMode === "episodes" ? " active" : ""}`}>
                <input
                  type="radio"
                  name="train-budget"
                  checked={budgetMode === "episodes"}
                  onChange={() => setBudgetMode("episodes")}
                  disabled={loading}
                />
                Episodes
              </label>
            </div>
          </fieldset>
          {budgetMode === "timesteps" ? (
            <label>
              Total Timesteps
              <input
                type="number"
                min={1000}
                step={10000}
                value={totalTs}
                onChange={(e) => setTotalTs(Number(e.target.value))}
                disabled={loading}
              />
            </label>
          ) : (
            <label>
              목표 에피소드 수
              <input
                type="number"
                min={1}
                step={10}
                value={nEpisodes}
                onChange={(e) => setNEpisodes(Number(e.target.value))}
                disabled={loading}
              />
            </label>
          )}
          <p className="hint">
            {budgetMode === "timesteps"
              ? "PPO 환경 step 누적 횟수 기준"
              : "스케줄링 1판(reset~done) 완료 1회 = 에피소드 1회"}
          </p>
          <label>
            Learning Rate
            <input
              type="number"
              step={0.0001}
              value={lr}
              onChange={(e) => setLr(Number(e.target.value))}
              disabled={loading}
            />
          </label>
        </section>

        <section className="card">
          <h3>보상 가중치</h3>
          <label>
            동일 OPER 보너스: {wSameOper.toFixed(1)}
            <input
              type="range"
              min={0}
              max={5}
              step={0.5}
              value={wSameOper}
              onChange={(e) => setWSameOper(Number(e.target.value))}
              disabled={loading}
            />
          </label>
          <label>
            Idle 패널티(분당): {wIdle.toFixed(1)}
            <input
              type="range"
              min={-3}
              max={0}
              step={0.1}
              value={wIdle}
              onChange={(e) => setWIdle(Number(e.target.value))}
              disabled={loading}
            />
          </label>
        </section>
      </div>

      <section className="card">
        <h3>학습 데이터 범위</h3>
        <fieldset className="mode-group train-range-group">
          <legend>기간 선택</legend>
          <div className="mode-pills">
            <label className={`mode-pill${rangeMode === "current" ? " active" : ""}`}>
              <input
                type="radio"
                name="train-range"
                checked={rangeMode === "current"}
                onChange={() => setRangeMode("current")}
                disabled={loading}
              />
              현재 선택
            </label>
            <label className={`mode-pill${rangeMode === "period" ? " active" : ""}`}>
              <input
                type="radio"
                name="train-range"
                checked={rangeMode === "period"}
                onChange={() => setRangeMode("period")}
                disabled={loading || trainFolders.length === 0}
              />
              RULE_TIMEKEY 구간
            </label>
            <label className={`mode-pill${rangeMode === "pick" ? " active" : ""}`}>
              <input
                type="radio"
                name="train-range"
                checked={rangeMode === "pick"}
                onChange={() => setRangeMode("pick")}
                disabled={loading || trainFolders.length === 0}
              />
              직접 선택
            </label>
          </div>
        </fieldset>

        {rangeMode === "current" && (
          <p className="hint">
            사이드바 데이터셋 1개: <code>{config.input_folder}</code>
          </p>
        )}

        {rangeMode === "period" && (
          <div className="grid-2">
            <label>
              시작 RULE_TIMEKEY
              <input
                type="text"
                placeholder="YYYYMMDDHHmmss"
                value={fromDate}
                onChange={(e) => setFromDate(e.target.value)}
                disabled={loading}
              />
            </label>
            <label>
              종료 RULE_TIMEKEY
              <input
                type="text"
                placeholder="YYYYMMDDHHmmss"
                value={toDate}
                onChange={(e) => setToDate(e.target.value)}
                disabled={loading}
              />
            </label>
          </div>
        )}

        {rangeMode === "pick" && (
          <div className="train-folder-pick">
            {trainFolders.length === 0 ? (
              <p className="hint">train 기간 데이터가 없습니다. dataset 폴더에 train JSON을 준비하세요.</p>
            ) : (
              trainFolders.map((folder) => (
                <label key={folder} className="train-folder-option">
                  <input
                    type="checkbox"
                    checked={pickedFolders.includes(folder)}
                    onChange={() => toggleFolder(folder)}
                    disabled={loading}
                  />
                  <code>{folder}</code>
                </label>
              ))
            )}
          </div>
        )}

        <p className="hint">
          학습 대상: <strong>{rangeSummary}</strong>
          {rangeMode !== "current" && " · 복수 기간은 VecEnv로 병렬 수집합니다."}
        </p>
      </section>

      <div className="grid-2">
        <section className="card">
          <h3>입력 데이터 요약</h3>
          {summary ? (
            <div className="metrics-row">
              <Metric label="EQP 수" value={String(summary.eqp_count)} />
              <Metric label="LOT 수" value={String(summary.lot_count)} />
              <Metric label="제품 종류" value={String(summary.prod_count)} />
              <Metric label="공정 종류" value={String(summary.oper_count)} />
              <Metric label="Batch 레시피" value={String(summary.batch_info_count ?? 0)} />
              <Metric label="시뮬 종료(분)" value={String(summary.sim_end_minutes)} />
            </div>
          ) : (
            <p className="hint">데이터를 불러올 수 없습니다. dataset 폴더에 JSON을 준비하세요.</p>
          )}
          <button type="button" className="btn btn-secondary" onClick={onRefresh} disabled={loading}>
            데이터 새로고침
          </button>
          {summary && (summary.batch_info?.length ?? 0) > 0 && (
            <div className="batch-info-train">
              <BatchInfoTable rows={summary.batch_info} compact />
            </div>
          )}
        </section>

        <section className="card">
          <h3>모델 상태</h3>
          <p className={modelExists ? "status-ok" : "status-warn"}>
            {modelExists ? "저장된 모델이 있습니다." : "저장된 모델이 없습니다."}
          </p>
        </section>
      </div>

      <button
        type="button"
        className={`btn btn-primary${loading ? " is-loading" : ""}`}
        onClick={handleTrain}
        disabled={loading || !summary || (rangeMode === "pick" && pickedFolders.length === 0)}
      >
        {loading ? "학습 진행 중..." : "학습 시작"}
      </button>

      {error && <p className="error">{error}</p>}

      {showLive && (
        <section className="card train-live">
          <h3>학습 진행</h3>
          <div className="test-progress">
            <div className="test-progress-bar" style={{ width: `${progressPct}%` }} />
            <span className="test-progress-label">{progressLabel}</span>
          </div>

          <div className="train-log-wrap">
            <div className="train-log" ref={logRef}>
              {(status?.logs ?? []).length === 0 ? (
                <p className="hint train-log-empty">학습 로그를 기다리는 중…</p>
              ) : (
                (status?.logs ?? []).map((entry, i) => (
                  <div key={`${entry.time}-${i}`} className={`train-log-line train-log-${entry.level}`}>
                    <span className="train-log-time">{formatLogTime(entry.time)}</span>
                    <span className="train-log-msg">{entry.message}</span>
                  </div>
                ))
              )}
            </div>
          </div>

          {showCharts && (
            <div className="train-charts grid-2">
              <PlotChart {...buildTrainRewardChart(series)} />
              <PlotChart {...buildTrainLossChart(series)} />
              <div className="train-chart-wide">
                <PlotChart {...buildTrainExplainedVarChart(series)} />
              </div>
            </div>
          )}
        </section>
      )}

      {metrics && (
        <section className="card">
          <h3>학습 결과 (3 에피소드 평균)</h3>
          <div className="metrics-row">
            <Metric label="평균 보상" value={metrics.mean_reward.toFixed(1)} />
            <Metric label="공정 전환(평균)" value={metrics.mean_oper_sw.toFixed(1)} />
            <Metric label="제품 전환(평균)" value={metrics.mean_prod_sw.toFixed(1)} />
            <Metric label="Idle 합계(평균)" value={`${metrics.mean_idle.toFixed(0)}분`} />
          </div>
        </section>
      )}
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}
