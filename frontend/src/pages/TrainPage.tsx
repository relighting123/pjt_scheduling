import { useState } from "react";
import { api } from "../lib/api";
import type { AppConfig, DataSummary, TrainMetrics } from "../types";

interface TrainPageProps {
  config: AppConfig;
  summary: DataSummary | null;
  modelExists: boolean;
  onTrained: () => void;
  onRefresh: () => void;
}

export default function TrainPage({
  config,
  summary,
  modelExists,
  onTrained,
  onRefresh,
}: TrainPageProps) {
  const [totalTs, setTotalTs] = useState(config.default_timesteps);
  const [lr, setLr] = useState(config.default_learning_rate);
  const [wSameOper, setWSameOper] = useState(config.default_w_same_oper);
  const [wIdle, setWIdle] = useState(config.default_w_idle_per_min);
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState<TrainMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleTrain = async () => {
    setLoading(true);
    setError(null);
    setMetrics(null);
    try {
      const res = await api.train({
        total_timesteps: totalTs,
        learning_rate: lr,
        w_same_oper: wSameOper,
        w_idle_per_min: wIdle,
      });
      setMetrics(res.metrics);
      onTrained();
    } catch (e) {
      setError(e instanceof Error ? e.message : "학습 실패");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <h2>모델 학습</h2>

      <div className="card-stagger">
      <div className="grid-2">
        <section className="card">
          <h3>학습 파라미터</h3>
          <label>
            Total Timesteps
            <input
              type="number"
              min={1000}
              step={10000}
              value={totalTs}
              onChange={(e) => setTotalTs(Number(e.target.value))}
            />
          </label>
          <label>
            Learning Rate
            <input
              type="number"
              step={0.0001}
              value={lr}
              onChange={(e) => setLr(Number(e.target.value))}
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
            />
          </label>
        </section>
      </div>

      <div className="grid-2">
        <section className="card">
          <h3>입력 데이터 요약</h3>
          {summary ? (
            <div className="metrics-row">
              <Metric label="EQP 수" value={String(summary.eqp_count)} />
              <Metric label="LOT 수" value={String(summary.lot_count)} />
              <Metric label="제품 종류" value={String(summary.prod_count)} />
              <Metric label="공정 종류" value={String(summary.oper_count)} />
              <Metric label="시뮬 종료(분)" value={String(summary.sim_end_minutes)} />
            </div>
          ) : (
            <p className="hint">데이터를 불러올 수 없습니다. 샘플 데이터를 생성하세요.</p>
          )}
          <button type="button" className="btn btn-secondary" onClick={onRefresh}>
            데이터 새로고침
          </button>
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
        disabled={loading || !summary}
      >
        {loading ? "학습 진행 중..." : "학습 시작"}
      </button>

      {error && <p className="error">{error}</p>}

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
