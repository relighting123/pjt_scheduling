import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import PlotChart from "../components/PlotChart";
import ExpandableErrorBanner from "../components/ExpandableErrorBanner";
import { api } from "../lib/api";
import { buildTrainExplainedVarChart, buildTrainLossChart, buildTrainRewardChart, hasTrainChartData } from "../lib/charts";
import type { AppConfig, DataSummary, RewardConfig, TrainMetrics, TrainStatusResponse } from "../types";

interface Props {
  config: AppConfig; summary: DataSummary | null;
  modelExists: boolean; onTrained: () => void; onRefresh: () => void;
}

const EMPTY: TrainStatusResponse["series"] = { timesteps:[], ep_rew_mean:[], eval_timesteps:[], eval_reward:[], policy_loss:[], value_loss:[], explained_variance:[] };

type Budget = "timesteps"|"episodes";
type Range  = "current"|"period"|"pick";
type TrainTab = "charts"|"log"|"metrics";

const REWARD_SLIDERS: {
  key: keyof RewardConfig;
  label: string;
  min: number;
  max: number;
  step: number;
}[] = [
  { key: "w_plan_hit", label: "계획 달성", min: 0, max: 8, step: 0.5 },
  { key: "w_pacing", label: "Pacing 추종", min: 0, max: 8, step: 0.5 },
  { key: "w_flow_balance", label: "Flow balance", min: 0, max: 5, step: 0.5 },
  { key: "w_same_oper", label: "동일 OPER", min: 0, max: 5, step: 0.5 },
  { key: "w_same_prod", label: "동일 PPK", min: 0, max: 5, step: 0.5 },
  { key: "w_prod_switch", label: "PPK 전환", min: 0, max: 5, step: 0.5 },
  { key: "w_completion", label: "완료 보너스", min: 0, max: 5, step: 0.5 },
  { key: "w_idle_per_min", label: "Idle/분", min: -3, max: 3, step: 0.1 },
  { key: "w_conversion", label: "전환 패널티", min: -15, max: 0, step: 0.5 },
  { key: "w_late_finish", label: "Late finish", min: -5, max: 0, step: 0.1 },
  { key: "reward_clip", label: "Reward clip", min: 1, max: 10, step: 0.5 },
];

const REWARD_TOGGLES: { key: keyof RewardConfig; label: string }[] = [
  { key: "use_achievable_target", label: "Achievable target" },
  { key: "same_oper_conditional", label: "same_oper 조건부" },
];

function fmtTime(iso: string) {
  try { return new Date(iso).toLocaleTimeString("ko-KR",{hour12:false}); } catch { return iso; }
}
function periodLabel(f: string) { return f.split("/").pop() ?? f; }

export default function TrainPage({ config, summary, modelExists, onTrained, onRefresh }: Props) {
  const [totalTs, setTotalTs] = useState(config.default_timesteps);
  const [nEps, setNEps]       = useState(config.default_n_episodes ?? 100);
  const [budget, setBudget]   = useState<Budget>("timesteps");
  const [lr, setLr]           = useState(config.default_learning_rate);
  const [reward, setReward]   = useState<RewardConfig>(config.default_reward);
  const [loading, setLoading] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [metrics, setMetrics] = useState<TrainMetrics | null>(null);
  const [status, setStatus]   = useState<TrainStatusResponse | null>(null);
  const [error, setError]     = useState<string | null>(null);
  const [range, setRange]     = useState<Range>("current");
  const [fromDate, setFrom]   = useState("");
  const [toDate, setTo]       = useState("");
  const [picked, setPicked]   = useState<string[]>([]);
  const [trainTab, setTrainTab] = useState<TrainTab>("charts");
  const logRef = useRef<HTMLDivElement>(null);

  const trainFolders = useMemo(
    () => (config.input_folders?.length ? config.input_folders : [config.input_folder]).filter(f => f.includes("/train/")),
    [config],
  );

  useEffect(() => {
    if (!trainFolders.length) return;
    setPicked(prev => { const f = prev.filter(x=>trainFolders.includes(x)); return f.length?f:[trainFolders[0]]; });
    setFrom(prev => prev || periodLabel(trainFolders[0]));
    setTo(prev => prev || periodLabel(trainFolders[trainFolders.length-1]));
  }, [trainFolders, config.input_folder]);

  const body = () => {
    const base = {
      total_timesteps: totalTs,
      learning_rate: lr,
      ...reward,
      train_budget_mode: budget,
      ...(budget === "episodes" ? { n_episodes: nEps } : {}),
    };
    if (range==="period" && fromDate && toDate) return { ...base, from_date:fromDate, to_date:toDate, fac_id:config.fac_id };
    if (range==="pick" && picked.length) return { ...base, input_folders:picked };
    return { ...base, input_folder:config.input_folder };
  };

  const applyStatus = useCallback((s: TrainStatusResponse) => {
    setStatus(s);
    if (s.status==="completed") { if(s.metrics) setMetrics(s.metrics); setLoading(false); setStopping(false); onTrained(); }
    else if (s.status==="stopped") { setLoading(false); setStopping(false); onTrained(); }
    else if (s.status==="failed") { setError(s.error??"학습 실패"); setLoading(false); setStopping(false); }
  }, [onTrained]);

  useEffect(() => {
    api.getTrainingStatus().then(s => {
      if (s.status==="running") { setLoading(true); setStatus(s); }
      else if (s.status==="stopped") { setStatus(s); }
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!loading) return;
    let cancelled = false;
    const poll = async () => { try { const s=await api.getTrainingStatus(); if(!cancelled) applyStatus(s); } catch{/**/} };
    poll();
    const id = window.setInterval(poll, 1000);
    return () => { cancelled=true; window.clearInterval(id); };
  }, [loading, applyStatus]);

  useEffect(() => { const el=logRef.current; if(el) el.scrollTop=el.scrollHeight; }, [status?.logs]);

  const stopTrain = async () => {
    setStopping(true); setError(null);
    try { await api.stopTraining(); }
    catch(e) { setError(e instanceof Error?e.message:"학습 중지 실패"); setStopping(false); }
  };

  const train = async () => {
    setLoading(true); setStopping(false); setError(null); setMetrics(null);
    setStatus({ status:"running",progress:0,timesteps:0,total_timesteps:totalTs,episodes:0,total_episodes:nEps,train_budget_mode:budget,logs:[],series:EMPTY,metrics:null,error:null });
    try { await api.startTraining(body()); }
    catch(e) { setError(e instanceof Error?e.message:"학습 시작 실패"); setLoading(false); setStatus(null); }
  };

  const series = status?.series ?? EMPTY;
  const pct = Math.round((status?.progress ?? 0) * 100);
  const liveBudget = status?.train_budget_mode ?? budget;
  const progressLabel = liveBudget==="episodes"
    ? `${(status?.episodes??0).toLocaleString()} / ${(status?.total_episodes??nEps).toLocaleString()} ep · ${pct}%`
    : `${(status?.timesteps??0).toLocaleString()} / ${(status?.total_timesteps??totalTs).toLocaleString()} steps · ${pct}%`;

  const showLive   = loading || (status && status.status !== "idle");
  const showCharts = hasTrainChartData(series);

  const convergence = useMemo(() => {
    const rew = series.ep_rew_mean;
    if (rew.length < 10) return null;
    const tail = rew.slice(Math.floor(rew.length*0.8));
    const mean = tail.reduce((a,b)=>a+b,0)/tail.length;
    const std  = Math.sqrt(tail.reduce((a,b)=>a+(b-mean)**2,0)/tail.length);
    const cv   = Math.abs(mean)>0 ? std/Math.abs(mean) : Infinity;
    if (cv < 0.05) return { label:"수렴됨", cls:"badge-ok", note:`변동계수 ${(cv*100).toFixed(1)}%` };
    if (cv < 0.15) return { label:"준수렴", cls:"badge-warn", note:`변동계수 ${(cv*100).toFixed(1)}%` };
    return { label:"미수렴", cls:"badge-err", note:`변동계수 ${(cv*100).toFixed(1)}%` };
  }, [series.ep_rew_mean]);

  return (
    <div className="detail-page">
      <div className="detail-page-title">
        학습 결과
        <span className="page-badge badge badge-warn">Training</span>
      </div>

      {/* ── Control panel ── */}
      <aside className="ctrl-panel">
        <div className="card">
          <div className="card-title">학습 설정</div>

          <div className="budget-pills mb-2">
            {(["timesteps","episodes"] as Budget[]).map(m => (
              <label key={m} className={`budget-pill${budget===m?" active":""}`}>
                <input type="radio" checked={budget===m} onChange={() => setBudget(m)} disabled={loading} />
                {m==="timesteps" ? "Timesteps" : "Episodes"}
              </label>
            ))}
          </div>

          {budget==="timesteps"
            ? <div className="train-field"><label className="field-label">Total Timesteps</label><input type="number" className="input-number" min={1000} step={10000} value={totalTs} onChange={e=>setTotalTs(Number(e.target.value))} disabled={loading} /></div>
            : <div className="train-field"><label className="field-label">목표 에피소드</label><input type="number" className="input-number" min={1} step={10} value={nEps} onChange={e=>setNEps(Number(e.target.value))} disabled={loading} /></div>
          }
          <div className="train-field"><label className="field-label">Learning Rate</label><input type="number" className="input-number" step={0.0001} value={lr} onChange={e=>setLr(Number(e.target.value))} disabled={loading} /></div>
        </div>

        <div className="card">
          <div className="card-title">리워드 설정</div>
          {REWARD_SLIDERS.map(({ key, label, min, max, step }) => {
            const val = reward[key] as number;
            return (
              <div key={key} className="train-slider">
                <div className="train-slider-label">
                  {label} <span className="train-slider-val">{val.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min={min}
                  max={max}
                  step={step}
                  value={val}
                  onChange={e => setReward(prev => ({ ...prev, [key]: Number(e.target.value) }))}
                  disabled={loading}
                />
              </div>
            );
          })}
          <div className="folder-pick-list mt-2">
            {REWARD_TOGGLES.map(({ key, label }) => (
              <label key={key} className="folder-pick-item">
                <input
                  type="checkbox"
                  checked={reward[key] as boolean}
                  disabled={loading}
                  onChange={e => setReward(prev => ({ ...prev, [key]: e.target.checked }))}
                />
                <span>{label}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-title">데이터 범위</div>
          <div className="budget-pills mb-2" style={{ flexWrap:"wrap" }}>
            {([["current","현재"],["period","기간"],["pick","직접선택"]] as [Range,string][]).map(([m,l]) => (
              <label key={m} className={`budget-pill${range===m?" active":""}`} style={{ flex:"unset", padding:"0.38rem 0.65rem" }}>
                <input type="radio" checked={range===m} onChange={() => setRange(m)} disabled={loading} />{l}
              </label>
            ))}
          </div>
          {range==="period" && (
            <div style={{ display:"flex", gap:"0.5rem" }}>
              <div className="train-field" style={{flex:1}}><label className="field-label">시작</label><input type="text" className="input" value={fromDate} onChange={e=>setFrom(e.target.value)} disabled={loading} /></div>
              <div className="train-field" style={{flex:1}}><label className="field-label">종료</label><input type="text" className="input" value={toDate} onChange={e=>setTo(e.target.value)} disabled={loading} /></div>
            </div>
          )}
          {range==="pick" && (
            <div className="folder-pick-list">
              {trainFolders.map(f => (
                <label key={f} className="folder-pick-item">
                  <input type="checkbox" checked={picked.includes(f)} disabled={loading} onChange={() => setPicked(prev => prev.includes(f)?prev.filter(x=>x!==f):[...prev,f])} />
                  <span>{f}</span>
                </label>
              ))}
            </div>
          )}
          <div className="gap-row mt-2">
            <button type="button" className={`btn btn-primary${loading && !stopping?" loading":""}`} onClick={train} disabled={loading || !summary}>
              {loading && !stopping ? "" : "학습 시작"}
            </button>
            {loading && (
              <button type="button" className={`btn btn-ghost${stopping?" loading":""}`} onClick={stopTrain} disabled={stopping}>
                {stopping ? "" : "학습 중지"}
              </button>
            )}
            {!summary && <span className="hint">데이터셋 없음</span>}
          </div>
        </div>

        {modelExists && (
          <div className="card">
            <div className="card-title">모델 상태</div>
            <div className="gap-row">
              <span className="badge badge-ok">✓ 모델 있음</span>
              <button type="button" className="btn btn-ghost btn-xs" onClick={onRefresh}>새로고침</button>
            </div>
          </div>
        )}
      </aside>

      {/* ── Main content ── */}
      <div className="content-area">
        {error && <ExpandableErrorBanner message={error} />}

        {showLive && (
          <div className="card mb-2">
            <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:"0.65rem" }}>
              <div className="card-title" style={{ marginBottom:0 }}>
                학습 상태 ·{" "}
                <span className={`badge ${status?.status==="completed"?"badge-ok":status?.status==="running"?"badge-warn":status?.status==="stopped"?"badge-warn":"badge-err"}`}>
                  {status?.status==="completed"?"완료":status?.status==="running"?"진행 중":status?.status==="stopped"?"중지됨":status?.status==="failed"?"실패":"대기"}
                </span>
              </div>
              {convergence && (
                <span className={`badge ${convergence.cls}`}>{convergence.label} · {convergence.note}</span>
              )}
            </div>
            {loading && (
              <div>
                <div className="progress-bar"><div className="progress-fill" style={{ width:`${pct}%` }} /></div>
                <p className="hint mt-1">{progressLabel}</p>
              </div>
            )}
          </div>
        )}

        {(showCharts || metrics) ? (
          <>
            <div className="tabs mb-2">
              <button type="button" className={`tab-btn${trainTab==="charts"?" active":""}`} onClick={()=>setTrainTab("charts")} disabled={!showCharts}>학습 차트</button>
              <button type="button" className={`tab-btn${trainTab==="log"?" active":""}`} onClick={()=>setTrainTab("log")} disabled={!status?.logs?.length}>로그</button>
              <button type="button" className={`tab-btn${trainTab==="metrics"?" active":""}`} onClick={()=>setTrainTab("metrics")} disabled={!metrics}>최종 지표</button>
            </div>

            {trainTab==="charts" && showCharts && (
              <div className="tab-panel">
                <div className="card chart-wrap mb-2"><PlotChart {...buildTrainRewardChart(series)} /></div>
                <div className="grid-2">
                  <div className="card chart-wrap"><PlotChart {...buildTrainLossChart(series)} /></div>
                  <div className="card chart-wrap"><PlotChart {...buildTrainExplainedVarChart(series)} /></div>
                </div>
              </div>
            )}

            {trainTab==="log" && status?.logs?.length && (
              <div className="tab-panel card">
                <div className="card-title">학습 로그</div>
                <div ref={logRef} className="train-log">
                  {status.logs.map((l,i) => (
                    <div key={i} className={`log-line${l.level==="ERROR"?" log-err":""}`}>
                      <span className="log-time">{fmtTime(l.time)}</span>
                      <span className="log-msg">{l.message}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {trainTab==="metrics" && metrics && (
              <div className="tab-panel card">
                <div className="card-title">최종 학습 지표</div>
                <div className="kpi-grid" style={{ gridTemplateColumns:"repeat(auto-fill, minmax(160px,1fr))" }}>
                  {[
                    { label:"평균 보상",   value:metrics.mean_reward.toFixed(3) },
                    { label:"공정 전환",   value:metrics.mean_oper_sw.toFixed(1) },
                    { label:"제품 전환",   value:metrics.mean_prod_sw.toFixed(1) },
                    { label:"평균 Idle",   value:`${metrics.mean_idle.toFixed(1)}분` },
                    { label:"완료율",      value:`${(metrics.mean_completion*100).toFixed(1)}%` },
                  ].map(m => (
                    <div key={m.label} className="kpi-cell">
                      <div className="kpi-label">{m.label}</div>
                      <div className="kpi-value">{m.value}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : !showLive && (
          <div className="empty-state">
            <div className="empty-state-icon">◌</div>
            <p>학습을 시작하면 진행 상황과 수렴 차트가 여기에 표시됩니다.</p>
          </div>
        )}
      </div>
    </div>
  );
}
