import { useCallback, useEffect, useMemo, useState } from "react";
import PlotChart from "../components/PlotChart";
import FullscreenPanel from "../components/FullscreenPanel";
import ExpandableErrorBanner from "../components/ExpandableErrorBanner";
import { api } from "../lib/api";
import { ALGO_CHART_COLORS, buildAlgorithmGantt, type AlgoCompareEntry } from "../lib/charts";
import type {
  AlgorithmId, AlgorithmInfo, AppConfig, BenchSuiteId, SuiteBenchResponse,
  ToolChangeBenchCase, ToolChangeBenchResponse,
} from "../types";

interface Props { config: AppConfig | null; modelExists: boolean; }

type ViewMode = "summary" | "gantt";
type BenchView = "tool-change" | BenchSuiteId;

const REFERENCE_LABEL = "정답지";
const REFERENCE_COLOR = "#C9A227";

const BENCH_VIEWS: { id: BenchView; label: string; hint: string }[] = [
  {
    id: "tool-change",
    label: "Tool-Change Bench",
    hint: "정답을 구성적으로 증명 가능한 10개 시나리오(단일제품 전담 / 다품종 장비공유 / 재공편중·안전재공)에서, 오라클로 실제 구성한 정답지 스케줄과 알고리즘별 추론 결과를 생산량·전환횟수 기준으로 비교합니다. 실제 test 데이터셋과는 무관하며 코드에 내장된 케이스만 사용합니다.",
  },
  {
    id: "bench",
    label: "BENCH_SUITE (8종)",
    hint: "학습에 쓰는 대표 벤치마크 8종(대칭 기준 / 제품 과잉 / 부하 불균등 / 전환 과중). CLI benchmark/compare_vs_dedication.py --suite bench 와 같은 평가이며, dedication 기준선 대비 점수(생산−전환)를 케이스 전부에 대해 보여줍니다.",
  },
  {
    id: "holdout",
    label: "HOLDOUT_SUITE (6종)",
    hint: "학습에 쓰지 않은 일반화 검증 6종 — 같은 4개 카테고리지만 설비·제품·캐리어 수와 ST·전환시간이 전부 다른 시나리오입니다. 벤치 점수가 '외운 것'인지 '배운 것'인지는 여기서 드러납니다. CLI --suite holdout 과 같은 평가입니다.",
  },
];

function passClass(actual: number, opt: number, better: "eq" | "lte"): string {
  const ok = better === "eq" ? actual === opt : actual <= opt;
  return ok ? "var(--ok)" : "var(--err)";
}

export default function BenchmarkPage({ config, modelExists }: Props) {
  const [algorithms, setAlgorithms] = useState<AlgorithmInfo[]>([]);
  const [selectedAlgos, setSelectedAlgos] = useState<Set<AlgorithmId>>(new Set());
  const [report, setReport] = useState<ToolChangeBenchResponse | null>(null);
  const [suiteReport, setSuiteReport] = useState<SuiteBenchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<ViewMode>("summary");
  const [benchView, setBenchView] = useState<BenchView>("tool-change");
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [facIdOverride, setFacIdOverride] = useState("");
  const [facModel, setFacModel] = useState<{ facId: string; exists: boolean; loading: boolean }>(
    { facId: "", exists: modelExists, loading: true },
  );

  const facId = facIdOverride.trim() || config?.fac_id || "FAC001";
  const activeView = BENCH_VIEWS.find(v => v.id === benchView) ?? BENCH_VIEWS[0];

  // FAC_ID가 바뀌면 백엔드 세션 캐시를 비워(모델 다시 로딩 가능하게 하고)
  // 그 FAC_ID의 모델 존재 여부를 다시 확인한다 — scheduling_rl 선택 가능
  // 여부와 실행 시 로드되는 모델이 항상 현재 FAC_ID 기준이 되도록.
  useEffect(() => {
    let cancelled = false;
    setFacModel(prev => ({ ...prev, loading: true }));
    const timer = setTimeout(() => {
      api.reloadModel(facId)
        .then(r => {
          if (!cancelled) setFacModel({ facId: r.fac_id, exists: r.exists, loading: false });
        })
        .catch(() => {
          if (!cancelled) setFacModel(prev => ({ ...prev, loading: false }));
        });
    }, 400);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [facId]);

  const available = useMemo(
    () => algorithms.filter(a => !a.requires_model || facModel.exists),
    [algorithms, facModel.exists],
  );
  const algoLabels = useMemo(() => {
    const m: Record<string, string> = {};
    algorithms.forEach(a => { m[a.id] = a.name; });
    return m;
  }, [algorithms]);

  useEffect(() => { api.getAlgorithms().then(r => setAlgorithms(r.algorithms)).catch(() => {}); }, []);
  useEffect(() => { setSelectedAlgos(new Set(available.map(a => a.id))); }, [available]);

  const run = useCallback(async () => {
    const ids = [...selectedAlgos].filter(id => available.some(a => a.id === id));
    if (!ids.length) { setError("알고리즘을 선택하세요."); return; }
    setLoading(true); setError(null);
    try {
      if (benchView === "tool-change") {
        const r = await api.getToolChangeBench(ids as AlgorithmId[], facId);
        setReport(r);
        setSelectedCaseId(prev => prev && r.cases.some(c => c.id === prev) ? prev : (r.cases[0]?.id ?? null));
      } else {
        setSuiteReport(await api.getSuiteBench(benchView, ids as AlgorithmId[], facId));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "실행 실패");
    } finally {
      setLoading(false);
    }
  }, [selectedAlgos, available, facId, benchView]);

  const hasData = benchView === "tool-change" ? !!report?.cases?.length : !!suiteReport?.datasets?.length;
  const displayAlgos = benchView === "tool-change" ? (report?.algorithms ?? []) : (suiteReport?.algorithms ?? []);

  const selectedCase: ToolChangeBenchCase | null = useMemo(
    () => (selectedCaseId ? report?.cases.find(c => c.id === selectedCaseId) ?? null : null),
    [report, selectedCaseId],
  );

  const ganttEntries = useMemo((): AlgoCompareEntry[] => {
    if (!selectedCase) return [];
    const entries: AlgoCompareEntry[] = [
      { algorithm: "reference", label: REFERENCE_LABEL, result: selectedCase.reference },
    ];
    for (const r of selectedCase.results) {
      const algo = r.algorithm ?? "";
      entries.push({ algorithm: algo, label: algoLabels[algo] ?? algo, result: r });
    }
    return entries;
  }, [selectedCase, algoLabels]);

  const ganttAxis = useMemo(() => ({
    eqpIds: selectedCase?.reference.eqp_ids ?? [],
    timeStartMinutes: 0,
    timeEndMinutes: selectedCase?.sim_end_minutes ?? 1440,
    simBaseTime: selectedCase?.reference.sim_base_time,
  }), [selectedCase]);

  const entryColor = useCallback((algo: string) => (
    algo === "reference" ? REFERENCE_COLOR : (ALGO_CHART_COLORS[algo] ?? "#888")
  ), []);

  return (
    <div className="detail-page">
      <div className="detail-page-title">
        벤치마크
        <span className="page-badge badge badge-info">{activeView.label}</span>
      </div>

      <aside className="ctrl-panel">
        <div className="card">
          <div className="card-title">설명</div>
          <p className="hint">{activeView.hint}</p>
        </div>

        <div className="card">
          <div className="card-title">설정</div>
          <label className="field-label" htmlFor="bench-fac-id">FAC_ID</label>
          <input
            id="bench-fac-id"
            className="input"
            type="text"
            placeholder={config?.fac_id ?? "FAC001"}
            value={facIdOverride}
            onChange={e => setFacIdOverride(e.target.value)}
            disabled={loading}
          />
          <p className="hint mt-1">
            scheduling_rl 모델을 models/{"{FAC_ID}"}/ 에서 불러옵니다.
            FAC_ID를 바꾸면 다음 실행 때 그 FAC_ID의 모델을 새로 로딩합니다.
          </p>
          <p className="hint mt-1">
            모델 상태:{" "}
            {facModel.loading
              ? "확인 중..."
              : facModel.exists
              ? <span style={{ color: "var(--ok)" }}>있음 ({facModel.facId})</span>
              : <span style={{ color: "var(--err)" }}>없음 ({facId})</span>}
          </p>
        </div>

        <div className="card">
          <div className="card-title">알고리즘</div>
          <div className="algo-list mb-2">
            {algorithms.map(a => {
              const dis = a.requires_model && !facModel.exists;
              return (
                <label key={a.id} className={`algo-option${selectedAlgos.has(a.id) ? " selected" : ""}`}>
                  <input
                    type="checkbox"
                    disabled={dis || loading}
                    checked={selectedAlgos.has(a.id)}
                    onChange={() => setSelectedAlgos(prev => {
                      const n = new Set(prev);
                      n.has(a.id) ? n.delete(a.id) : n.add(a.id);
                      return n;
                    })}
                  />
                  <span className="algo-dot" style={{ background: ALGO_CHART_COLORS[a.id] ?? "#555" }} />
                  <span className={`algo-name${dis ? " algo-name-dim" : ""}`}>{a.name}{dis ? " (모델없음)" : ""}</span>
                </label>
              );
            })}
          </div>
          <button
            type="button"
            className={`btn btn-primary${loading ? " loading" : ""}`}
            onClick={run}
            disabled={loading || selectedAlgos.size === 0}
          >
            {loading ? "" : "벤치마크 실행"}
          </button>
        </div>

        {benchView === "tool-change" && !!report?.cases?.length && (
          <div className="card">
            <div className="card-title">케이스 선택</div>
            <div className="dataset-list">
              {report.cases.map(c => (
                <div
                  key={c.id}
                  className={`dataset-row${selectedCaseId === c.id ? " selected" : ""}`}
                  onClick={() => setSelectedCaseId(c.id)}
                >
                  <div>
                    <div className="dataset-label">{c.id}</div>
                    <div className="dataset-folder">{c.category}</div>
                  </div>
                  <div className="dataset-vals">
                    <span
                      className="dataset-val"
                      style={{ color: passClass(c.reference_kpi.prod, c.reference_kpi.prod_opt, "eq") }}
                    >
                      정답 {c.reference_kpi.prod}/{c.reference_kpi.prod_opt} · {c.reference_kpi.conv}회
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </aside>

      <div className="content-area">
        <div className="tabs mb-2">
          {BENCH_VIEWS.map(v => (
            <button
              key={v.id}
              type="button"
              className={`tab-btn${benchView === v.id ? " active" : ""}`}
              onClick={() => setBenchView(v.id)}
              disabled={loading}
            >
              {v.label}
            </button>
          ))}
        </div>

        {error && <ExpandableErrorBanner message={error} />}

        {!hasData && !loading && (
          <div className="empty-state">
            <div className="empty-state-icon">◌</div>
            <p>
              {benchView === "tool-change"
                ? "벤치마크를 실행하면 정답지 대비 알고리즘 결과가 여기에 표시됩니다."
                : "벤치마크를 실행하면 스위트 전체 케이스의 알고리즘별 점수가 여기에 표시됩니다."}
            </p>
          </div>
        )}

        {benchView === "tool-change" && !!report?.cases?.length && (
          <>
            <div className="tabs">
              <button type="button" className={`tab-btn${view === "summary" ? " active" : ""}`} onClick={() => setView("summary")}>전체 요약값</button>
              <button type="button" className={`tab-btn${view === "gantt" ? " active" : ""}`} onClick={() => setView("gantt")} disabled={!selectedCase}>문제별 간트차트</button>
            </div>

            {view === "summary" && (
              <div className="tab-panel">
                <FullscreenPanel title="알고리즘별 종합 성적" className="card mb-2">
                  <div className="table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>알고리즘</th>
                          <th className="num">생산(실제/정답)</th>
                          <th className="num">생산률</th>
                          <th className="num">전환(실제/정답)</th>
                          <th className="num">정답지 도달 케이스</th>
                        </tr>
                      </thead>
                      <tbody>
                        {displayAlgos.map(a => {
                          const s = report.summary[a];
                          if (!s) return null;
                          return (
                            <tr key={a}>
                              <td style={{ fontWeight: 700, color: ALGO_CHART_COLORS[a] }}>{algoLabels[a] ?? a}</td>
                              <td className="num">{s.prod}/{s.prod_opt}</td>
                              <td className="num" style={{ color: s.prod_pct >= 100 ? "var(--ok)" : s.prod_pct >= 80 ? "var(--warn)" : "var(--err)" }}>{s.prod_pct}%</td>
                              <td className="num" style={{ color: passClass(s.conv, s.conv_opt, "eq") }}>{s.conv}/{s.conv_opt}</td>
                              <td className="num">{s.n_optimal}/{s.n_sets}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </FullscreenPanel>

                <FullscreenPanel title="케이스별 상세" className="card">
                  <div className="table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>케이스</th>
                          <th>구성</th>
                          <th className="num">정답(생산/전환)</th>
                          {displayAlgos.map(a => (
                            <th key={a} className="num" style={{ color: ALGO_CHART_COLORS[a] }}>{algoLabels[a] ?? a}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {report.cases.map(c => (
                          <tr key={c.id} className={selectedCaseId === c.id ? "selected" : undefined} style={{ cursor: "pointer" }} onClick={() => setSelectedCaseId(c.id)}>
                            <td>{c.id}</td>
                            <td className="hint">{c.desc}</td>
                            <td className="num">{c.optimal.production}/{c.optimal.conversions}</td>
                            {displayAlgos.map(a => {
                              const k = c.kpi[a];
                              if (!k) return <td key={a} className="num hint">—</td>;
                              return (
                                <td key={a} className="num">
                                  <span style={{ color: passClass(k.prod, k.prod_opt, "eq") }}>{k.prod}</span>
                                  /{k.prod_opt} · <span style={{ color: passClass(k.conv, k.conv_opt, "eq") }}>{k.conv}</span>회
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </FullscreenPanel>
              </div>
            )}

            {view === "gantt" && selectedCase && (
              <div className="tab-panel gantt-workspace">
                <div className="gantt-workspace-head">
                  <span className="gantt-algo-badge">{selectedCase.id}</span>
                  <span className="hint" style={{ marginLeft: "0.75rem" }}>{selectedCase.test_focus}</span>
                </div>
                <FullscreenPanel title={`${selectedCase.id} — 정답지 vs 추론 결과`} className="chart-wrap gantt-chart-panel">
                  <div className="gantt-compare-stack">
                    {ganttEntries.map(entry => {
                      const color = entryColor(entry.algorithm);
                      return (
                        <section key={entry.algorithm} className="gantt-compare-section">
                          <div className="gantt-compare-section-head">
                            <span
                              className="gantt-algo-badge gantt-algo-badge--compare"
                              style={{ color, borderColor: `${color}55`, background: `${color}18` }}
                            >
                              {entry.label}
                            </span>
                          </div>
                          <PlotChart {...buildAlgorithmGantt(entry, ganttAxis)} scrollable />
                        </section>
                      );
                    })}
                  </div>
                </FullscreenPanel>
                <p className="hint mt-1">{selectedCase.optimal.derivation}</p>
              </div>
            )}
          </>
        )}

        {benchView !== "tool-change" && !!suiteReport?.datasets?.length && (
          <div className="tab-panel">
            <FullscreenPanel title="알고리즘별 종합 성적" className="card mb-2">
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>알고리즘</th>
                      <th className="num">생산(실제/최대)</th>
                      <th className="num">생산률</th>
                      <th className="num">전환</th>
                      <th className="num">점수(생산−전환)</th>
                      <th className="num">전담</th>
                      <th className="num">전량·무전환 케이스</th>
                      <th className="num">vs Dedication</th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayAlgos.map(a => {
                      const s = suiteReport.summary[a];
                      if (!s) return null;
                      const vs = suiteReport.vs_dedication[a];
                      return (
                        <tr key={a}>
                          <td style={{ fontWeight: 700, color: ALGO_CHART_COLORS[a] }}>{algoLabels[a] ?? a}</td>
                          <td className="num">{s.prod}/{s.max}</td>
                          <td className="num" style={{ color: s.prod_pct >= 100 ? "var(--ok)" : s.prod_pct >= 80 ? "var(--warn)" : "var(--err)" }}>{s.prod_pct}%</td>
                          <td className="num">{s.conv}회</td>
                          <td className="num">{s.score}</td>
                          <td className="num">{s.ded}/{s.n_eqp}</td>
                          <td className="num">{s.optimal}/{s.n_sets}</td>
                          <td className="num">
                            {vs ? (
                              <span style={{ color: vs.score_delta > 0 ? "var(--ok)" : vs.score_delta === 0 ? "var(--warn)" : "var(--err)" }}>
                                {vs.score_delta > 0 ? "+" : ""}{vs.score_delta} ({vs.win}승 {vs.tie}무 {vs.loss}패)
                              </span>
                            ) : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </FullscreenPanel>

            <FullscreenPanel title={`케이스별 상세 — 전체 ${suiteReport.datasets.length}종`} className="card">
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>케이스</th>
                      <th>카테고리</th>
                      <th>구성 / 검증 포인트</th>
                      <th className="num">sim(분)</th>
                      {displayAlgos.map(a => (
                        <th key={a} className="num" style={{ color: ALGO_CHART_COLORS[a] }}>{algoLabels[a] ?? a}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {suiteReport.datasets.map(d => (
                      <tr key={d.name}>
                        <td>{d.name}</td>
                        <td className="hint">{d.cat}</td>
                        <td className="hint">{d.desc} {d.tests}</td>
                        <td className="num">{d.sim}</td>
                        {displayAlgos.map(a => {
                          const k = d.algos[a];
                          if (!k) return <td key={a} className="num hint">—</td>;
                          return (
                            <td key={a} className="num">
                              <span style={{ color: passClass(k.prod, k.max, "eq") }}>{k.prod}</span>
                              /{k.max} · <span style={{ color: k.conv === 0 ? "var(--ok)" : "inherit" }}>{k.conv}회</span>
                              {" "}({k.score})
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </FullscreenPanel>
          </div>
        )}
      </div>
    </div>
  );
}
