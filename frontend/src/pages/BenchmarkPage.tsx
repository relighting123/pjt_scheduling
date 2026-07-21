import { useCallback, useEffect, useMemo, useState } from "react";
import PlotChart from "../components/PlotChart";
import FullscreenPanel from "../components/FullscreenPanel";
import ExpandableErrorBanner from "../components/ExpandableErrorBanner";
import { api } from "../lib/api";
import { ALGO_CHART_COLORS, buildAlgorithmGantt, type AlgoCompareEntry } from "../lib/charts";
import type {
  AlgorithmId, AlgorithmInfo, ToolChangeBenchCase, ToolChangeBenchResponse,
} from "../types";

interface Props { modelExists: boolean; }

type ViewMode = "summary" | "gantt";

const REFERENCE_LABEL = "정답지";
const REFERENCE_COLOR = "#C9A227";

function passClass(actual: number, opt: number, better: "eq" | "lte"): string {
  const ok = better === "eq" ? actual === opt : actual <= opt;
  return ok ? "var(--ok)" : "var(--err)";
}

export default function BenchmarkPage({ modelExists }: Props) {
  const [algorithms, setAlgorithms] = useState<AlgorithmInfo[]>([]);
  const [selectedAlgos, setSelectedAlgos] = useState<Set<AlgorithmId>>(new Set());
  const [report, setReport] = useState<ToolChangeBenchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<ViewMode>("summary");
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);

  const available = useMemo(
    () => algorithms.filter(a => !a.requires_model || modelExists),
    [algorithms, modelExists],
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
      const r = await api.getToolChangeBench(ids as AlgorithmId[]);
      setReport(r);
      setSelectedCaseId(prev => prev && r.cases.some(c => c.id === prev) ? prev : (r.cases[0]?.id ?? null));
    } catch (e) {
      setError(e instanceof Error ? e.message : "실행 실패");
    } finally {
      setLoading(false);
    }
  }, [selectedAlgos, available]);

  const hasData = !!report?.cases?.length;
  const displayAlgos = report?.algorithms ?? [];

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
        <span className="page-badge badge badge-info">Tool-Change Bench</span>
      </div>

      <aside className="ctrl-panel">
        <div className="card">
          <div className="card-title">설명</div>
          <p className="hint">
            정답을 구성적으로 증명 가능한 10개 시나리오(단일제품 전담 / 다품종 장비공유 /
            재공편중·안전재공)에서, 오라클로 실제 구성한 <b>정답지</b> 스케줄과
            알고리즘별 추론 결과를 생산량·전환횟수 기준으로 비교합니다. 실제 test
            데이터셋과는 무관하며 코드에 내장된 케이스만 사용합니다.
          </p>
        </div>

        <div className="card">
          <div className="card-title">알고리즘</div>
          <div className="algo-list mb-2">
            {algorithms.map(a => {
              const dis = a.requires_model && !modelExists;
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

        {hasData && (
          <div className="card">
            <div className="card-title">케이스 선택</div>
            <div className="dataset-list">
              {report!.cases.map(c => (
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
        {error && <ExpandableErrorBanner message={error} />}

        {!hasData && !loading && (
          <div className="empty-state">
            <div className="empty-state-icon">◌</div>
            <p>벤치마크를 실행하면 정답지 대비 알고리즘 결과가 여기에 표시됩니다.</p>
          </div>
        )}

        {hasData && (
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
                          const s = report!.summary[a];
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
                        {report!.cases.map(c => (
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
      </div>
    </div>
  );
}
