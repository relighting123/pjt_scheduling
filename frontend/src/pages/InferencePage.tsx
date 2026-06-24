import { useCallback, useEffect, useMemo, useState } from "react";
import type {
  AlgorithmCompareResponse,
  AlgorithmId,
  InferenceResult,
  ScheduleRecord,
} from "../types";
import { api } from "../lib/api";
import PlotChart from "../components/PlotChart";

/* ─── constants ─────────────────────────────────────────── */

const ALGOS: { id: AlgorithmId; label: string }[] = [
  { id: "rl",          label: "RL"           },
  { id: "minprogress", label: "Min Progress"  },
  { id: "earliest_st", label: "Earliest ST"   },
];

const PROD_COLORS = [
  "#3b82f6","#22c55e","#f97316","#eab308",
  "#a855f7","#ef4444","#06b6d4","#ec4899",
];

const DARK_BASE = {
  paper_bgcolor: "#161b22",
  plot_bgcolor:  "#161b22",
  font: { family: "IBM Plex Mono, monospace", color: "#8b949e", size: 11 },
};

/* ─── helpers ────────────────────────────────────────────── */

function buildProdMap(keys: string[]): Record<string, string> {
  return Object.fromEntries(
    [...keys].sort().map((k, i) => [k, `P${i + 1}`]),
  );
}

function buildProdColors(pmap: Record<string, string>): Record<string, string> {
  return Object.fromEntries(
    Object.keys(pmap)
      .sort()
      .map((k, i) => [k, PROD_COLORS[i % PROD_COLORS.length]]),
  );
}

interface Stats {
  makespan: number;
  completed: number;
  idle: number;
  operSw: number;
  prodSw: number;
}

function computeStats(r: InferenceResult): Stats {
  const makespan = r.schedule.length
    ? Math.max(...r.schedule.map(s => s.END_TM))
    : 0;
  const completed = Object.values(r.stats.completed_qty).reduce((a, b) => a + b, 0);
  return {
    makespan,
    completed,
    idle:   r.stats.idle_total,
    operSw: r.stats.oper_switches,
    prodSw: r.stats.prod_switches,
  };
}

function buildTimeTraces(
  r: InferenceResult,
  prodKeys: string[],
  pmap: Record<string, string>,
  pcolors: Record<string, string>,
) {
  const sorted = [...r.schedule].sort((a, b) => a.END_TM - b.END_TM);
  const cum: Record<string, number> = {};
  const td: Record<string, { x: number[]; y: number[] }> = {};

  for (const pk of prodKeys) {
    cum[pk] = 0;
    td[pk]  = { x: [0], y: [0] };
  }

  for (const row of sorted) {
    const pk = row.PLAN_PROD_KEY;
    if (!(pk in cum)) { cum[pk] = 0; td[pk] = { x: [0], y: [0] }; }
    cum[pk] += row.WF_QTY ?? 1;
    td[pk].x.push(row.END_TM);
    td[pk].y.push(cum[pk]);
  }

  return prodKeys.map(pk => ({
    type:   "scatter"       as const,
    mode:   "lines+markers" as const,
    name:   pmap[pk] ?? pk,
    x:      td[pk]?.x ?? [0],
    y:      td[pk]?.y ?? [0],
    line:   { color: pcolors[pk], shape: "hv" as const, width: 1.5 },
    marker: { color: pcolors[pk], size: 4 },
    hovertemplate: `T=%{x:.0f}min  ${pmap[pk] ?? pk}=%{y} WF<extra></extra>`,
  }));
}

function buildGanttTraces(
  schedule: ScheduleRecord[],
  pmap:    Record<string, string>,
  pcolors: Record<string, string>,
  limit:   number | null,
) {
  const rows = limit != null ? schedule.filter(r => r.START_TM <= limit) : schedule;
  const byProd: Record<string, ScheduleRecord[]> = {};
  for (const r of rows) {
    const pk = r.PLAN_PROD_KEY;
    if (!byProd[pk]) byProd[pk] = [];
    byProd[pk].push(r);
  }
  return Object.entries(byProd).map(([pk, lots]) => ({
    type:        "bar"  as const,
    orientation: "h"   as const,
    name:        pmap[pk] ?? pk,
    y:           lots.map(r => r.EQP_ID),
    x:           lots.map(r => r.END_TM - r.START_TM),
    base:        lots.map(r => r.START_TM),
    customdata:  lots.map(r => [r.LOT_ID, r.END_TM]),
    marker:      { color: pcolors[pk] ?? "#8b949e", opacity: 0.85 },
    hovertemplate:
      `%{y}  %{base:.0f}→%{customdata[1]:.0f}min` +
      `<br>Lot: %{customdata[0]}<extra>${pmap[pk] ?? pk}</extra>`,
  }));
}

function hlShapes(t: number | null) {
  if (t == null) return [];
  return [{
    type: "line" as const,
    x0: t, x1: t, y0: 0, y1: 1,
    yref: "paper" as const,
    line: { color: "#f97316", dash: "dash" as const, width: 1.5 },
  }];
}

const fmt = (v: number) => Math.round(v).toLocaleString();

/* ─── component ──────────────────────────────────────────── */

export default function InferencePage() {
  const [compare,  setCompare]  = useState<AlgorithmCompareResponse | null>(null);
  const [selected, setSelected] = useState<AlgorithmId>("rl");
  const [hlTime,   setHlTime]   = useState<number | null>(null);
  const [running,  setRunning]  = useState(false);
  const [error,    setError]    = useState<string | null>(null);

  const runCompare = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await api.runCompare(
        ["rl", "minprogress", "earliest_st"],
        { include_history: false },
      );
      setCompare(res);
      setHlTime(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }, []);

  useEffect(() => { runCompare(); }, [runCompare]);

  /* ── derived ── */
  const pmap    = useMemo(() => buildProdMap(compare?.prod_keys ?? []),    [compare]);
  const pcolors = useMemo(() => buildProdColors(pmap),                     [pmap]);

  const result = useMemo(
    () => compare?.results.find(r => r.algorithm === selected) ?? null,
    [compare, selected],
  );

  const allStats = useMemo(
    () => compare?.results.map(r => ({ algo: r.algorithm as string, ...computeStats(r) })) ?? [],
    [compare],
  );

  const best = useMemo((): Stats => {
    if (!allStats.length) return { makespan: 0, completed: 0, idle: 0, operSw: 0, prodSw: 0 };
    return {
      makespan:  Math.min(...allStats.map(s => s.makespan)),
      completed: Math.max(...allStats.map(s => s.completed)),
      idle:      Math.min(...allStats.map(s => s.idle)),
      operSw:    Math.min(...allStats.map(s => s.operSw)),
      prodSw:    Math.min(...allStats.map(s => s.prodSw)),
    };
  }, [allStats]);

  const resultStats = useMemo(() => result ? computeStats(result) : null, [result]);

  const timeTraces = useMemo(
    () => result ? buildTimeTraces(result, compare?.prod_keys ?? [], pmap, pcolors) : [],
    [result, compare, pmap, pcolors],
  );

  const ganttTraces = useMemo(
    () => result ? buildGanttTraces(result.schedule, pmap, pcolors, hlTime) : [],
    [result, pmap, pcolors, hlTime],
  );

  const eqpIds = useMemo(
    () => [...new Set((result?.schedule ?? []).map(r => r.EQP_ID))].sort(),
    [result],
  );

  const planQtyByProd = useMemo(() => {
    const m: Record<string, number> = {};
    for (const p of compare?.plan ?? []) {
      m[p.plan_prod_key] = (m[p.plan_prod_key] ?? 0) + p.d0_plan_qty;
    }
    return m;
  }, [compare]);

  const completedByProd = useMemo(() => {
    const m: Record<string, number> = {};
    for (const [k, v] of Object.entries(result?.stats.completed_qty ?? {})) {
      const pk = k.split("|")[0];
      m[pk] = (m[pk] ?? 0) + v;
    }
    return m;
  }, [result]);

  const simEnd = compare?.sim_end_minutes ?? 0;
  const ganttH = Math.max(320, eqpIds.length * 30 + 100);
  const errSet = new Set(compare?.errors.map(e => e.algorithm) ?? []);

  const timeLayout = useMemo(() => ({
    ...DARK_BASE,
    margin: { l: 56, r: 16, t: 10, b: 44 },
    xaxis: {
      title: { text: "Time (min)", standoff: 6 },
      gridcolor: "#21262d", linecolor: "#30363d",
      ...(simEnd > 0 ? { range: [0, simEnd] } : {}),
    },
    yaxis: {
      title: { text: "Cum. WF", standoff: 4 },
      gridcolor: "#21262d", linecolor: "#30363d",
    },
    legend: { bgcolor: "rgba(0,0,0,0)", font: { color: "#8b949e", size: 11 } },
    height: 240,
    shapes: hlShapes(hlTime),
  }), [simEnd, hlTime]);

  const ganttLayout = useMemo(() => ({
    ...DARK_BASE,
    barmode: "overlay" as const,
    margin: { l: 100, r: 16, t: 10, b: 44 },
    xaxis: {
      title: { text: "Time (min)", standoff: 6 },
      gridcolor: "#21262d", linecolor: "#30363d",
      ...(simEnd > 0 ? { range: [0, simEnd] } : {}),
    },
    yaxis: {
      autorange: "reversed" as const,
      gridcolor: "#21262d", linecolor: "#30363d",
      tickfont: { size: 10 },
    },
    legend: { bgcolor: "rgba(0,0,0,0)", font: { color: "#8b949e", size: 11 } },
    height: ganttH,
    shapes: hlShapes(hlTime),
  }), [simEnd, ganttH, hlTime]);

  /* ── status ── */
  const statusText = running ? "Running…" : error ? "Error" : compare ? `Sim end: ${simEnd} min` : "No data";
  const dotCls = running ? "dot dot-run" : error ? "dot dot-err" : compare ? "dot dot-ok" : "dot";

  type MetricKey = keyof Stats;
  const METRIC_ROWS: { key: MetricKey; label: string }[] = [
    { key: "makespan",  label: "Makespan (min)" },
    { key: "completed", label: "Completed WF"   },
    { key: "idle",      label: "Idle (min)"      },
    { key: "operSw",    label: "Oper SW"         },
    { key: "prodSw",    label: "Prod SW"         },
  ];

  /* ─── render ──────────────────────────────────────────── */
  return (
    <div className="app">
      {/* top bar */}
      <header className="topbar">
        <span className="topbar-title">Scheduling Dashboard</span>
        <div className="topbar-spacer" />
        <div className="topbar-status">
          <span className={dotCls} />
          {statusText}
        </div>
        <button
          type="button"
          className="btn btn-primary"
          onClick={runCompare}
          disabled={running}
        >
          {running ? "Running…" : "▶ Run"}
        </button>
      </header>

      {/* main */}
      <main className="main">
        {error && <div className="error-banner">{error}</div>}

        {!compare && !error && (
          <div className="empty">
            <div className="empty-icon">⚙️</div>
            <div>{running ? "Running inference for all algorithms…" : "Loading…"}</div>
          </div>
        )}

        {compare && (
          <>
            {/* algorithm comparison table */}
            <section>
              <div className="section-label">Algorithm Comparison</div>
              <div className="cmp-wrap">
                <table className="cmp-table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      {ALGOS.map(a => (
                        <th key={a.id} style={{ textAlign: "right" }}>
                          {a.label}
                          {errSet.has(a.id) && (
                            <span style={{ color: "#f85149", marginLeft: 4 }}>✕</span>
                          )}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {METRIC_ROWS.map(({ key, label }) => (
                      <tr key={key}>
                        <td className="lc">{label}</td>
                        {ALGOS.map(a => {
                          const s = allStats.find(x => x.algo === a.id);
                          if (!s || errSet.has(a.id)) {
                            return <td key={a.id} className="cmp-none">—</td>;
                          }
                          const v = s[key];
                          return (
                            <td key={a.id} className={v === best[key] ? "cmp-best" : ""}>
                              {fmt(v)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* algo tabs */}
            <div className="algo-tabs">
              {ALGOS.map(a => (
                <button
                  key={a.id}
                  type="button"
                  className={`algo-tab${selected === a.id ? " active" : ""}`}
                  onClick={() => { setSelected(a.id); setHlTime(null); }}
                >
                  {a.label}
                </button>
              ))}
            </div>

            {/* per-algo detail */}
            {errSet.has(selected) ? (
              <div className="error-banner">
                {compare.errors.find(e => e.algorithm === selected)?.message ?? "Algorithm failed"}
              </div>
            ) : !result ? (
              <div className="empty"><div>No results for this algorithm</div></div>
            ) : (
              <>
                {/* KPI row */}
                {resultStats && (
                  <div className="kpi-row">
                    <div className="kpi-chip">
                      <span className="kpi-label">Makespan</span>
                      <span className="kpi-value kv-blue">
                        {fmt(resultStats.makespan)}<span className="kv-unit">min</span>
                      </span>
                    </div>
                    <div className="kpi-chip">
                      <span className="kpi-label">Completed WF</span>
                      <span className="kpi-value kv-accent">{fmt(resultStats.completed)}</span>
                    </div>
                    <div className="kpi-chip">
                      <span className="kpi-label">Idle</span>
                      <span className="kpi-value">
                        {fmt(resultStats.idle)}<span className="kv-unit">min</span>
                      </span>
                    </div>
                    <div className="kpi-chip">
                      <span className="kpi-label">Oper SW</span>
                      <span className="kpi-value kv-warn">{fmt(resultStats.operSw)}</span>
                    </div>
                    <div className="kpi-chip">
                      <span className="kpi-label">Prod SW</span>
                      <span className="kpi-value kv-warn">{fmt(resultStats.prodSw)}</span>
                    </div>
                  </div>
                )}

                {/* cumulative completions chart */}
                <div className="chart-card">
                  <div className="chart-header">
                    <span className="chart-title">Cumulative Completions</span>
                    <span className="chart-hint">Click a point to filter Gantt ↓</span>
                  </div>
                  <PlotChart
                    data={timeTraces}
                    layout={timeLayout}
                    onPointClick={(_, xVal) => {
                      if (xVal !== undefined) setHlTime(xVal);
                    }}
                  />
                </div>

                {/* gantt chart */}
                <div className="chart-card">
                  <div className="chart-header">
                    <span className="chart-title">Gantt Chart</span>
                    {hlTime != null ? (
                      <span className="filter-badge">
                        T ≤ {Math.round(hlTime)} min
                        <button
                          type="button"
                          className="filter-badge-x"
                          onClick={() => setHlTime(null)}
                        >
                          ×
                        </button>
                      </span>
                    ) : (
                      <span className="chart-hint">All lots</span>
                    )}
                  </div>
                  <PlotChart data={ganttTraces} layout={ganttLayout} />
                </div>
              </>
            )}

            {/* product legend */}
            {compare.prod_keys.length > 0 && (
              <section>
                <div className="section-label">Product Legend</div>
                <div className="legend-wrap">
                  <table className="legend-table">
                    <thead>
                      <tr>
                        <th>Abbrev</th>
                        <th>Product Key</th>
                        <th style={{ textAlign: "right" }}>Plan</th>
                        <th style={{ textAlign: "right" }}>Completed</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...compare.prod_keys].sort().map((pk, i) => (
                        <tr key={pk}>
                          <td>
                            <span
                              className="swatch"
                              style={{ background: PROD_COLORS[i % PROD_COLORS.length] }}
                            />
                            <strong>{pmap[pk]}</strong>
                          </td>
                          <td style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: 11 }}>
                            {pk}
                          </td>
                          <td style={{ textAlign: "right", fontFamily: "IBM Plex Mono, monospace" }}>
                            {planQtyByProd[pk] != null ? fmt(planQtyByProd[pk]) : "—"}
                          </td>
                          <td style={{ textAlign: "right", fontFamily: "IBM Plex Mono, monospace", color: "#3fb950" }}>
                            {completedByProd[pk] != null ? fmt(completedByProd[pk]) : "0"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            )}
          </>
        )}
      </main>
    </div>
  );
}
