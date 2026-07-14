import { useCallback, useState } from "react";
import { api } from "../lib/api";
import type { OptimalBenchResponse } from "../types";

export default function OptimalBenchPanel() {
  const [report, setReport] = useState<OptimalBenchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setReport(await api.getOptimalBench());
    } catch (e) {
      setError(e instanceof Error ? e.message : "실행 실패");
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="card mb-2">
      <div className="card-title">
        최적해 벤치마크
        <span className="page-badge badge badge-info" style={{ marginLeft: "0.5rem" }}>Optimal Bench</span>
      </div>
      <p className="hint mb-2">
        정답을 수학적으로 증명 가능한 소규모 시나리오에서 알고리즘이 실제 최적값에 도달하는지 채점합니다.
        Test 데이터셋과는 무관하며, 코드에 내장된 검증용 케이스만 사용합니다.
      </p>
      <div className="gap-row mb-2">
        <button
          type="button"
          className={`btn btn-primary btn-sm${loading ? " loading" : ""}`}
          onClick={run}
          disabled={loading}
        >
          {loading ? "" : "벤치마크 실행"}
        </button>
        {report && (
          <span className="hint">
            {Object.entries(report.summary)
              .map(([algo, s]) => `${algo} ${s.passed}/${s.total}`)
              .join("  ·  ")}
          </span>
        )}
      </div>
      {error && <p className="hint" style={{ color: "var(--err)" }}>{error}</p>}
      {report && (
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>케이스</th>
                <th>알고리즘</th>
                <th>결과</th>
                <th className="num">생산(실제/최적)</th>
                <th className="num">전환(실제/최적)</th>
                <th>증명</th>
              </tr>
            </thead>
            <tbody>
              {report.runs.map(r => (
                <tr key={`${r.case}-${r.algorithm}`}>
                  <td>{r.case}</td>
                  <td>{r.algorithm}</td>
                  <td style={{ color: r.passed ? "var(--ok)" : "var(--err)", fontWeight: 700 }}>
                    {r.passed ? "PASS" : "FAIL"}
                  </td>
                  <td className="num">{r.actual.production}/{r.target.production}</td>
                  <td className="num">{r.actual.conversions}/{r.target.conversions}</td>
                  <td className="hint">{r.target.proof}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
