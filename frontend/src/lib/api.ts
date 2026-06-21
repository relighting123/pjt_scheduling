import type {
  AlgorithmCompareResponse,
  AlgorithmId,
  AlgorithmInfo,
  AppConfig,
  DataSummary,
  InferenceResult,
  SampleScenario,
  GeneratorConfig,
  TestBenchmarkResponse,
  TestDatasetsResponse,
  TrainMetrics,
  TrainStatusResponse,
} from "../types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch {
    throw new Error(
      "서버에 연결할 수 없습니다. API(8000)가 실행 중인지 확인하고 python main.py ui 를 재시작하세요.",
    );
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = body.detail;
    let message: string;
    if (typeof detail === "string") {
      message = detail;
    } else if (Array.isArray(detail)) {
      message = detail.map((d: { msg?: string; loc?: unknown[] }) => {
        const loc = Array.isArray(d.loc) ? d.loc.filter((x) => x !== "body").join(".") : "";
        return loc ? `${loc}: ${d.msg ?? "invalid"}` : (d.msg ?? "invalid");
      }).join("\n");
    } else if (detail?.errors) {
      message = detail.errors.join("\n");
    } else {
      message = `요청 실패 (${res.status})`;
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

function sanitizeGeneratorConfig(
  cfg?: Partial<GeneratorConfig>,
): Partial<GeneratorConfig> | undefined {
  if (!cfg) return undefined;
  const out: Partial<GeneratorConfig> = { ...cfg };
  const seed = out.seed;
  if (seed == null || seed === ("" as unknown as number) || Number.isNaN(Number(seed))) {
    delete out.seed;
  }
  return out;
}

export const api = {
  health: () => request<{ status: string }>("/api/health"),
  getConfig: () => request<AppConfig>("/api/config"),
  setInputFolder: (input_folder: string) =>
    request<{ message: string; input_folder: string; input_dir: string; output_dir: string }>(
      "/api/config/input",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input_folder }),
      },
    ),
  createSample: (opts?: {
    fac_id?: string;
    split?: string;
    scenario?: string;
    bootstrap?: boolean;
    from_date?: string;
    to_date?: string;
    use_period_count?: boolean;
    generator_config?: Partial<GeneratorConfig>;
  }) => {
    const scenario = opts?.scenario ?? "random";
    const generator_config =
      scenario === "random" ? sanitizeGeneratorConfig(opts?.generator_config) : undefined;
    return request<{
      message: string;
      input_folder: string;
      input_dir: string;
      scenario: string;
      generator_config?: GeneratorConfig;
    }>("/api/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scenario,
        fac_id: opts?.fac_id ?? "FAC001",
        split: opts?.split ?? "train",
        bootstrap: opts?.bootstrap ?? false,
        use_period_count: opts?.use_period_count ?? false,
        ...(opts?.from_date ? { from_date: opts.from_date } : {}),
        ...(opts?.to_date ? { to_date: opts.to_date } : {}),
        ...(generator_config ? { generator_config } : {}),
      }),
    });
  },
  fetchDataset: (opts?: {
    fac_id?: string;
    split?: string;
    snapshot?: string;
    from_date?: string;
    to_date?: string;
  }) =>
    request<{ message: string; input_folder: string; input_dir: string }>("/api/fetch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fac_id: opts?.fac_id ?? "FAC001",
        split: opts?.split ?? "train",
        ...(opts?.snapshot ? { snapshot: opts.snapshot } : {}),
        ...(opts?.from_date ? { from_date: opts.from_date } : {}),
        ...(opts?.to_date ? { to_date: opts.to_date } : {}),
      }),
    }),
  getDataSummary: () => request<DataSummary>("/api/data/summary"),
  getModelStatus: () => request<{ exists: boolean }>("/api/model/status"),
  getAlgorithms: () =>
    request<{ algorithms: AlgorithmInfo[] }>("/api/algorithms"),
  getSampleScenarios: () =>
    request<{ scenarios: SampleScenario[] }>("/api/sample/scenarios"),
  getGeneratorConfigDefaults: () =>
    request<{ defaults: GeneratorConfig }>("/api/sample/generator-config"),
  train: (body: {
    total_timesteps: number;
    learning_rate: number;
    w_same_oper: number;
    w_idle_per_min: number;
    train_budget_mode?: "timesteps" | "episodes";
    n_episodes?: number;
    input_folder?: string;
    input_folders?: string[];
    from_date?: string;
    to_date?: string;
    fac_id?: string;
  }) =>
    request<{ message: string; metrics: TrainMetrics; input_folders?: string[] }>("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  startTraining: (body: {
    total_timesteps: number;
    learning_rate: number;
    w_same_oper: number;
    w_idle_per_min: number;
    train_budget_mode?: "timesteps" | "episodes";
    n_episodes?: number;
    input_folder?: string;
    input_folders?: string[];
    from_date?: string;
    to_date?: string;
    fac_id?: string;
  }) =>
    request<{ message: string; input_folders?: string[] }>("/api/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  getTrainingStatus: () => request<TrainStatusResponse>("/api/train/status"),
  runInference: (algorithm: AlgorithmId = "rl", input_folder?: string) =>
    request<InferenceResult>("/api/inference", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ algorithm, ...(input_folder ? { input_folder } : {}) }),
    }),
  runCompare: (algorithms: AlgorithmId[], input_folder?: string) =>
    request<AlgorithmCompareResponse>("/api/inference/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ algorithms, ...(input_folder ? { input_folder } : {}) }),
    }),
  getTestDatasets: (fac_id?: string) =>
    request<TestDatasetsResponse>(
      `/api/test/datasets${fac_id ? `?fac_id=${encodeURIComponent(fac_id)}` : ""}`,
    ),
  getSavedTestBenchmark: (fac_id?: string) =>
    request<TestBenchmarkResponse>(
      `/api/test/benchmark/saved${fac_id ? `?fac_id=${encodeURIComponent(fac_id)}` : ""}`,
    ),
  clearSavedTestBenchmark: (fac_id?: string) =>
    request<TestBenchmarkResponse>(
      `/api/test/benchmark/saved${fac_id ? `?fac_id=${encodeURIComponent(fac_id)}` : ""}`,
      { method: "DELETE" },
    ),
  initTestBenchmark: (algorithms: AlgorithmId[], fac_id?: string) =>
    request<TestBenchmarkResponse>("/api/test/benchmark/init", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ algorithms, ...(fac_id ? { fac_id } : {}) }),
    }),
  runTestBenchmarkOne: (opts: {
    algorithms: AlgorithmId[];
    input_folder: string;
    fac_id?: string;
    progress_current: number;
    progress_total: number;
    done: boolean;
  }) =>
    request<TestBenchmarkResponse>("/api/test/benchmark/run-one", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(opts),
    }),
  runTestBenchmark: (algorithms: AlgorithmId[], opts?: { fac_id?: string; input_folders?: string[] }) =>
    request<TestBenchmarkResponse>("/api/test/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        algorithms,
        ...(opts?.fac_id ? { fac_id: opts.fac_id } : {}),
        ...(opts?.input_folders ? { input_folders: opts.input_folders } : {}),
      }),
    }),
};
