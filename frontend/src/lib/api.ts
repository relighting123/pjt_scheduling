import type {
  AlgorithmCompareResponse,
  AlgorithmId,
  AlgorithmInfo,
  AppConfig,
  DataSummary,
  InferenceResult,
  SampleScenario,
  GeneratorConfig,
  TrainMetrics,
} from "../types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = body.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail?.errors
          ? detail.errors.join("\n")
          : `요청 실패 (${res.status})`;
    throw new Error(message);
  }
  return res.json() as Promise<T>;
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
  }) =>
    request<{
      message: string;
      input_folder: string;
      input_dir: string;
      scenario: string;
      generator_config?: GeneratorConfig;
    }>("/api/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scenario: opts?.scenario ?? "random",
        fac_id: opts?.fac_id ?? "FAC001",
        split: opts?.split ?? "train",
        bootstrap: opts?.bootstrap ?? false,
        use_period_count: opts?.use_period_count ?? false,
        ...(opts?.from_date ? { from_date: opts.from_date } : {}),
        ...(opts?.to_date ? { to_date: opts.to_date } : {}),
        ...(opts?.generator_config ? { generator_config: opts.generator_config } : {}),
      }),
    }),
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
  }) =>
    request<{ message: string; metrics: TrainMetrics }>("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
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
  getInferenceResult: () => request<InferenceResult>("/api/inference/result"),
};
