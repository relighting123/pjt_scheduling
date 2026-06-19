import type {
  AlgorithmCompareResponse,
  AlgorithmId,
  AlgorithmInfo,
  AppConfig,
  DataSummary,
  InferenceResult,
  SampleScenario,
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
  createSample: (input_folder?: string, scenario: string = "default") =>
    request<{ message: string; input_folder: string; input_dir: string; scenario: string }>("/api/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario, ...(input_folder ? { input_folder } : {}) }),
    }),
  getDataSummary: () => request<DataSummary>("/api/data/summary"),
  getModelStatus: () => request<{ exists: boolean }>("/api/model/status"),
  getAlgorithms: () =>
    request<{ algorithms: AlgorithmInfo[] }>("/api/algorithms"),
  getSampleScenarios: () =>
    request<{ scenarios: SampleScenario[] }>("/api/sample/scenarios"),
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
