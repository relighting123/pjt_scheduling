import type {
  AlgorithmCompareResponse,
  AlgorithmId,
  AlgorithmInfo,
  AppConfig,
  DataSummary,
  InferenceResult,
  TestBenchmarkResponse,
  TestDatasetsResponse,
  TrainMetrics,
  TrainStatusResponse,
  RewardConfig,
} from "../types";

export type TrainRequestBody = {
  total_timesteps: number;
  learning_rate: number;
  train_budget_mode?: "timesteps" | "episodes";
  n_episodes?: number;
  input_folder?: string;
  input_folders?: string[];
  from_date?: string;
  to_date?: string;
  fac_id?: string;
} & RewardConfig;

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
    const statusLabel = `HTTP ${res.status}`;
    const raw = await res.text();
    let body: Record<string, unknown> = {};
    try {
      body = raw ? (JSON.parse(raw) as Record<string, unknown>) : {};
    } catch {
      if (raw.trim()) {
        throw new Error(`${statusLabel}\n${raw.trim()}`);
      }
    }
    const detail = body.detail;
    let message: string;
    if (typeof detail === "string") {
      message = detail;
    } else if (Array.isArray(detail)) {
      message = detail.map((d: { msg?: string; loc?: unknown[] }) => {
        const loc = Array.isArray(d.loc) ? d.loc.filter((x) => x !== "body").join(".") : "";
        return loc ? `${loc}: ${d.msg ?? "invalid"}` : (d.msg ?? "invalid");
      }).join("\n");
    } else if (detail && typeof detail === "object") {
      const parts: string[] = [];
      if (Array.isArray((detail as { errors?: string[] }).errors)) {
        parts.push(...(detail as { errors: string[] }).errors);
      }
      if (typeof (detail as { message?: string }).message === "string") {
        parts.push((detail as { message: string }).message);
      }
      if (typeof (detail as { hint?: string }).hint === "string") {
        parts.push((detail as { hint: string }).hint);
      }
      message = parts.length ? parts.join("\n") : `요청 실패 (${res.status})`;
    } else {
      message = `요청 실패 (${res.status})`;
    }
    if (!message.includes(String(res.status))) {
      message = `${statusLabel}\n${message}`;
    }
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
  train: (body: TrainRequestBody) =>
    request<{ message: string; metrics: TrainMetrics; input_folders?: string[] }>("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  startTraining: (body: TrainRequestBody) =>
    request<{ message: string; input_folders?: string[] }>("/api/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  getTrainingStatus: () => request<TrainStatusResponse>("/api/train/status"),
  stopTraining: () =>
    request<{ message: string }>("/api/train/stop", {
      method: "POST",
    }),
  runInference: (opts: {
    algorithm?: AlgorithmId;
    input_folder?: string;
    decision_log?: boolean;
    include_history?: boolean;
    enable_wip_inflow?: boolean;
    save_output?: boolean;
    fac_id?: string;
    rule_timekey?: string;
    nodb?: boolean;
    lot_cd?: string;
    db_load?: boolean;
    db_alias?: string;
    no_history?: boolean;
    max_conversions?: number;
    max_conversions_per_eqp?: number;
  } = {}) =>
    request<InferenceResult>("/api/inference", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        algorithm: opts.algorithm ?? "rl",
        decision_log: opts.decision_log ?? false,
        include_history: opts.include_history ?? false,
        enable_wip_inflow: opts.enable_wip_inflow ?? false,
        save_output: opts.save_output ?? false,
        nodb: opts.nodb ?? false,
        db_load: opts.db_load ?? false,
        no_history: opts.no_history ?? false,
        ...(opts.input_folder ? { input_folder: opts.input_folder } : {}),
        ...(opts.fac_id ? { fac_id: opts.fac_id } : {}),
        ...(opts.rule_timekey ? { rule_timekey: opts.rule_timekey } : {}),
        ...(opts.lot_cd ? { lot_cd: opts.lot_cd } : {}),
        ...(opts.db_alias ? { db_alias: opts.db_alias } : {}),
        ...(opts.max_conversions != null ? { max_conversions: opts.max_conversions } : {}),
        ...(opts.max_conversions_per_eqp != null
          ? { max_conversions_per_eqp: opts.max_conversions_per_eqp }
          : {}),
      }),
    }),
  runCompare: (
    algorithms: AlgorithmId[],
    opts: {
      input_folder?: string;
      decision_log?: boolean;
      include_history?: boolean;
      enable_wip_inflow?: boolean;
      fac_id?: string;
      rule_timekey?: string;
      nodb?: boolean;
      lot_cd?: string;
      max_conversions?: number;
      max_conversions_per_eqp?: number;
    } = {},
  ) =>
    request<AlgorithmCompareResponse>("/api/inference/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        algorithms,
        decision_log: opts.decision_log ?? false,
        include_history: opts.include_history ?? false,
        enable_wip_inflow: opts.enable_wip_inflow ?? false,
        nodb: opts.nodb ?? false,
        ...(opts.input_folder ? { input_folder: opts.input_folder } : {}),
        ...(opts.fac_id ? { fac_id: opts.fac_id } : {}),
        ...(opts.rule_timekey ? { rule_timekey: opts.rule_timekey } : {}),
        ...(opts.lot_cd ? { lot_cd: opts.lot_cd } : {}),
        ...(opts.max_conversions != null ? { max_conversions: opts.max_conversions } : {}),
        ...(opts.max_conversions_per_eqp != null
          ? { max_conversions_per_eqp: opts.max_conversions_per_eqp }
          : {}),
      }),
    }),
  getInferenceResult: (input_folder?: string) =>
    request<InferenceResult>(
      `/api/inference/result${input_folder ? `?input_folder=${encodeURIComponent(input_folder)}` : ""}`,
    ),
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
