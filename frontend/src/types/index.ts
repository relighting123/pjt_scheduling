import type { ArrangeRow, AssignedLot, AbstractArrangeRow } from "./arrange";

export type { ArrangeRow, AssignedLot, AbstractArrangeRow, AbstractLotUnit } from "./arrange";

export type AlgorithmId = "rl" | "minprogress" | "earliest_st";

export interface AlgorithmInfo {
  id: AlgorithmId;
  name: string;
  description: string;
  requires_model: boolean;
}

export interface ScheduleRecord {
  EQP_ID: string;
  LOT_ID: string;
  CARRIER_ID?: string;
  PLAN_PROD_KEY: string;
  OPER_ID?: string;
  ST?: string;
  SEQ?: number;
  START_TM: number;
  END_TM: number;
  PROC_TIME?: number;
  START_TM_STR?: string;
  END_TM_STR?: string;
  WF_QTY?: number;
  /** 유입 재공(abstract arrange) 여부 */
  ABSTRACT?: boolean;
  /** 유입 재공 투입 가능 시각 (분). 0이면 초기 재공. */
  OPER_IN_TIME?: number;
  LOT_CD?: string;
  TEMP?: string;
  CONVERSION?: boolean;
}

export interface PlanRecord {
  plan_prod_key: string;
  oper_id: string;
  d0_plan_qty: number;
  d1_plan_qty?: number;
  priority?: number;
}

export interface HistorySnap {
  step: number;
  time: number;
  schedule: ScheduleRecord[];
  completed: Record<string, number>;
  wip_waiting?: Record<string, number>;
  idle_total: number;
  oper_sw: number;
  prod_sw: number;
  arrange?: ArrangeRow[];
  arrange_actual?: ArrangeRow[];
  arrange_abstract?: AbstractArrangeRow[];
  assigned?: AssignedLot | null;
  eqp_states?: Record<string, unknown>;
  events?: SimEvent[];
}

export type SimEventKind =
  | "MOVE_OUT"
  | "IDLE"
  | "JOB_ASSIGNED"
  | "CONV_ASSIGNED"
  | "TOOL_RELEASE"
  | "WIP_INJECT"
  | "TOOL_OCCUPY"
  | "PROCESS_END"
  | "IDLE_DECISION"
  | "CONV_START"
  | "CONV_END";

export interface SimEvent {
  time: number;
  kind: SimEventKind | string;
  eqp_id?: string;
  lot_id?: string;
  lot_cd?: string;
  eqp_model?: string;
  plan_prod_key?: string;
  oper_id?: string;
  next_oper_id?: string;
  next_plan_prod_key?: string;
  next_oper_in_time?: number;
  from_lot_cd?: string;
  to_lot_cd?: string;
  from_temp?: string;
  to_temp?: string;
  start_tm?: number;
  end_tm?: number;
  conv_duration_min?: number;
  conv_end_tm?: number;
  tool_from_delta?: number;
  tool_to_delta?: number;
  oper_in_time?: number;
  eqp_status?: string;
}

export interface ConversionPlan {
  eqp_id: string;
  from_lot_cd: string;
  to_lot_cd: string;
  from_temp?: string;
  to_temp?: string;
  conv_start_min: number;
  conv_end_min: number;
  conv_time?: number;
}

export interface InferenceStats {
  idle_total: number;
  oper_switches: number;
  prod_switches: number;
  completed_qty: Record<string, number>;
  conversions?: number;
  remaining_wip?: Record<string, number>;
  remaining_current_wip?: Record<string, number>;
  steps?: number;
  terminated?: boolean;
  truncated?: boolean;
  current_time?: number;
  sim_end_minutes?: number;
  termination_mode?: string;
  enable_wip_inflow?: boolean;
  source_file?: string;
}

export interface DecisionLogFeasibleOption {
  flat: number;
  ppk: string;
  oper_id: string;
  lot_id?: string | null;
}

export interface DecisionLogBlockedBucket {
  ppk: string;
  oper_id: string;
  reason: string;
  detail: string;
  wip_qty?: number;
}

export interface DecisionLogEntry {
  step: number;
  sim_time: number;
  sim_time_after: number;
  time_advanced: boolean;
  eqp_id?: string | null;
  action_requested_flat: number;
  action_requested_ppk?: string | null;
  action_requested_oper?: string | null;
  resolved_flat?: number | null;
  resolved_ppk?: string | null;
  resolved_oper?: string | null;
  selected_eqp_id?: string | null;
  selected_ppk?: string | null;
  selected_oper_id?: string | null;
  selected_lot_id?: string | null;
  selection_reason?: string;
  action_corrected: boolean;
  status: string;
  reason: string;
  reward: number;
  /** 리워드 항목별 분해 (term → 기여값). 합 ≈ reward(클립 전) */
  reward_breakdown?: Record<string, number>;
  /** 벌크 블록 시작 여부 / 블록 크기 / 크기 레벨 (BulkFill 전용) */
  block_start?: boolean;
  block_size?: number | null;
  size_level?: number;
  assigned_lot_id?: string | null;
  failure_code?: string;
  failure_detail?: string;
  feasible_options?: DecisionLogFeasibleOption[];
  blocked_buckets?: DecisionLogBlockedBucket[];
}

export interface InferMeta {
  fac_id: string;
  rule_timekey: string;
  lot_cd?: string | null;
  input_folder: string;
  fetched_from_db: boolean;
  nodb: boolean;
  db_loaded?: boolean;
}

export interface InferRunOptions {
  fac_id?: string;
  rule_timekey?: string;
  nodb?: boolean;
  lot_cd?: string;
  db_load?: boolean;
  db_alias?: string;
  no_history?: boolean;
  decision_log?: boolean;
  enable_wip_inflow?: boolean;
  include_history?: boolean;
  max_conversions?: number;
  max_conversions_per_eqp?: number;
  conversion_minutes?: number;
}

export interface InferenceResult {
  schedule: ScheduleRecord[];
  history: HistorySnap[];
  event_log?: SimEvent[];
  decision_log?: DecisionLogEntry[];
  conversion_plans?: ConversionPlan[];
  stats: InferenceStats;
  plan: PlanRecord[];
  prod_keys: string[];
  oper_ids: string[];
  eqp_ids: string[];
  sim_end_minutes: number;
  /** 시뮬 기준 시각(0분) = RULE_TIMEKEY. 간트 시각축 base. */
  sim_base_time?: string;
  algorithm?: AlgorithmId | string;
  infer_meta?: InferMeta;
}

export interface BatchInfoRecord {
  plan_prod_key: string;
  oper_id: string;
  lot_cd: string;
  temp: string;
}

export interface DataSummary {
  eqp_count: number;
  lot_count: number;
  prod_count: number;
  oper_count: number;
  batch_info_count: number;
  sim_end_minutes: number;
  sim_base_time: string;
  eqp_ids: string[];
  prod_keys: string[];
  oper_ids: string[];
  batch_info: BatchInfoRecord[];
  warnings?: string[];
}

export interface RewardConfig {
  w_same_setup: number;
  w_same_oper: number;
  w_same_prod: number;
  w_prod_switch: number;
  w_idle_per_min: number;
  w_completion: number;
  w_plan_hit: number;
  w_pacing: number;
  pacing_coverage_scale: number;
  w_conversion: number;
  w_avoidable_conversion: number;
  conversion_amortize_factor: number;
  w_bulk_block_bonus: number;
  w_dedication_misuse: number;
  w_redundant_cover: number;
  w_flow_balance: number;
  flow_balance_starving_cover_min: number;
  reward_clip: number;
  use_achievable_target: boolean;
  same_oper_conditional: boolean;
}

export interface EnvDefaults {
  conversion_minutes: number;
  max_conversions: number | null;
  max_conversions_per_eqp: number | null;
}

export interface AppConfig {
  model_dir: string;
  input_folder: string;
  fac_id?: string;
  dataset_split?: string;
  train_snapshot?: string;
  sql_dir?: string;
  input_dir: string;
  output_dir: string;
  infer_input_dir?: string;
  infer_output_dir?: string;
  input_folders: string[];
  default_timesteps: number;
  default_n_episodes: number;
  default_learning_rate: number;
  default_reward: RewardConfig;
  default_env?: EnvDefaults;
}

export interface TrainMetrics {
  mean_reward: number;
  mean_oper_sw: number;
  mean_prod_sw: number;
  mean_idle: number;
  mean_completion: number;
}

export interface TrainLogEntry {
  time: string;
  level: string;
  message: string;
}

export interface TrainSeries {
  timesteps: number[];
  ep_rew_mean: number[];
  eval_timesteps: number[];
  eval_reward: number[];
  policy_loss: number[];
  value_loss: number[];
  explained_variance: number[];
}

export interface TrainStatusResponse {
  status: "idle" | "running" | "completed" | "failed" | "stopped";
  progress: number;
  timesteps: number;
  total_timesteps: number;
  episodes: number;
  total_episodes: number;
  train_budget_mode: "timesteps" | "episodes";
  logs: TrainLogEntry[];
  series: TrainSeries;
  metrics: TrainMetrics | null;
  error: string | null;
}

export interface AlgorithmCompareError {
  algorithm: AlgorithmId;
  message: string;
}

export interface AlgorithmCompareResponse {
  results: InferenceResult[];
  errors: AlgorithmCompareError[];
  plan: PlanRecord[];
  prod_keys: string[];
  oper_ids: string[];
  eqp_ids: string[];
  sim_end_minutes: number;
  infer_meta?: InferMeta;
}

export interface TestDatasetInfo {
  input_folder: string;
  label: string;
}

export interface TestBenchmarkDataset {
  input_folder: string;
  label: string;
  error?: string;
  results: InferenceResult[];
  errors: AlgorithmCompareError[];
  plan: PlanRecord[];
  prod_keys: string[];
  oper_ids: string[];
  eqp_ids: string[];
  sim_end_minutes: number;
}

export interface TestBenchmarkResponse {
  fac_id: string;
  algorithms?: AlgorithmId[];
  status?: "idle" | "running" | "complete";
  progress?: { current: number; total: number; label: string };
  updated_at?: string | null;
  datasets: TestBenchmarkDataset[];
}

export interface TestDatasetsResponse {
  fac_id: string;
  datasets: TestDatasetInfo[];
}

export type AppMode = "dashboard" | "train" | "test" | "inference" | "dataset";
