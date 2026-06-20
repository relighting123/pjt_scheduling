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
}

export interface InferenceStats {
  idle_total: number;
  oper_switches: number;
  prod_switches: number;
  completed_qty: Record<string, number>;
}

export interface InferenceResult {
  schedule: ScheduleRecord[];
  initial_schedule: ScheduleRecord[];
  history: HistorySnap[];
  stats: InferenceStats;
  plan: PlanRecord[];
  prod_keys: string[];
  oper_ids: string[];
  eqp_ids: string[];
  sim_end_minutes: number;
  algorithm?: AlgorithmId;
}

export interface DataSummary {
  eqp_count: number;
  lot_count: number;
  prod_count: number;
  oper_count: number;
  sim_end_minutes: number;
  sim_base_time: string;
  eqp_ids: string[];
  prod_keys: string[];
  oper_ids: string[];
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
  default_learning_rate: number;
  default_w_same_oper: number;
  default_w_idle_per_min: number;
}

export interface TrainMetrics {
  mean_reward: number;
  mean_oper_sw: number;
  mean_prod_sw: number;
  mean_idle: number;
  mean_completion: number;
}

export interface AlgorithmCompareError {
  algorithm: AlgorithmId;
  message: string;
}

export interface AlgorithmCompareResponse {
  results: InferenceResult[];
  errors: AlgorithmCompareError[];
  initial_schedule: ScheduleRecord[];
  plan: PlanRecord[];
  prod_keys: string[];
  oper_ids: string[];
  eqp_ids: string[];
  sim_end_minutes: number;
}

export interface SampleScenario {
  id: string;
  name: string;
  description: string;
  configurable?: boolean;
}

export interface GeneratorConfig {
  n_products: number;
  n_eqps: number;
  n_opers: number;
  lots_per_oper: number;
  wf_qty: number;
  st_min: number;
  st_max: number;
  st_std: number;
  eligibility: number;
  plan_qty_min: number;
  plan_qty_max: number;
  plan_priority: number;
  train_period_count: number;
  test_period_count: number;
  split_qty: number;
  seed: number | null;
}

export type AppMode = "train" | "inference" | "dataset";
