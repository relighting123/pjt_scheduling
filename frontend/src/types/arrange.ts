export interface ArrangeRow {
  eqp_id: string;
  lot_id: string;
  oper_id?: string;
  plan_prod_key: string;
  /** EQP×LOT 소요시간(분) – availability ST */
  st: number;
  proc_time?: number;
  eqp_model: string;
  lot_cd?: string;
  temp?: string;
  /** 초기 스케줄 START_TM (분) */
  initial_start_tm?: number;
  wf_qty: number;
  /** PROC/LOAD/SELE/RESV/WAIT – WAIT만 알고리즘이 자유 배정, 나머지는 이 EQP에 입력 순서대로 강제 배정 */
  lot_stat_cd?: string;
}

export interface AbstractLotUnit {
  lot_id: string;
  oper_in_time: number;
  from_oper: string;
  wf_qty: number;
  carrier_id?: string;
  assigned?: boolean;
}

export interface AbstractArrangeRow {
  abs_key: string;
  plan_prod_key: string;
  eqp_model: string;
  wip_qty: number;
  wip_qty_init: number;
  proc_time: number;
  min_inject_time: number;
  oper_id: string;
  from_oper?: string;
  wf_qty: number;
  plan_priority: number;
  d0_plan_qty: number;
  lot_units?: AbstractLotUnit[];
}

export interface AssignedLot {
  kind?: "actual" | "abstract";
  eqp_id: string;
  lot_id: string;
  plan_prod_key: string;
  eqp_model?: string;
  lot_cd?: string;
  temp?: string;
  conversion?: boolean;
  abs_key?: string;
  /** 소요시간(분) */
  st?: number;
  wf_qty?: number;
  oper_in_time?: number;
  oper_id?: string;
  start_tm: number;
  /** PROC/LOAD/SELE/RESV/WAIT */
  lot_stat_cd?: string;
}
