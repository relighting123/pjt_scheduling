-- @db: Prd
-- eqp_queue_init.sql → eqp_queue_init.json
-- AI 스케줄러가 배정을 결정하지 않는, 이미 투입이 확정된 EQP별 가공 큐(초기 상태).
-- EQP_ID별로 SEQ_NO 오름차순이 처리 순서이며, START_TM/END_TM은 이미 확정된
-- 시작/종료 시각을 그대로 쓴다(시뮬레이터가 ST로 역산하지 않음).
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (필수)
SELECT
    q.EQP_ID,
    q.SEQ_NO,
    q.LOT_ID,
    q.CARRIER_ID,
    q.OPER_ID,
    q.START_TM,
    q.END_TM,
    q.PLAN_PROD_ATTR_VAL,
    q.WF_QTY
FROM EQP_QUEUE_INIT q
WHERE q.FAC_ID = :FAC_ID
  AND q.RULE_TIMEKEY = :RULE_TIMEKEY
ORDER BY q.EQP_ID, q.SEQ_NO
