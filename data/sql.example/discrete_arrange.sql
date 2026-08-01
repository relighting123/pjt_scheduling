-- @db: Prd
-- discrete_arrange.sql → discrete_arrange.json
-- Actual(이산) arrange: EQP×LOT 투입 가능 조합 + LOT 현재 공정 (전건 자유배정 후보)
-- 공정 순서(SEQ)는 flow.json 의 OPER_SEQ 로 유도 (OPER_ID 필수)
-- 이미 투입이 확정된 큐(강제 배정)는 여기가 아니라 eqp_queue_init.sql 로 별도 제공한다.
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (필수)
SELECT
    a.EQP_ID,
    a.LOT_ID,
    a.PLAN_PROD_ATTR_VAL,
    a.OPER_ID,
    a.CARRIER_ID,
    a.ST,
    a.EQP_MODEL_CD,
    a.WF_QTY
FROM AVAILABILITY a
WHERE a.FAC_ID = :FAC_ID
  AND a.RULE_TIMEKEY = :RULE_TIMEKEY
