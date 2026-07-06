-- @db: Prd
-- discrete_arrange.sql → discrete_arrange.json
-- Actual(이산) arrange: EQP×LOT 투입 가능 조합 + LOT 현재 공정
-- 공정 순서(SEQ)는 flow.json 의 OPER_SEQ 로 유도 (OPER_ID 필수)
-- LOT_STAT_CD: PROC/LOAD/SELE/RESV/WAIT (미제공 시 WAIT 취급)
--   WAIT 이외는 스케줄러가 자유 배정하지 않고, 해당 EQP_ID에 입력 순서대로 강제 배정한다.
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (필수)
SELECT
    a.EQP_ID,
    a.LOT_ID,
    a.PLAN_PROD_ATTR_VAL,
    a.OPER_ID,
    a.CARRIER_ID,
    a.ST,
    a.EQP_MODEL_CD,
    a.WF_QTY,
    a.LOT_STAT_CD
FROM AVAILABILITY a
WHERE a.FAC_ID = :FAC_ID
  AND a.RULE_TIMEKEY = :RULE_TIMEKEY
