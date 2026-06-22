-- @db: Prd
-- discrete_arrange.sql → discrete_arrange.json
-- Actual(이산) arrange: EQP×LOT 투입 가능 조합 + LOT 현재 공정
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (기간, YYYYMMDDHHmmss)
SELECT
    a.EQP_ID,
    a.LOT_ID,
    a.PLAN_PROD_KEY,
    a.OPER_ID,
    a.SEQ,
    a.CARRIER_ID,
    a.ST,
    a.EQP_MODEL,
    a.WF_QTY
FROM AVAILABILITY a
WHERE a.FAC_ID = :FAC_ID
  AND a.RULE_TIMEKEY = :RULE_TIMEKEY
