-- @db: Prd
-- discrete_arrange.sql → discrete_arrange.json
-- Actual(이산) arrange: EQP×LOT 투입 가능 조합 + LOT 현재 공정
-- 공정 순서(SEQ)는 flow.json 의 SEQ_ID 로 유도 (OPER_ID 필수)
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (필수), :LOT_CD (선택, NULL이면 전체)
SELECT
    a.EQP_ID,
    a.LOT_ID,
    a.PLAN_PROD_KEY,
    a.OPER_ID,
    a.CARRIER_ID,
    a.ST,
    a.EQP_MODEL,
    a.WF_QTY
FROM AVAILABILITY a
LEFT JOIN LOT_MASTER m
    ON m.FAC_ID = a.FAC_ID
   AND m.LOT_ID = a.LOT_ID
   AND m.RULE_TIMEKEY = a.RULE_TIMEKEY
WHERE a.FAC_ID = :FAC_ID
  AND a.RULE_TIMEKEY = :RULE_TIMEKEY
  AND (:LOT_CD IS NULL OR m.LOT_CD = :LOT_CD)
