-- @db: Prd
-- abstract_arrange.sql → abstract_arrange.json
-- Abstract arrange: PPK×OPER×EQP_MODEL feasible route (+ ST)
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (필수), :LOT_CD (선택, 미사용 시 무시 가능)
SELECT
    a.PLAN_PROD_KEY,
    a.OPER_ID,
    a.EQP_MODEL,
    a.ST
FROM ABSTRACT_ARRANGE a
WHERE a.FAC_ID = :FAC_ID
  AND a.RULE_TIMEKEY = :RULE_TIMEKEY
