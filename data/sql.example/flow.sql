-- @db: Prd
-- flow.sql → flow.json
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (필수), :LOT_CD (선택, 미사용 시 무시 가능)
SELECT
    f.PLAN_PROD_KEY,
    f.OPER_SEQ,
    f.OPER_ID
FROM FLOW f
WHERE f.FAC_ID = :FAC_ID
  AND f.RULE_TIMEKEY = :RULE_TIMEKEY
ORDER BY f.PLAN_PROD_KEY, f.OPER_SEQ
