-- @db: Prd
-- flow.sql → flow.json
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (기간, YYYYMMDDHHmmss)
SELECT
    f.PLAN_PROD_KEY,
    f.SEQ_ID,
    f.OPER_ID
FROM FLOW f
WHERE f.FAC_ID = :FAC_ID
  AND f.RULE_TIMEKEY = :RULE_TIMEKEY
ORDER BY f.PLAN_PROD_KEY, f.SEQ_ID
