-- plan.sql → plan.json
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (기간, YYYYMMDDHHmmss)
SELECT
    p.PLAN_PROD_KEY,
    p.OPER_ID,
    p.D0_PLAN_QTY,
    p.D1_PLAN_QTY,
    p.PLAN_PRIORITY
FROM PLAN p
WHERE p.FAC_ID = :FAC_ID
  AND p.RULE_TIMEKEY = :RULE_TIMEKEY
