-- @db: Prd
-- plan.sql → plan.json
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (필수), :LOT_CD (선택, 미사용 시 무시 가능)
SELECT
    p.PLAN_PROD_ATTR_VAL,
    p.OPER_ID,
    p.D0_PLAN_QTY,
    p.D1_PLAN_QTY,
    p.PLAN_PRIORITY
FROM PLAN p
WHERE p.FAC_ID = :FAC_ID
  AND p.RULE_TIMEKEY = :RULE_TIMEKEY
