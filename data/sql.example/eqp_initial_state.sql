-- @db: Prd
-- eqp_initial_state.sql → eqp_initial_state.json
-- EQP별 초기 LOT_CD / TEMP / 직전 PPK·OPER 상태 (conversion 초기 상태)
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (필수), :LOT_CD (선택)
SELECT
    e.EQP_ID,
    e.LOT_CD,
    e.TEMP,
    e.PLAN_PROD_ATTR_VAL,
    e.OPER_ID
FROM EQP_INITIAL_STATE e
WHERE e.FAC_ID = :FAC_ID
  AND e.RULE_TIMEKEY = :RULE_TIMEKEY
  AND (:LOT_CD IS NULL OR e.LOT_CD = :LOT_CD)
