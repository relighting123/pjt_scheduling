-- @db: Prd
-- eqp_conv_plan.sql → eqp_conv_plan.json
-- 외부(MES 등)에서 확정된, 현재 진행 중이거나 예정된 EQP 전환 계획.
-- START_TM이 RULE_TIMEKEY 이전/동일이면 시뮬 시작과 동시에 즉시 전환이 개시된다.
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (필수)
SELECT
    e.EQP_ID,
    e.FROM_LOT_CD,
    e.FROM_TEMP,
    e.TO_LOT_CD,
    e.TO_TEMP,
    e.START_TM
FROM EQP_CONV_PLAN e
WHERE e.FAC_ID = :FAC_ID
  AND e.RULE_TIMEKEY = :RULE_TIMEKEY
