-- @db: Prd
-- eqp_down.sql → eqp_down.json
-- EQP별 PM/개조 등 다운타임 구간. DOWN_END_TM이 NULL이면 무제한(종료 미정) 다운으로 처리된다.
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (필수)
SELECT
    e.EQP_ID,
    e.DOWN_START_TM,
    e.DOWN_END_TM
FROM EQP_DOWN_PLAN e
WHERE e.FAC_ID = :FAC_ID
  AND e.RULE_TIMEKEY = :RULE_TIMEKEY
