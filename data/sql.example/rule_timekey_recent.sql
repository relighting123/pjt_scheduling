-- @db: Prd
-- rule_timekey_recent.sql – FAC_ID 기준 최근 N일 범위 RULE_TIMEKEY
-- collector --prevdays N: 앱에서 현재시각-N일 ~ 현재시각 RULE_TIMEKEY 바인드 생성
-- 바인드: :FAC_ID, :FROM_RULE_TIMEKEY, :TO_RULE_TIMEKEY
-- ※ PLAN 테이블/컬럼은 사이트 스키마에 맞게 수정하세요.
SELECT DISTINCT
    p.RULE_TIMEKEY
FROM PLAN p
WHERE p.FAC_ID = :FAC_ID
  AND p.RULE_TIMEKEY >= :FROM_RULE_TIMEKEY
  AND p.RULE_TIMEKEY <= :TO_RULE_TIMEKEY
ORDER BY p.RULE_TIMEKEY
