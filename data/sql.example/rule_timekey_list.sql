-- @db: Prd
-- rule_timekey_list.sql – FAC_ID + 구간 RULE_TIMEKEY 목록 (오름차순)
-- collector --from/--to 구간 수집 시 DB 목록 우선 사용
-- 바인드: :FAC_ID, :FROM_RULE_TIMEKEY, :TO_RULE_TIMEKEY (모두 필수)
-- ※ PLAN 테이블/컬럼은 사이트 스키마에 맞게 수정하세요.
SELECT DISTINCT p.RULE_TIMEKEY
FROM PLAN p
WHERE p.FAC_ID = :FAC_ID
  AND p.RULE_TIMEKEY >= :FROM_RULE_TIMEKEY
  AND p.RULE_TIMEKEY <= :TO_RULE_TIMEKEY
ORDER BY p.RULE_TIMEKEY
