-- @db: Prd
-- rule_timekey_recent.sql – FAC_ID 기준 최근 N개 RULE_TIMEKEY (최신→과거, 앱에서 오름차순 정렬)
-- collector --prevdays 수집 시 DB 목록 우선 사용
-- 바인드: :FAC_ID, :PREV_DAYS (필수, 정수)
-- ※ PLAN 테이블/컬럼은 사이트 스키마에 맞게 수정하세요.
SELECT RULE_TIMEKEY
FROM (
    SELECT DISTINCT p.RULE_TIMEKEY
    FROM PLAN p
    WHERE p.FAC_ID = :FAC_ID
    ORDER BY p.RULE_TIMEKEY DESC
)
WHERE ROWNUM <= :PREV_DAYS
