-- @db: Prd
-- rule_timekey_latest.sql – FAC_ID 기준 최신 RULE_TIMEKEY
-- collector/infer 가 DB 에서 최신 키를 가져올 때 사용 (SQL_JSON_MAP 외 메타 쿼리)
-- 바인드: :FAC_ID (필수)
-- ※ PLAN 테이블/컬럼은 사이트 스키마에 맞게 수정하세요.
SELECT MAX(p.RULE_TIMEKEY) AS RULE_TIMEKEY
FROM PLAN p
WHERE p.FAC_ID = :FAC_ID
