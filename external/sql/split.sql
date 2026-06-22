-- @db: main
-- split.sql → split.json
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (기간, YYYYMMDDHHmmss)
-- PPK × EQP_MODEL 별 1회 투입 wafer split 크기 (장)
SELECT
    s.PLAN_PROD_KEY,
    s.EQP_MODEL,
    s.SPLIT_QTY
FROM SPLIT_RULE s
WHERE s.FAC_ID = :FAC_ID
  AND s.RULE_TIMEKEY = :RULE_TIMEKEY
