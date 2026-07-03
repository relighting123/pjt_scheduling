-- @db: Prd
-- split.sql → split.json
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (필수), :LOT_CD (선택, 미사용 시 무시 가능)
-- PPK × OPER × EQP_MODEL_CD 별 1회 투입 wafer split 크기 (장)
SELECT
    s.PLAN_PROD_ATTR_VAL,
    s.OPER_ID,
    s.EQP_MODEL_CD,
    s.SPLIT_QTY
FROM SPLIT_RULE s
WHERE s.FAC_ID = :FAC_ID
  AND s.RULE_TIMEKEY = :RULE_TIMEKEY
