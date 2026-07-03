-- @db: Prd
-- batch_info.sql → batch_info.json
-- (PPK, OPER)별 conversion용 LOT_CD / TEMP 레시피
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (필수)
SELECT
    b.PLAN_PROD_ATTR_VAL,
    b.OPER_ID,
    b.LOT_CD,
    b.TEMP
FROM BATCH_INFO b
WHERE b.FAC_ID = :FAC_ID
  AND b.RULE_TIMEKEY = :RULE_TIMEKEY
