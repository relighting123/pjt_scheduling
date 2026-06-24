-- @db: Prd
-- tool_capacity.sql → tool_capacity.json
-- LOT_CD × EQP_MODEL 동시 가공 상한
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (필수), :LOT_CD (선택)
SELECT
    t.LOT_CD,
    t.EQP_MODEL,
    t.MAX_TOOL
FROM TOOL_CAPACITY t
WHERE t.FAC_ID = :FAC_ID
  AND t.RULE_TIMEKEY = :RULE_TIMEKEY
  AND (:LOT_CD IS NULL OR t.LOT_CD = :LOT_CD)
