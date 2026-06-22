-- @db: Prd
-- lot_master.sql → lot_master.json
-- LOT별 fallback LOT_CD / TEMP 정보 (batch_info가 없을 때 보조)
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (필수), :LOT_CD (선택)
SELECT
    l.LOT_ID,
    l.LOT_CD,
    l.TEMP
FROM LOT_MASTER l
WHERE l.FAC_ID = :FAC_ID
  AND l.RULE_TIMEKEY = :RULE_TIMEKEY
  AND (:LOT_CD IS NULL OR l.LOT_CD = :LOT_CD)
