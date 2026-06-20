-- availability.sql → availability.json
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (기간, YYYYMMDDHHmmss)
SELECT
    a.EQP_ID,
    a.LOT_ID,
    a.PLAN_PROD_KEY,
    a.ST,
    a.EQP_MODEL,
    a.WF_QTY
FROM AVAILABILITY a
WHERE a.FAC_ID = :FAC_ID
  AND a.RULE_TIMEKEY = :RULE_TIMEKEY
