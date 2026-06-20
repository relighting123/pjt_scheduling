-- schedule.sql → schedule.json
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (기간, YYYYMMDDHHmmss)
SELECT
    s.EQP_ID,
    s.LOT_ID,
    s.CARRIER_ID,
    s.PLAN_PROD_KEY,
    s.EQP_MODEL,
    s.ST,
    s.SEQ,
    s.STARTTM,
    s.ENDTM
FROM SCHEDULE s
WHERE s.FAC_ID = :FAC_ID
  AND s.RULE_TIMEKEY = :RULE_TIMEKEY
