-- @db: Prd
-- batch_info.sql → batch_info.json
-- (PPK, OPER)별 conversion용 LOT_CD / TEMP 레시피
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (기간, YYYYMMDDHHmmss)
--
-- 동일 PPK라도 공정(OPER)마다 다른 LOT_CD/TEMP를 둘 수 있음.
-- 시뮬레이터는 배정 시 (PLAN_PROD_KEY, OPER_ID)로 lookup → conversion/tool 판단.
SELECT
    b.PLAN_PROD_KEY,
    b.OPER_ID,
    b.LOT_CD,
    b.TEMP
FROM BATCH_INFO b
WHERE b.FAC_ID = :FAC_ID
  AND b.RULE_TIMEKEY = :RULE_TIMEKEY
