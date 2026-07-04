-- @db: Prd
-- plan.sql → plan.json
-- 바인드: :FAC_ID (필수), :RULE_TIMEKEY (필수), :LOT_CD (선택, 미사용 시 무시 가능)
-- PLAN_PRIORITY: null 허용. null이 아닌 값 중 작을수록 우선, null은 최하위(마지막) 취급.
-- OVER_PRODUCTION_YN: 'Y'/'N'. 생략 시 'Y'(제약 없음)로 취급.
--   Y: 계획(D0_PLAN_QTY) 달성 후에도 takt time 기준으로 고르게 계속 생산.
--   N: 계획 달성 후에는 OVER_PRODUCTION_YN='Y'인 다른 재공이 하나도 남아있지 않을 때만 생산(그 외엔 배정 지연).
SELECT
    p.PLAN_PROD_ATTR_VAL,
    p.OPER_ID,
    p.D0_PLAN_QTY,
    p.D1_PLAN_QTY,
    p.PLAN_PRIORITY,
    p.OVER_PRODUCTION_YN
FROM PLAN p
WHERE p.FAC_ID = :FAC_ID
  AND p.RULE_TIMEKEY = :RULE_TIMEKEY
