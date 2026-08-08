-- @db: Prd
-- RTS 추론 결과 적재 테이블
-- (RTS_RSLT_MAS: 매 회차 동일 FAC_ID 전체 교체(최신 결과만 유지), RTS_EQPCONVPLAN_INF/HIS·RTS_EQPCAPA_INF/HIS: 누적, HIS: 이력 누적)
-- 최초 1회: python main.py db-load --ddl-only
-- 또는:     python main.py db-load --ddl --facid FAC001 --split infer
-- 테이블별로 다른 DB에 적재하고 싶으면 config/output_db_routing.yaml.example 참고.

-- ── 스케줄 결과 (가공 계획) ────────────────────────────────────────────────
CREATE TABLE RTS_RSLT_MAS (
    FAC_ID          VARCHAR2(32)   NOT NULL,
    RULE_TIMEKEY    VARCHAR2(14)   NOT NULL,
    LOT_CD          VARCHAR2(64)   NOT NULL,
    TEMPER_VAL      VARCHAR2(32),
    EQP_ID          VARCHAR2(64)   NOT NULL,
    EQP_MODEL_CD    VARCHAR2(32),
    SEQ_NO          NUMBER(10)     NOT NULL,
    PLAN_PROD_ATTR_VAL   VARCHAR2(64)   NOT NULL,
    OPER_ID         VARCHAR2(64)   NOT NULL,
    LOT_ID          VARCHAR2(64)   NOT NULL,
    CARRIER_ID      VARCHAR2(64),
    LOT_STAT_CD     VARCHAR2(16)   DEFAULT 'WAIT' NOT NULL,
    FLOW_ID         VARCHAR2(64),
    WF_QTY          NUMBER(10),
    ST              NUMBER(10),
    PRGS_ENABLE_EQP_LVAL  VARCHAR2(4000),  -- 해당 재공이 투입 가능한 EQP_ID 목록(콤마 구분)
    PLAN_QTY        NUMBER(10),            -- 제품/공정별 당일(D0) 계획 수량
    START_TIME      VARCHAR2(14)   NOT NULL,
    END_TIME        VARCHAR2(14)   NOT NULL,
    PRODUCE_QTY     NUMBER(10)     NOT NULL,
    FUNCTION_NM     VARCHAR2(64)   DEFAULT 'TEST' NOT NULL,
    CRT_USER_ID     VARCHAR2(32)   DEFAULT 'RTS' NOT NULL,
    CRT_TM          TIMESTAMP      DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT PK_RTS_RSLT_MAS PRIMARY KEY (FAC_ID, RULE_TIMEKEY, EQP_ID, SEQ_NO)
);

CREATE INDEX IX_RTS_RSLT_MAS_FAC ON RTS_RSLT_MAS (FAC_ID, RULE_TIMEKEY);
CREATE INDEX IX_RTS_RSLT_MAS_LOT ON RTS_RSLT_MAS (LOT_ID);

-- PRODUCE_QTY/PLAN_QTY는 RULE_TIMEKEY(07:00) 기준 당일 값이며, 다음 07:00에는
-- 그 다음 날 계획으로 넘어간다(누적되지 않음).
CREATE TABLE RTS_RSLT_HIS (
    FAC_ID          VARCHAR2(32)   NOT NULL,
    RULE_TIMEKEY    VARCHAR2(14)   NOT NULL,
    LOT_CD          VARCHAR2(64)   NOT NULL,
    TEMPER_VAL      VARCHAR2(32),
    EQP_ID          VARCHAR2(64)   NOT NULL,
    EQP_MODEL_CD    VARCHAR2(32),
    SEQ_NO          NUMBER(10)     NOT NULL,
    PLAN_PROD_ATTR_VAL   VARCHAR2(64)   NOT NULL,
    OPER_ID         VARCHAR2(64)   NOT NULL,
    LOT_ID          VARCHAR2(64)   NOT NULL,
    CARRIER_ID      VARCHAR2(64),
    LOT_STAT_CD     VARCHAR2(16)   DEFAULT 'WAIT' NOT NULL,
    FLOW_ID         VARCHAR2(64),
    WF_QTY          NUMBER(10),
    ST              NUMBER(10),
    PRGS_ENABLE_EQP_LVAL  VARCHAR2(4000),
    PLAN_QTY        NUMBER(10),
    START_TIME      VARCHAR2(14)   NOT NULL,
    END_TIME        VARCHAR2(14)   NOT NULL,
    PRODUCE_QTY     NUMBER(10)     NOT NULL,
    FUNCTION_NM     VARCHAR2(64)   DEFAULT 'TEST' NOT NULL,
    CRT_USER_ID     VARCHAR2(32)   DEFAULT 'RTS' NOT NULL,
    CRT_TM          TIMESTAMP      DEFAULT SYSTIMESTAMP NOT NULL,
    EXEC_TIMEKEY    VARCHAR2(14)   NOT NULL,
    CONSTRAINT PK_RTS_RSLT_HIS PRIMARY KEY (FAC_ID, EXEC_TIMEKEY, RULE_TIMEKEY, EQP_ID, SEQ_NO)
);

CREATE INDEX IX_RTS_RSLT_HIS_FAC ON RTS_RSLT_HIS (FAC_ID, RULE_TIMEKEY);

-- ── Conversion 계획 ─────────────────────────────────────────────────────────
CREATE TABLE RTS_EQPCONVPLAN_INF (
    FAC_ID                  VARCHAR2(32)   NOT NULL,
    RULE_TIMEKEY            VARCHAR2(14)   NOT NULL,
    PRCS_STAT_CD            VARCHAR2(16)   DEFAULT 'PLAN' NOT NULL,
    JOB_ID                  VARCHAR2(128)  NOT NULL,
    REQ_GBN_CD              VARCHAR2(16)   DEFAULT 'RTS' NOT NULL,
    EQP_ID                  VARCHAR2(64)   NOT NULL,
    EQP_MODEL_CD            VARCHAR2(32),
    TESTER_EQP_MODEL_CD     VARCHAR2(32),
    CONV_START_TM           VARCHAR2(12)   NOT NULL,
    CONV_END_TM             VARCHAR2(12)   NOT NULL,
    CONV_TIME               NUMBER(10)     NOT NULL,
    LOT_CD                  VARCHAR2(64),
    PRB_CARD_NO             VARCHAR2(32)   DEFAULT '-' NOT NULL,
    TEMPER_VAL              VARCHAR2(32),
    PLAN_PROD_ATTR_VAL      VARCHAR2(64),
    TO_LOT_CD               VARCHAR2(64),
    TO_PRB_CARD_NO          VARCHAR2(32)   DEFAULT '-' NOT NULL,
    PRB_CARD_NO_LVAL        VARCHAR2(32)   DEFAULT '-' NOT NULL,
    TO_TEMPER_VAL           VARCHAR2(32),
    TO_PLAN_PROD_ATTR_VAL   VARCHAR2(64),
    OPER_ID                 VARCHAR2(64),
    TO_OPER_ID              VARCHAR2(64),
    REASON_CD               VARCHAR2(32)   DEFAULT 'CONV',
    REASON_CTN              VARCHAR2(256),
    TRANSMIT_YN             CHAR(1)        DEFAULT 'N' NOT NULL,
    TRANSMIT_TM             VARCHAR2(14),
    CRT_USER_ID             VARCHAR2(32)   DEFAULT 'RTS' NOT NULL,
    CHG_USER_ID             VARCHAR2(32)   DEFAULT 'RTS' NOT NULL,
    CRT_TM                  DATE           DEFAULT SYSDATE NOT NULL,
    CHG_TM                  DATE           DEFAULT SYSDATE NOT NULL,
    EXEC_TIMEKEY            VARCHAR2(14)   NOT NULL,
    CONSTRAINT PK_RTS_EQPCONVPLAN_INF PRIMARY KEY (RULE_TIMEKEY, JOB_ID)
);

CREATE INDEX IX_RTS_EQPCONVPLAN_INF_FAC ON RTS_EQPCONVPLAN_INF (FAC_ID, RULE_TIMEKEY);

CREATE TABLE RTS_EQPCONVPLAN_HIS (
    FAC_ID                  VARCHAR2(32)   NOT NULL,
    RULE_TIMEKEY            VARCHAR2(14)   NOT NULL,
    PRCS_STAT_CD            VARCHAR2(16)   DEFAULT 'PLAN' NOT NULL,
    JOB_ID                  VARCHAR2(128)  NOT NULL,
    REQ_GBN_CD              VARCHAR2(16)   DEFAULT 'RTS' NOT NULL,
    EQP_ID                  VARCHAR2(64)   NOT NULL,
    EQP_MODEL_CD            VARCHAR2(32),
    TESTER_EQP_MODEL_CD     VARCHAR2(32),
    CONV_START_TM           VARCHAR2(12)   NOT NULL,
    CONV_END_TM             VARCHAR2(12)   NOT NULL,
    CONV_TIME               NUMBER(10)     NOT NULL,
    LOT_CD                  VARCHAR2(64),
    PRB_CARD_NO             VARCHAR2(32)   DEFAULT '-' NOT NULL,
    TEMPER_VAL              VARCHAR2(32),
    PLAN_PROD_ATTR_VAL      VARCHAR2(64),
    TO_LOT_CD               VARCHAR2(64),
    TO_PRB_CARD_NO          VARCHAR2(32)   DEFAULT '-' NOT NULL,
    PRB_CARD_NO_LVAL        VARCHAR2(32)   DEFAULT '-' NOT NULL,
    TO_TEMPER_VAL           VARCHAR2(32),
    TO_PLAN_PROD_ATTR_VAL   VARCHAR2(64),
    OPER_ID                 VARCHAR2(64),
    TO_OPER_ID              VARCHAR2(64),
    REASON_CD               VARCHAR2(32)   DEFAULT 'CONV',
    REASON_CTN              VARCHAR2(256),
    TRANSMIT_YN             CHAR(1)        DEFAULT 'N' NOT NULL,
    TRANSMIT_TM             VARCHAR2(14),
    CRT_USER_ID             VARCHAR2(32)   DEFAULT 'RTS' NOT NULL,
    CHG_USER_ID             VARCHAR2(32)   DEFAULT 'RTS' NOT NULL,
    CRT_TM                  DATE           DEFAULT SYSDATE NOT NULL,
    CHG_TM                  DATE           DEFAULT SYSDATE NOT NULL,
    EXEC_TIMEKEY            VARCHAR2(14)   NOT NULL,
    CONSTRAINT PK_RTS_EQPCONVPLAN_HIS PRIMARY KEY (EXEC_TIMEKEY, RULE_TIMEKEY, JOB_ID)
);

CREATE INDEX IX_RTS_EQPCONVPLAN_HIS_RTK ON RTS_EQPCONVPLAN_HIS (RULE_TIMEKEY);

-- ── 적정 장비 대수 산출 (PPK/OPER/EQP_MODEL별 필요 대수 vs 실제 배치 대수) ──
-- REQ_EQP_CNT   : 마감까지 잔여량 ÷ 모델 ST 기준 필요 대수(재공 미고려)
-- PLAN_EQP_CNT  : 재공이 그 대수를 먹일 수 있는지 사전 체크 후 확정한 정원(구동 가능 대수)
-- ALLOC_EQP_CNT : 이번 스케줄에서 실제로 그 버킷을 돌린 서로 다른 장비 수
-- (INF: 동일 FAC_ID+RULE_TIMEKEY 기존 행만 교체, 다른 회차는 누적. 옵션: CONFIG.env.capa_output_enabled)
CREATE TABLE RTS_EQPCAPA_INF (
    FAC_ID                  VARCHAR2(32)   NOT NULL,
    RULE_TIMEKEY            VARCHAR2(14)   NOT NULL,
    FUNCTION_NM             VARCHAR2(64)   DEFAULT 'TEST' NOT NULL,
    PLAN_PROD_ATTR_VAL      VARCHAR2(64)   NOT NULL,
    OPER_ID                 VARCHAR2(64)   NOT NULL,
    EQP_MODEL_CD            VARCHAR2(32)   NOT NULL,
    ST                      NUMBER(10,2),          -- 장당 ST(분/매), 모델 기준
    HORIZON_MIN             NUMBER(10),            -- 마감(soft cutoff)까지 남은 시간(분)
    PLAN_QTY                NUMBER(10),            -- D0 계획 수량
    DONE_QTY                NUMBER(10),            -- 산출 시점 기 배정 수량
    REMAIN_QTY              NUMBER(10),            -- 마감까지 처리해야 할 잔여량(재공 상한 반영)
    WIP_QTY                 NUMBER(10),            -- ready 재공 매수
    WIP_CARRIER_CNT         NUMBER(10),            -- ready 재공 carrier 수
    CAPABLE_EQP_CNT         NUMBER(10),            -- 처리 가능(다운 제외) 장비 수
    REQ_EQP_CNT             NUMBER(10)     NOT NULL,
    PLAN_EQP_CNT            NUMBER(10)     NOT NULL,
    ALLOC_EQP_CNT           NUMBER(10)     NOT NULL,
    PLAN_EQP_LVAL           VARCHAR2(4000),        -- 정원으로 선정된 EQP_ID 목록(콤마 구분)
    ALLOC_EQP_LVAL          VARCHAR2(4000),        -- 실제 배치된 EQP_ID 목록(콤마 구분)
    RUN_QTY                 NUMBER(10),            -- 실제 배치 수량(매)
    RUNNABLE_YN             CHAR(1)        DEFAULT 'N' NOT NULL,  -- 재공 기준 구동 가능 조건 여부
    MASK_APPLY_YN           CHAR(1)        DEFAULT 'N' NOT NULL,  -- 정원 마스킹 적용 여부
    REASON_CD               VARCHAR2(16),          -- OK/WIP/EQP/NO_WIP/NO_EQP/DONE/INFLOW
    CRT_USER_ID             VARCHAR2(32)   DEFAULT 'RTS' NOT NULL,
    CRT_TM                  TIMESTAMP      DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT PK_RTS_EQPCAPA_INF PRIMARY KEY
        (FAC_ID, RULE_TIMEKEY, PLAN_PROD_ATTR_VAL, OPER_ID, EQP_MODEL_CD)
);

CREATE INDEX IX_RTS_EQPCAPA_INF_FAC ON RTS_EQPCAPA_INF (FAC_ID, RULE_TIMEKEY);

CREATE TABLE RTS_EQPCAPA_HIS (
    FAC_ID                  VARCHAR2(32)   NOT NULL,
    RULE_TIMEKEY            VARCHAR2(14)   NOT NULL,
    FUNCTION_NM             VARCHAR2(64)   DEFAULT 'TEST' NOT NULL,
    PLAN_PROD_ATTR_VAL      VARCHAR2(64)   NOT NULL,
    OPER_ID                 VARCHAR2(64)   NOT NULL,
    EQP_MODEL_CD            VARCHAR2(32)   NOT NULL,
    ST                      NUMBER(10,2),
    HORIZON_MIN             NUMBER(10),
    PLAN_QTY                NUMBER(10),
    DONE_QTY                NUMBER(10),
    REMAIN_QTY              NUMBER(10),
    WIP_QTY                 NUMBER(10),
    WIP_CARRIER_CNT         NUMBER(10),
    CAPABLE_EQP_CNT         NUMBER(10),
    REQ_EQP_CNT             NUMBER(10)     NOT NULL,
    PLAN_EQP_CNT            NUMBER(10)     NOT NULL,
    ALLOC_EQP_CNT           NUMBER(10)     NOT NULL,
    PLAN_EQP_LVAL           VARCHAR2(4000),
    ALLOC_EQP_LVAL          VARCHAR2(4000),
    RUN_QTY                 NUMBER(10),
    RUNNABLE_YN             CHAR(1)        DEFAULT 'N' NOT NULL,
    MASK_APPLY_YN           CHAR(1)        DEFAULT 'N' NOT NULL,
    REASON_CD               VARCHAR2(16),
    CRT_USER_ID             VARCHAR2(32)   DEFAULT 'RTS' NOT NULL,
    CRT_TM                  TIMESTAMP      DEFAULT SYSTIMESTAMP NOT NULL,
    EXEC_TIMEKEY            VARCHAR2(14)   NOT NULL,
    CONSTRAINT PK_RTS_EQPCAPA_HIS PRIMARY KEY
        (FAC_ID, EXEC_TIMEKEY, RULE_TIMEKEY, PLAN_PROD_ATTR_VAL, OPER_ID, EQP_MODEL_CD)
);

CREATE INDEX IX_RTS_EQPCAPA_HIS_FAC ON RTS_EQPCAPA_HIS (FAC_ID, RULE_TIMEKEY);

-- ── KPI 성능 이력 (옵션: 추론 시 --save-kpi / save_kpi=true) ──────────────────
CREATE TABLE RTS_PERFMON_HIS (
    FAC_ID          VARCHAR2(32)   NOT NULL,
    RULE_TIMEKEY    VARCHAR2(14)   NOT NULL,
    FUNCTION_NM     VARCHAR2(64)   NOT NULL,
    KPI_NM          VARCHAR2(64)   NOT NULL,
    KPI_VAL         NUMBER(18,4),
    CRT_USER_ID     VARCHAR2(32)   DEFAULT 'RTS' NOT NULL,
    CRT_TM          TIMESTAMP      DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT PK_RTS_PERFMON_HIS PRIMARY KEY (RULE_TIMEKEY, FAC_ID, FUNCTION_NM, KPI_NM)
);

CREATE INDEX IX_RTS_PERFMON_HIS_FAC ON RTS_PERFMON_HIS (FAC_ID, RULE_TIMEKEY);

-- ── 검증 이력: 투입 불가 장비 재공 선택 건수 (EQP/PPK/OPER 조합별, 옵션: --save-kpi / save_kpi=true) ──
CREATE TABLE RTS_VALIDATION (
    FAC_ID                  VARCHAR2(32)   NOT NULL,
    RULE_TIMEKEY            VARCHAR2(14)   NOT NULL,
    FUNCTION_NM             VARCHAR2(64)   NOT NULL,
    EQP_ID                  VARCHAR2(64)   NOT NULL,
    PLAN_PROD_ATTR_VAL      VARCHAR2(64)   NOT NULL,
    OPER_ID                 VARCHAR2(64)   NOT NULL,
    VIOLATION_CNT           NUMBER(10)     NOT NULL,
    CRT_USER_ID             VARCHAR2(32)   DEFAULT 'RTS' NOT NULL,
    CRT_TM                  TIMESTAMP      DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT PK_RTS_VALIDATION PRIMARY KEY
        (RULE_TIMEKEY, FAC_ID, FUNCTION_NM, EQP_ID, PLAN_PROD_ATTR_VAL, OPER_ID)
);

CREATE INDEX IX_RTS_VALIDATION_FAC ON RTS_VALIDATION (FAC_ID, RULE_TIMEKEY);
