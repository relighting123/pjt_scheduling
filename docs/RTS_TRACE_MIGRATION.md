# RTS_TRACE_INF/HIS 적용 가이드

## 개요

배정 시점에 idle EQP가 고를 수 있었던 (PPK,OPER) 버킷 후보 전부(실제 선택된 것 포함)를
남기는 `RTS_TRACE_INF`/`RTS_TRACE_HIS` 테이블을 추가하는 기능입니다. `RTS_EQPCONVPLAN_INF/HIS`와
동일한 패턴으로 동작합니다:

- **RTS_TRACE_INF**: 동일 `FAC_ID + RULE_TIMEKEY` 기존 행만 DELETE 후 INSERT (같은 회차를
  다시 돌려도 중복이 쌓이지 않음, 최신 결과만 유지)
- **RTS_TRACE_HIS**: `EXEC_TIMEKEY`가 PK에 포함되어 회차마다 계속 누적

`RTS_RSLT_MAS`/`RTS_RSLT_HIS`는 이 기능과 무관하게 기존 컬럼 그대로입니다(변경 없음).

관련 코드: `simulation/simulator.py`(`_candidate_bucket_snapshot`),
`data/writer/rts_json.py`(`_build_rts_trace_rows`), `data/writer/rts_sql.py`(`_insert_rts_trace`).

---

## 1. 신규 설치 (아직 RTS_* 테이블이 없는 환경)

`data/sql.example/rts_output_tables.sql`에 `RTS_TRACE_INF`/`HIS` DDL이 이미 포함되어
있으므로 평소와 동일하게 한 번만 실행하면 됩니다.

```bash
python main.py db-load --ddl-only
```

별도 조치가 필요 없습니다. 아래 2번은 **건너뛰세요**.

---

## 2. 기존 배포 환경 (RTS_RSLT_MAS 등이 이미 있는 DB)

`db-load --ddl-only`는 `rts_output_tables.sql` 전체를 실행하기 때문에 이미 있는 테이블의
`CREATE TABLE`에서 오류가 납니다. 새로 추가된 두 테이블만 별도 파일로 분리해뒀으니
이것만 1회 실행하세요: `data/sql.example/rts_trace_create.sql`.

### 방법 A — SQL 클라이언트로 직접 실행 (권장, 가장 단순)

`sqlplus`, SQL Developer, DBeaver 등 사용 중인 Oracle 클라이언트로 대상 스키마에 접속해
파일 내용을 그대로 실행합니다.

```bash
sqlplus <user>/<password>@<dsn> @data/sql.example/rts_trace_create.sql
```

### 방법 B — 프로젝트 DB 설정 그대로 재사용

`config/databases.yaml`에 등록된 alias(예: `Prd`)로 접속하고 싶다면 아래 스니펫을
1회성으로 실행합니다.

```bash
python - <<'EOF'
from pathlib import Path
from data.db_registry import DbRegistry
from data.writer.db_load import execute_sql_file

registry = DbRegistry()
with registry:
    conn = registry.connect("Prd")  # 대상 DB alias로 교체
    execute_sql_file(conn, Path("data/sql.example/rts_trace_create.sql"))
EOF
```

운영/개발 DB를 분리해 쓰는 경우 `APP_ENV=production`(또는 `development`)을 붙여서
실행하세요(`docs/DEPLOYMENT.md` 1.2절 참고).

---

## 3. 적용 확인

```sql
-- 테이블 생성 확인
SELECT table_name FROM user_tables WHERE table_name LIKE 'RTS_TRACE%';

-- 컬럼 확인
SELECT column_name, data_type FROM user_tab_columns
WHERE table_name = 'RTS_TRACE_INF' ORDER BY column_id;
```

추론을 한 번 실행해 실제로 행이 쌓이는지 확인합니다.

```bash
python main.py infer --facid FAC001 --algorithm scheduling_rl
python main.py db-load --facid FAC001 --split infer
```

```sql
SELECT COUNT(*) FROM RTS_TRACE_INF WHERE FAC_ID = 'FAC001';
```

`RTS_RSLT_MAS`의 한 행(`EQP_ID + CARRIER_ID`)에 대해 `RTS_TRACE_INF`에 여러 후보 행이
붙어있는지도 확인해볼 수 있습니다.

```sql
SELECT t.CANDIDATE_PPK, t.CANDIDATE_OPER_ID, t.IS_SELECTED, t.WIP_SHARE, t.COVERAGE_RATIO
FROM RTS_TRACE_INF t
WHERE t.FAC_ID = 'FAC001' AND t.EQP_ID = '<임의 EQP_ID>' AND t.CARRIER_ID = '<임의 CARRIER_ID>'
ORDER BY t.IS_SELECTED DESC;
```

---

## 4. 데이터량이 부담될 경우

배정 건수 × 평균 후보 수만큼 행이 늘어납니다. 필요하면 `config.py`의
`EnvConfig.trace_output_enabled`를 `False`로 두면 두 테이블 모두 빈 상태로 출력됩니다
(코드 변경 없이 설정값만 조정하면 됨).

```python
# config.py
trace_output_enabled: bool = False
```

---

## 5. 롤백

기능을 완전히 되돌리고 싶다면(테이블까지 제거):

```sql
DROP TABLE RTS_TRACE_INF PURGE;
DROP TABLE RTS_TRACE_HIS PURGE;
```

`RTS_RSLT_MAS`/`RTS_RSLT_HIS`는 이 기능으로 컬럼이 추가된 적이 없으므로 별도 조치가
필요 없습니다.
