```
ecps-churn/
├─ README.md
├─ .env.example # DB/S3 등 환경변수 샘플✅
├─ pyproject.toml # 의존성(lightgbm, shap, pandas, boto3, psycopg2-binary, pyarrow 등)
├─ conf/
│ ├─ model.yaml # LGBM 하이퍼파라미터✅
│ ├─ features.yaml # S3 피처 스키마(컬럼명/타입/스케일 규칙)✅
│ ├─ mapping.yaml # "피처명 → DB 컬럼" 매핑 규칙✅
│ └─ scoring_policy.seed.yaml # 초기 컷포인트(seed) 또는 샘플✅
├─ src/
│ ├─ common/
│ │ ├─ log.py # 로깅 통일✅
│ │ ├─ io.py # S3/로컬 읽기쓰기(파케/CSV 자동)✅
│ │ └─ settings.py # .env + 환경변수 로드 (DB/S3 경로 등)✅
│ ├─ features/
│ │ ├─ schema.py # features.yaml 로드/검증(Pydantic/voluptuous)
│ │ └─ validate.py # 누락/타입/범위 체크(결측 카운트 기록)
│ ├─ model/
│ │ ├─ train_lgbm.py # 학습 + 메트릭 저장 + 모델 아티팩트(S3)
│ │ ├─ predict_lgbm.py # 추론 + SHAP 산출 (선택)✅
│ │ ├─ shap_utils.py # SHAP Top3 추출 유틸
│ │ └─ registry.py # 모델 save/load (S3 키 규약, 버전 문자열)✅
│ ├─ pipelines/
│ │ ├─ batch_train.py # (1) S3→DF 로드 → (2) 학습 → (3) 모델/정책 업데이트✅
│ │ └─ batch_score.py # (1) S3 당일 피처 → (2) 추론 → (3) DB 적재✅
│ ├─ db/
│ │ ├─ writer.py # ← 이전에 준 적재 코드(UPSERT/current 종료 포함)✅
│ │ ├─ mapping.py # mapping.yaml 적용해 DF 컬럼명/스케일 정규화✅
│ │ └─ migrations/
│ │ ├─ 0001_base.sql # 최초 DDL(이미 배포됐으면 비워도 됨)
│ │ └─ 0002_fix_types.sql # 예: total_spend numeric(12,2) 등
│ │ └─ 0003_perf_indexes.sql # 예: 조회/성능 인덱스
│ └─ utils/
│ ├─ date.py # dt 파티션/컷오프 계산
│ └─ df.py # 퍼센타일/클리핑/누락치 처리
├─ docker/
│ ├─ writer.Dockerfile # Lambda(ECR)로 DB writer 실행할 때
│ └─ runtime-requirements.txt
├─ scripts/
│ ├─ run_train_local.sh # 로컬 학습 실행(윈도우면 .ps1로 추가)✅
│ ├─ run_score_local.sh # 로컬 스코어링 실행✅
│ └─ upload_model.sh # S3로 모델/설정 업로드
└─ notebooks/ # 선택(EDA/SHAP 확인 등)
```
