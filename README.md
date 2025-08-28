ecps-churn/
├─ docs/                                   ⏳ 아키텍처/ADR/운영가이드/쿼리모음
│  ├─ architecture.md
│  ├─ adr/                                 ⏳ 의사결정 기록(ADR)
│  └─ operations.md                        ⏳ 운영/배포/롤백 가이드
├─ config/
│  ├─ risk_cuts.json                       ✅✅ risk_band 컷포인트(예: {"VH":0.9,"H":0.75,"M":0.5})
│  ├─ columns_map.yaml                     ✅✅ 산출→DB 컬럼 매핑
│  ├─ features.yaml                        ✅✅ 피처 스키마/검증 규칙
│  ├─ model_lgbm.yaml                      ✅✅ LGBM 하이퍼파라미터/seed/early_stopping
│  ├─ data_split.yaml                      ✅✅ train/valid/test 컷오프/윈도우
│  ├─ env/
│  │  ├─ dev.env.example                   ✅✅ 로컬/개발 환경변수 템플릿
│  │  └─ prod.env.example                  ⏳ 운영용 템플릿
│  └─ stepfunctions/
│     ├─ daily_scoring_input.json          ⏳ 일일 스코어링 입력 템플릿
│     └─ weekly_training_input.json        ⏳ 주간 트레이닝 입력 템플릿
├─ data/                                   (로컬 샘플/더미)
│  ├─ raw/                                 ✅ 원천 샘플
│  └─ samples/                             ✅ 가공/피처/예측 샘플
├─ src/
│  ├─ common/
│  │  ├─ io.py                             ✅ S3/로컬 I/O, CSV/Parquet 헬퍼
│  │  ├─ log.py                            ✅ 표준 로깅 설정
│  │  └─ utils.py                          ✅ 시간대, 시드, 검증 유틸
│  ├─ fe/                                  (SageMaker Processing: 피처)
│  │  ├─ fe_main.py                        ✅ 로드→변환→검증→아웃풋
│  │  └─ validate_features.py              ✅ features.yaml 기반 스키마 체크
│  ├─ train/                               (SageMaker Training)
│  │  ├─ train_lgbm.py                     ✅ 학습/모델 저장(메타 포함)
│  │  └─ eval_metrics.py                   ✅ AUC/PR-AUC/KS/임계값 튜닝
│  ├─ infer/                               (Batch Transform/Processing 추론)
│  │  ├─ predict.py                        ✅ 배치 추론(확률 산출)
│  │  └─ postprocess.py                    ✅ risk_band/percentile/타임스탬프
│  └─ pipelines/                           (로컬·SM 진입점)
│     ├─ run_processing.py                 ✅ 피처 파이프 실행
│     ├─ run_training.py                   ✅ 학습→평가→아티팩트 저장
│     └─ run_scoring.py                    ✅ 데이터 컷오프→추론→후처리
├─ lambda/
│  ├─ db_writer/                           (RDS 적재)
│  │  ├─ handler.py                        ✅ S3 or payload→upsert 실행
│  │  ├─ requirements.txt                  ✅ psycopg2-binary 등
│  │  └─ Dockerfile                        ✅ linux/arm64(사용자 선호 반영)
│  └─ campaign_executor/                   (선택: 12시 캠페인 자동 시행)
│     ├─ handler.py                        ⏳ 타겟 선별→로그/액션 insert→발송
│     ├─ requirements.txt                  ⏳
│     └─ Dockerfile                        ⏳
├─ jobs/                                   (배치 스크립트/테스트 이벤트)
│  ├─ run_daily_scoring.sh                 ✅ 로컬/CI에서 일일 스코어링 실행
│  ├─ sample_event_writer.json             ✅ db_writer 테스트 페이로드
│  └─ sample_event_campaign.json           ⏳ 캠페인 실행 샘플
├─ sql/
│  ├─ analytics.sql                        ✅✅ analytics 스키마/테이블/인덱스
│  ├─ mart.sql                             ⏳ 마트/뷰 정의
│  ├─ upsert_prediction.sql                ✅✅ INSERT ... ON CONFLICT ...
│  └─ campaign_sql.sql                     ⏳ 타겟선정/로그/액션 SQL
├─ infra/                                  (IaC)
│  ├─ rds/                                 ⏳ 파라미터/보안그룹/엔드포인트
│  ├─ s3/                                  ⏳ 버킷/라이프사이클
│  ├─ sagemaker/                           ⏳ Processing/Training/ModelRegistry
│  ├─ stepfunctions/                       ⏳ 상태머신 정의(JSON/YAML)
│  ├─ eventbridge/                         ⏳ 크론(배치 00:00 KST, 캠페인 12:00)
│  └─ iam/                                 ⏳ 권한 정책
├─ tests/
│  ├─ test_fe.py                           🧪 피처 스키마/결측/범위 체크
│  ├─ test_postprocess.py                  🧪 리스크밴드/퍼센타일 산출 검증
│  ├─ test_db_writer.py                    🧪 트랜잭션/업서트 스모크
│  └─ test_campaign_sql.py                 🧪 타겟쿼리 결과/안전장치 검증
├─ .github/workflows/
│  └─ ci.yml                               ✅ lint/test/build(→ECR) 파이프라인
├─ Makefile                                ✅ make build/train/score/deploy 타겟
├─ pyproject.toml                          ✅ 패키징/의존성(또는 requirements.txt)
└─ README.md                               ✅ 프로젝트 개요/실행법/구성

------------------------------------

0) 스키마·계약 고정 (재작업 방지의 핵심)

sql/analytics.sql

왜: DB 컬럼·타입·PK/UK를 최종 고정(모든 파일의 기준점).

Done when: analytics.prediction_user_churn 정의(컬럼·인덱스·UK), 필요한 enum/스키마까지 포함.

config/columns_map.yaml

왜: “산출 컬럼명 → DB 컬럼명” 단방향 매핑 계약 확정. 추론/후처리/적재가 함께 참조.

Done when: 모델 산출 필드(예: user_id, scored_at, model_name, model_version, feature_version, data_cutoff_at, churn_score, predicted_label_mcc, predicted_label_bal, risk_band, score_percentile, imputations_count …)가 DB와 1:1 매핑.

sql/upsert_prediction.sql

왜: 적재 방식(INSERT…ON CONFLICT)도 계약에 포함. 람다와 파이프라인이 그대로 호출.

Done when: 충돌 키(예: user_id, scored_at)와 업데이트 규칙(예: spec_id류는 COALESCE 등)이 확정.

config/features.yaml

왜: 피처 스키마/타입/결측/인코딩 규칙을 단일 진실 소스로 고정.

Done when: 각 피처의 dtype, 허용 범위, null 처리, 카테고리 라벨링/인코딩 규칙까지 명시.

config/risk_cuts.json

왜: risk_band 경계(VH/H/M/L) 재사용. 모델·후처리·리포트가 동일 규칙 사용.

Done when: 밴드명과 임계값(상→하 정렬) 명확. 예: {"VH":0.90,"H":0.75,"M":0.50,"L":0.0}

config/data_split.yaml

왜: 학습/검증/테스트 컷오프 기준(기간/윈도우/슬라이딩)을 통일.

Done when: 기준일·윈도우·seed 고정.

config/model_lgbm.yaml

왜: 하이퍼파라미터/early stopping/seed/클래스가중치 등 재현성 보장.

Done when: 파라미터와 평가 지표 목록이 명확, 저장 경로/모델 레지스트리 키 포함.

config/env/dev.env.example

왜: 모든 스크립트·람다가 참조하는 공통 ENV 키 고정(나중에 .env만 채워 사용).

Done when: DB_HOST/PORT/NAME/USER/PASSWORD, AWS_REGION, S3_BUCKET, ARTIFACT_PREFIX, SCHEMA/TABLE, LOG_LEVEL … 등이 정의.

1) 공통 유틸 (이후 모든 코드가 가져다 씀)

pyproject.toml

왜: 의존성와 파이썬 버전 고정(모듈 import 경로, 툴링 일관성).

Done when: lightgbm, pandas, numpy, pyarrow, boto3, pydantic(or voluptuous), psycopg2-binary, python-dotenv, loguru(or std logging) 등 명시.

src/common/log.py

왜: 포맷/레벨/구조로그(배치ID 등) 통일.

Done when: 환경변수 기반 LOG_LEVEL, JSON/텍스트 선택, 타임존(KST/UTC) 일관.

src/common/utils.py

왜: 시드 고정, 경로 합성, 타임존/컷오프 계산 등 공통 로직.

Done when: 재사용 함수(예: now_utc(), set_global_seed(), parse_cutoff(config)).

src/common/io.py

왜: S3/로컬 I/O 계약(읽기/쓰기/파케·CSV) 통일.

Done when: read_df(path), write_df(df, path, format), s3://와 로컬 모두 동작.

2) 피처 처리 (Processing 단계)

src/fe/validate_features.py

왜: 프로덕션에서 가장 잦은 수정 포인트 → 초기에 검증기 견고화.

Done when: features.yaml대로 컬럼·dtype·null·카테고리 미스매치시 명확한 에러.

src/fe/fe_main.py

왜: 로드→변환→검증→저장을 한 곳에서 실행(Processing 엔트리포인트).

Done when: 입력/출력 경로·컷오프 파라미터화, validate_features 통과, 아티팩트 경로 규칙 확정.

3) 학습/평가 (Training 단계)

src/train/eval_metrics.py

왜: 지표/스레시홀드 정책(MCC/BAL 튜닝) 먼저 고정해야 모델코드가 흔들리지 않음.

Done when: ROC-AUC/PR-AUC/KS/정밀도-재현율 커브 & MCC/Balanced Accuracy 기반 최적 임계값 산출.

src/train/train_lgbm.py

왜: 위 지표 함수를 호출하는 학습 엔트리, 모델/피처 버전 메타 저장.

Done when: 재현성(seed), 모델/피처 버전 태깅, 아티팩트(S3/로컬) 저장, 로그에 성능 요약.

4) 추론/후처리 (Batch Transform 단계)

src/infer/postprocess.py

왜: 리스크밴드/퍼센타일/타임스탬프 정책(계약)을 확정.

Done when: risk_cuts.json 사용, 동배치 내 percentile 계산, imputations_count 등 포함.

src/infer/predict.py

왜: 모델 로딩→배치 추론→후처리 연결.

Done when: 입력 피처 스냅샷·모델 아티팩트 경로 파라미터화, 산출 스키마는 columns_map.yaml과 일치.

5) 파이프라인 엔트리 (오케스트레이션)

src/pipelines/run_processing.py

왜: 컷오프/입출력/환경 파라미터를 한 번에 주입해 재사용.

Done when: CLI/ENV 모두 지원, 성공 시 아웃풋 경로 로그.

src/pipelines/run_training.py

왜: Processing 산출물→Training→평가→아티팩트 저장까지 일괄.

Done when: 종료 코드 0, 최종 성능/임계값 로그, 모델 버전 반환.

src/pipelines/run_scoring.py

왜: Processing 재사용→Predict→Postprocess→산출 저장.

Done when: 스코어 산출 프레임이 columns_map/upsert와 100% 호환.

6) 적재 러너 (Lambda)

lambda/db_writer/requirements.txt

왜: 컨테이너 빌드 전에 의존성 고정(arm64 호환).

Done when: psycopg2-binary, boto3, pandas, pyarrow, python-dotenv 등.

lambda/db_writer/Dockerfile

왜: 사용자 선호(arm64, Schema V2) 반영한 빌드.

Done when: linux/arm64, pip install -r requirements.txt, non-root, tzdata(옵션).

lambda/db_writer/handler.py

왜: upsert_prediction.sql과 columns_map.yaml 계약을 실제로 집행.

Done when: 이벤트({"bucket","key"}) 또는 직접 payload 둘 다 지원, 트랜잭션/재시도/에러로그 포함.

7) 배치 스크립트·샘플 (로컬/CI 테스트)

jobs/sample_event_writer.json

왜: 람다 적재 테스트용 입력 확정.

Done when: 샘플 S3 경로/로컬 파일 경로 모두 예시 제공.

jobs/run_daily_scoring.sh

왜: CI/로컬에서 하루치 스코어 E2E 실행.

Done when: run_processing → run_scoring → (선택) db_writer 호출 흐름 자동화.

8) 자료/디렉터리 생성 (런타임 I/O 경로)

data/raw/

왜: 로컬 샘플 원천 위치를 고정(테스트 재현성).

Done when: 예시 파일 최소 1개 or README 설명.

data/samples/

왜: 피처/예측 샘플 저장 기본 경로.

Done when: 경로만 있어도 OK(파이프라인이 채움).

9) CI/자동화·메타

.github/workflows/ci.yml

왜: lint/test/build 그리고 (옵션) ECR 빌드까지 자동화.

Done when: PR 트리거로 포맷/테스트/도커 빌드 성공, main 병합 시 이미지 푸시(선택).

Makefile

왜: 동일 명령으로 로컬/CI 일관 실행.

Done when: make fe/train/score/build-dbwriter/deploy-dbwriter 타깃들 동작.

README.md

왜: 팀 합의·실행법·경로 규칙 문서화(재작업 최소화의 마지막 퍼즐).

Done when: 의존성/환경설정/단계별 실행 예/에러 트러블슈팅 포함.