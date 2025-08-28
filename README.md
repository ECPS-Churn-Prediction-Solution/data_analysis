ecps-churn/
├─ docs/                                   ⏳ 아키텍처/ADR/운영가이드/쿼리모음
│  ├─ architecture.md
│  ├─ adr/                                 ⏳ 의사결정 기록(ADR)
│  └─ operations.md                        ⏳ 운영/배포/롤백 가이드
├─ config/
│  ├─ risk_cuts.json                       ✅ risk_band 컷포인트(예: {"VH":0.9,"H":0.75,"M":0.5})
│  ├─ columns_map.yaml                     ✅ 산출→DB 컬럼 매핑
│  ├─ features.yaml                        ✅ 피처 스키마/검증 규칙
│  ├─ model_lgbm.yaml                      ✅ LGBM 하이퍼파라미터/seed/early_stopping
│  ├─ data_split.yaml                      ✅ train/valid/test 컷오프/윈도우
│  ├─ env/
│  │  ├─ dev.env.example                   ✅ 로컬/개발 환경변수 템플릿
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
│  ├─ analytics.sql                        ✅ analytics 스키마/테이블/인덱스
│  ├─ mart.sql                             ⏳ 마트/뷰 정의
│  ├─ upsert_prediction.sql                ✅ INSERT ... ON CONFLICT ...
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
