# EduRisk AI

학생의 입학 정보, 인구통계 정보, 사회경제 정보, 1/2학기 학업 성과를 기반으로 중도 포기 위험을 예측하고, n8n으로 교수자/조교에게 자동 알림을 보내는 프로젝트입니다.

현재 버전은 합성 데이터가 아니라 UCI Machine Learning Repository의 실제 공개 데이터셋 `Predict Students' Dropout and Academic Success`를 사용합니다. 이 데이터셋은 고등교육 기관 학생 4,424명의 학업 경로, 인구통계, 사회경제 정보, 1/2학기 학업 성과를 포함하며, 라이선스는 CC BY 4.0입니다.

예측 라벨은 아래와 같습니다.

- `0`: 정상 수강 가능성 높음
- `1`: 중도 포기 위험 높음

## 미션별 충족 요약

| 미션 | 구현 내용 | 확인 파일/API |
| --- | --- | --- |
| 미션 1 | UCI 실제 데이터 `student_dropout_real.csv` 로드, 범주형 One-Hot Encoding, 수치형 StandardScaler 정규화 | `scripts/train_model.py`, `/preprocessing-info` |
| 미션 2 | TensorFlow Keras DNN 이진 분류 모델과 9개 모델 확률 평균 앙상블 학습 및 저장 | `scripts/train_model.py`, `scripts/train_advanced_model.py`, `/model-info`, `model/ensemble/` |
| 미션 3 | n8n Webhook 기반 예측 서비스, 위험도 분기, 알림/기록 자동화 | `n8n/edurisk_workflow.json`, `/service-info` |

난이도와 아이디어 점수를 높이기 위해 단순 예측을 넘어서 9개 TensorFlow 모델 앙상블, 위험 등급 4단계, 설명 가능한 위험 요인 점수, 맞춤 개입 계획, 배치 예측, n8n 등급별 자동 분기까지 포함했습니다.

## 프로젝트 구조

```text
EduRiskAI/
├── data/
│   ├── download_real_student_dropout_data.py
│   ├── generate_student_dropout_data.py
│   ├── student_dropout_real.csv
│   ├── student_dropout_real_source.json
│   └── student_dropout.csv
├── notebooks/
│   └── edurisk_tensorflow_model.ipynb
├── scripts/
│   ├── make_docx.py
│   ├── train_model.py
│   └── train_advanced_model.py
├── api/
│   ├── main.py
│   ├── sample_request.json
│   └── sample_batch_request.json
├── model/
│   ├── ensemble/
│   └── README.md
├── n8n/
│   ├── edurisk_workflow.json
│   ├── edurisk_workflow_local_test.json
│   └── edurisk_workflow_advanced_local.json
├── report/
│   ├── 과제보고서.md
│   └── 과제보고서.docx
└── requirements.txt
```

## 실행 방법

Windows PowerShell 기준입니다.

```powershell
cd EduRiskAI
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

TensorFlow Windows 패키지는 Python 3.14에서 설치가 어려울 수 있으므로 Python 3.13 환경 사용을 권장합니다.

실제 UCI 데이터를 다시 다운로드하려면 아래 명령을 실행합니다.

```powershell
python data\download_real_student_dropout_data.py
```

기존 합성 샘플 데이터를 다시 생성하려면 아래 명령을 실행합니다. 현재 기본 학습에는 사용하지 않습니다.

```powershell
python data\generate_student_dropout_data.py
```

TensorFlow 모델을 학습하고 모델 파일을 저장합니다.

```powershell
python scripts\train_model.py
```

학습이 끝나면 `model/edurisk_model.keras`, `model/preprocessor.joblib`, `model/metadata.json`이 생성됩니다.

최종 제출용 고급 모델은 아래 명령으로 학습한 TensorFlow 확률 앙상블입니다. 서로 다른 seed로 학습한 regularized DNN 9개의 예측 확률을 평균하여 단일 모델보다 안정적인 결과를 냅니다.

```powershell
python scripts\train_advanced_model.py --ensemble-size 9 --preset regularized --threshold 0.5
```

앙상블 학습이 끝나면 `model/ensemble/edurisk_regularized_01.keras`부터 `model/ensemble/edurisk_regularized_09.keras`까지 저장되고, FastAPI는 `metadata.json`의 `ensemble_model_paths`를 읽어 자동으로 9개 모델 평균 예측을 수행합니다.

추가 실험 옵션입니다.

```powershell
python scripts\train_model.py --auto-threshold
python scripts\train_model.py --use-engineered-features
python scripts\train_model.py --class-weight
python scripts\train_advanced_model.py --ensemble-size 5 --preset compact
python scripts\train_advanced_model.py --ensemble-size 5 --preset wide
```

- `--auto-threshold`: validation set에서 F1-score 기준 threshold를 자동 탐색합니다.
- `--use-engineered-features`: 1/2학기 승인율, 학업 모멘텀, 재정 위험 점수 등 파생 특성을 추가합니다.
- `--class-weight`: 클래스 불균형을 보정합니다.

## FastAPI 서버 실행

```powershell
uvicorn api.main:app --reload --port 8000
```

브라우저에서 아래 주소를 확인합니다.

- API 문서: `http://127.0.0.1:8000/docs`
- 상태 확인: `http://127.0.0.1:8000/health`
- 전처리 설명: `http://127.0.0.1:8000/preprocessing-info`
- 모델 설명: `http://127.0.0.1:8000/model-info`
- n8n 서비스 설명: `http://127.0.0.1:8000/service-info`

단일 학생 예측 요청 예시입니다.

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/predict `
  -Method Post `
  -ContentType "application/json" `
  -InFile api\sample_request.json
```

응답 예시는 아래 형태입니다.

```json
{
  "dropout_probability": 0.994,
  "dropout_probability_percent": 99.4,
  "prediction": "중도 포기 위험",
  "label": 1,
  "risk_level": "Critical",
  "threshold": 0.5,
  "top_risk_factors": [
    {"label": "2학기 이수 승인율", "contribution": 20.0},
    {"label": "2학기 평균 성적", "contribution": 15.0}
  ],
  "recommended_actions": ["개별 상담 진행", "2학기 미이수 과목 점검"]
}
```

배치 예측 요청 예시입니다.

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/predict-batch `
  -Method Post `
  -ContentType "application/json" `
  -InFile api\sample_batch_request.json
```

확장 API는 아래 정보를 함께 반환합니다.

- `risk_level`: `Low`, `Medium`, `High`, `Critical`
- `risk_score`: 모델 확률을 보완하는 규칙 기반 설명 점수
- `risk_score_breakdown`: 등록금 납부 상태, 채무 여부, 1/2학기 이수 승인율과 성적, 입학 성적, 이전 학력 성적 등 위험 기여도
- `top_risk_factors`: 가장 큰 위험 요인 3개
- `intervention_plan`: 담당자, 조치 기한, 맞춤 개입 행동 목록
- `/predict-batch`: 여러 학생을 한 번에 예측하고 위험 등급별 요약을 반환

## n8n 연결

`n8n/edurisk_workflow.json`을 n8n에 Import 합니다.

계정 Credential 없이 FastAPI 연동만 먼저 확인하려면 `n8n/edurisk_workflow_local_test.json`을 Import 합니다.

위험 등급별 자동 분기까지 확인하려면 `n8n/edurisk_workflow_advanced_local.json`을 Import 합니다.

- n8n을 로컬에서 직접 실행하면 HTTP Request URL을 `http://127.0.0.1:8000/predict`로 둡니다.
- n8n을 Docker로 실행하면 HTTP Request URL을 `http://host.docker.internal:8000/predict`로 바꿉니다.
- Slack, Gmail, Google Sheets 노드는 각 서비스의 Credential을 연결한 뒤 사용합니다.
- AI Agent 노드는 위험 학생의 사유와 권장 조치를 자연어 알림 문장으로 정리하는 역할입니다.
- 고급 로컬 워크플로우는 `Critical`, `High`, `Medium`, `Low` 위험 등급에 따라 서로 다른 자동화 경로를 응답으로 보여줍니다.

Webhook 입력 예시는 `api/sample_request.json`과 동일합니다. UCI 실제 데이터 컬럼명에는 공백과 괄호가 포함되므로, n8n 워크플로우는 Webhook body를 그대로 FastAPI에 전달하도록 구성했습니다.

## 전처리 요약

- 데이터 출처: UCI Machine Learning Repository, `Predict Students' Dropout and Academic Success`
- 원본 파일: `data/student_dropout_real.csv`
- 원본 타깃: `Target` = `Dropout`, `Enrolled`, `Graduate`
- 이진 분류 라벨: `Dropout`은 1, `Enrolled`와 `Graduate`는 0
- `Marital status`, `Application mode`, `Course`, `Previous qualification`, `Gender` 등 범주형/코드형 특성은 One-Hot Encoding
- `Admission grade`, `Age at enrollment`, 1/2학기 이수 과목 수와 성적, `GDP` 등 수치형 특성은 StandardScaler 정규화

전처리는 `ColumnTransformer`로 구성하여 학습과 예측에서 동일한 변환 객체를 사용합니다. 학습 후 `model/preprocessor.joblib`로 저장되므로 FastAPI 예측 서버에서도 같은 One-Hot Encoding과 정규화 규칙이 적용됩니다.

## 모델 요약

- TensorFlow Keras regularized DNN 확률 앙상블
- 앙상블 구성: 서로 다른 seed로 학습한 9개 TensorFlow 모델의 예측 확률 평균
- 은닉층 preset: Dense 96, Dense 48, Dense 24 + BatchNormalization + Dropout
- 출력층: Dense 1 + sigmoid
- 손실 함수: `binary_crossentropy`
- Optimizer: Adam
- 위험 판단 기준: 기본 threshold `0.5`
- 추가 옵션: `python scripts\train_model.py --auto-threshold` 실행 시 validation F1 기준 threshold 자동 탐색
- 최종 앙상블 성능: Accuracy `0.8972`, Precision `0.9038`, Recall `0.7606`, F1-score `0.8260`
- 이전 단일 모델 성능: Accuracy `0.8893`, Precision `0.8633`, Recall `0.7782`, F1-score `0.8185`
- 고급 실험 옵션: compact/wide/deep preset, 파생 특성 엔지니어링, class weight 학습을 선택적으로 실행 가능

추가 실험 비교입니다.

| 실험 | Accuracy | Precision | Recall | F1-score | 판단 |
| --- | ---: | ---: | ---: | ---: | --- |
| 단일 DNN threshold 0.5 | 0.8893 | 0.8633 | 0.7782 | 0.8185 | 기준 모델 |
| 5개 regularized 앙상블 | 0.8938 | 0.8893 | 0.7641 | 0.8220 | 개선 |
| 7개 regularized 앙상블 | 0.8949 | 0.8996 | 0.7570 | 0.8222 | 추가 개선 |
| 9개 regularized 앙상블 | 0.8972 | 0.9038 | 0.7606 | 0.8260 | 최종 채택 |
| 9개 앙상블 가중 평균 | 0.8972 | 0.9038 | 0.7606 | 0.8260 | 단순 평균과 동일하여 미채택 |

## 실제 데이터 출처

- UCI Machine Learning Repository: `Predict Students' Dropout and Academic Success`
- 공식 페이지: `https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success`
- DOI: `10.24432/C5MC89`
- 라이선스: `CC BY 4.0`
