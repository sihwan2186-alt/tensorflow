# Model Artifacts

학습 스크립트를 실행하면 아래 파일이 생성됩니다.

- `edurisk_model.keras`: 단일 모델 호환성을 위한 첫 번째 TensorFlow DNN 모델
- `ensemble/edurisk_regularized_01.keras` ~ `ensemble/edurisk_regularized_09.keras`: 최종 제출용 9개 TensorFlow 앙상블 모델
- `preprocessor.joblib`: One-Hot Encoding과 StandardScaler가 포함된 전처리 파이프라인
- `metadata.json`: threshold, feature 목록, 앙상블 경로, 성능 지표

FastAPI는 `metadata.json`의 `ensemble_model_paths`가 있으면 여러 모델의 예측 확률을 평균하여 최종 dropout probability를 계산합니다.
