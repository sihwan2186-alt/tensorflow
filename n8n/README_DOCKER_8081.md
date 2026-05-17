# n8n Docker 8081 실행 안내

## 1. FastAPI 먼저 실행

프로젝트 루트(`EduRiskAI`)에서 예측 API를 먼저 실행합니다.

```powershell
uvicorn api.main:app --reload --port 8001
```

브라우저에서 아래 주소가 열리면 준비 완료입니다.

```text
http://127.0.0.1:8001/health
```

## 2. n8n Docker 새로 실행

`n8n` 폴더에서 실행합니다.

완전히 새로 시작해야 하면 기존 n8n 컨테이너와 볼륨을 먼저 삭제합니다.

```powershell
cd n8n
docker compose down -v
```

그 다음 다시 실행합니다.

```powershell
cd n8n
docker compose up -d
```

n8n 접속 주소:

```text
http://localhost:8081
```

## 3. JSON Import

새로 설치한 n8n 화면에서 Import from File을 선택한 뒤 아래 파일을 올립니다.

```text
n8n/import/edurisk_workflow_docker_8081.json
```

이 JSON은 Docker n8n 기준으로 FastAPI 호출 주소가 아래처럼 설정되어 있습니다.

```text
http://host.docker.internal:8001/predict
```

## 4. Webhook 주소

n8n에서 워크플로를 열고 Webhook 노드의 Listen for test event를 누른 뒤 호출합니다.

```text
http://localhost:8081/webhook-test/edurisk-advanced-docker-8081
```

UI에서 직접 Import한 뒤 워크플로를 Active로 켠 경우에는 아래 운영 주소를 사용합니다.

```text
http://localhost:8081/webhook/edurisk-advanced-docker-8081
```

CLI로 Import한 현재 실행 환경에서는 n8n 2.x가 운영 Webhook을 아래 경로로 등록했습니다.

```text
http://localhost:8081/webhook/edurisk-docker-8081/webhook%2520trigger/edurisk-advanced-docker-8081
```

## 5. CLI로 Import할 때

UI 대신 컨테이너 안에서 바로 Import하려면 n8n 실행 후 아래 명령을 사용할 수 있습니다.

```powershell
docker exec -it edurisk-n8n n8n import:workflow --input=/files/import/edurisk_workflow_docker_8081.json
```
