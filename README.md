# MLflow + FastAPI ML Service

FastAPI-сервис с мониторингом и алертингом для ML-модели из MLflow.

## Эндпоинты

- `POST /predict` — предсказание по признакам
- `POST /updateModel` — обновление модели по `run_id`
- `GET /health` — статус сервиса
- `GET /metrics` — метрики Prometheus

## Переменные окружения

- `MLFLOW_TRACKING_URI` — URI MLflow Tracking Server
- `DEFAULT_RUN_ID` — run_id модели для загрузки при старте

## Запуск
```bash
export MLFLOW_TRACKING_URI=http://158.160.2.37:5000/
export DEFAULT_RUN_ID=49dbed45ad1a4e889ab467482facbf00
docker compose up --build
```

Сервис доступен на `http://<ip>:8890`.

## Тесты
```bash
pytest tests/ -v
```

## Мониторинг

- Grafana: `http://158.160.2.37:3000` — дашборд "ML Service Monitoring"
- Evidently: `http://158.160.2.37:8000` — drift отчёты
