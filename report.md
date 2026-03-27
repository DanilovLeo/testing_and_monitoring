# Отчёт по дз4: тестирование и мониторинг ML-сервиса

## Задание 1 Обработка ошибок

В `ml_service/app.py` добавлена обработка следующих ситуаций:

1. **Модель не загружена** (`/predict`) — возвращает `503` если модель не инициализирована при старте
2. **Неверные признаки** (`/predict`) — возвращает `422` если в запросе отсутствуют нужные фичи (`ValueError` в `to_dataframe`)
3. **Ошибка инференса** (`/predict`) — возвращает `500` при любом исключении в `model.predict_proba`
4. **Run ID не найден** (`/updateModel`) — возвращает `404` при `MlflowException`
5. **MLflow недоступен** (`/updateModel`) — возвращает `503` при общем исключении подключения
6. **Ошибка предобработки** — счётчик `missing_features_total` инкрементируется, метрика логируется

## Задание 2 Тесты

Тесты находятся в `tests/`:
- `test_preprocessing.py` — тесты предобработки данных (`to_dataframe`)
- `test_handlers.py` — тесты хэндлеров `/predict`, `/updateModel`, `/health`
- `test_integration.py` — интеграционные тесты сервиса

Запуск:
```bash
pytest tests/ -v
```

## Задание 3 Мониторинг (Prometheus + Grafana)

Метрики логируются в `ml_service/metrics.py`, эндпоинт `/metrics` отдаёт данные в формате Prometheus.

Дашборд **"ML Service Monitoring"** в Grafana (`http://158.160.2.37:3000`) содержит 12 панелей, разбитых на 4 группы:

**Технические метрики сервиса:**
- Request Rate (RPS)
- Response Time P95
- Requests In Progress

**Метрики данных:**
- Missing Features Rate
- Feature Value Distribution (capital.gain)
- Preprocessing Duration

**Метрики модели:**
- Model Inference Duration
- Prediction Distribution (class 0 vs 1)
- Prediction Probability Distribution

**Метрики модели и обновлений:**
- Model Info (run_id, model_type, features)
- Model Update Count
- Last Update Timestamp

## Задание 4 Алертинг

Contact point: **tg** (Telegram-бот)

В Grafana настроено 5 алертов в группе **ML Alerts** (папка ML Service Alerts, interval 1m):

| Алерт | Условие |
|---|---|
| High Response Time | P95 latency > 0.5s |
| High Model Inference Time | avg inference > 100ms |
| High Error Rate | 5xx rate > 0.1 rps |
| Model Updated | changes in timestamp > 0 |
| High Missing Features Rate | missing features rate > 0.1 |

Для проверки срабатывания алерта "High Response Time" можно временно снизить порог до 0 — уведомление придёт в Telegram.

## Задание 5 — Drift мониторинг (Evidently)

Реализован в `ml_service/drift.py`.

**Принцип работы:**
- Первые 20 запросов к `/predict` собираются как reference dataset
- Каждые следующие 10 запросов считается `ColumnDriftMetric` по всем 6 фичам модели
- Отчёт сохраняется в `/tmp/drift_reports/drift_report_N.html`
- Делается попытка отправки в Evidently UI (`http://158.160.2.37:8000`, project `019d061f-cc08-7b5e-b932-d792a1f258e2`)

Фичи для мониторинга: `race`, `sex`, `native.country`, `occupation`, `education`, `capital.gain`
