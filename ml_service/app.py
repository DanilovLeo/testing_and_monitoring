import time
import asyncio
from typing import Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import logging
from mlflow.exceptions import MlflowException
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from ml_service import config
from ml_service.features import to_dataframe
from ml_service.mlflow_utils import configure_mlflow
from ml_service.model import Model
from ml_service.schemas import (
    PredictRequest,
    PredictResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)
from ml_service import metrics

MODEL = Model()
logger = logging.getLogger(__name__)


def get_model_type() -> str:
    model_data = MODEL.get()
    if model_data.model is None:
        return 'unknown'
    return type(model_data.model).__name__


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_mlflow()
    try:
        run_id = config.default_run_id()
        MODEL.set(run_id=run_id)
        logger.info(f"Model loaded successfully, run_id={run_id}")
        # Обновляем метрики информации о модели
        metrics.model_info.info({
            'run_id': run_id,
            'model_type': get_model_type(),
            'feature_names': ','.join(MODEL.features or [])
        })
        metrics.model_last_update_timestamp.set(time.time())
    except Exception as e:
        logger.warning(f"Could not load model on startup: {e}. Service will start without model.")
    yield

def create_app() -> FastAPI:
    app = FastAPI(title='MLflow FastAPI service', version='1.0.0', lifespan=lifespan)

    @app.get('/metrics')
    def prometheus_metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get('/health')
    def health() -> dict[str, Any]:
        model_state = MODEL.get()
        run_id = model_state.run_id
        return {'status': 'ok', 'run_id': run_id}

    @app.post('/predict', response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        start_time = time.time()
        metrics.http_requests_in_progress.labels(method='POST', endpoint='/predict').inc()

        try:
            model_data = MODEL.get()
            model = model_data.model
            model_type = get_model_type()

            if model is None:
                metrics.http_requests_total.labels(
                    method='POST', endpoint='/predict', status_code='503'
                ).inc()
                raise HTTPException(status_code=503, detail='Model is not loaded yet')

            # Предобработка
            preprocess_start = time.time()
            try:
                df = to_dataframe(request, needed_columns=MODEL.features)
            except ValueError as e:
                metrics.missing_features_total.inc()
                metrics.http_requests_total.labels(
                    method='POST', endpoint='/predict', status_code='422'
                ).inc()
                raise HTTPException(status_code=422, detail=str(e))
            finally:
                preprocess_duration = time.time() - preprocess_start
                metrics.feature_preprocessing_duration_seconds.labels(
                    model_type=model_type
                ).observe(preprocess_duration)
                metrics.feature_preprocessing_summary.labels(
                    model_type=model_type
                ).observe(preprocess_duration)

            # Логируем числовые фичи
            numeric_features = ['capital.gain', 'capital.loss', 'age', 'fnlwgt',
                                'hours.per.week', 'education.num']
            for feat in numeric_features:
                val = getattr(request, feat.replace('.', '_'), None)
                if val is not None:
                    metrics.feature_value.labels(feature_name=feat).set(float(val))

            # Инференс
            inference_start = time.time()
            try:
                probability = model.predict_proba(df)[0][1]
            except Exception as e:
                metrics.http_requests_total.labels(
                    method='POST', endpoint='/predict', status_code='500'
                ).inc()
                raise HTTPException(status_code=500, detail=f"Model inference error: {e}")
            finally:
                inference_duration = time.time() - inference_start
                metrics.model_inference_duration_seconds.labels(
                    model_type=model_type
                ).observe(inference_duration)
                metrics.model_inference_summary.labels(
                    model_type=model_type
                ).observe(inference_duration)

            prediction = int(probability >= 0.5)

            # Метрики предсказания
            metrics.model_prediction_total.labels(
                predicted_class=str(prediction)
            ).inc()
            metrics.model_prediction_probability.observe(probability)

            metrics.http_requests_total.labels(
                method='POST', endpoint='/predict', status_code='200'
            ).inc()

            return PredictResponse(prediction=prediction, probability=float(probability))

        finally:
            duration = time.time() - start_time
            metrics.http_request_duration_seconds.labels(
                method='POST', endpoint='/predict'
            ).observe(duration)
            metrics.http_request_duration_summary.labels(
                method='POST', endpoint='/predict'
            ).observe(duration)
            metrics.http_requests_in_progress.labels(
                method='POST', endpoint='/predict'
            ).dec()

    @app.post('/updateModel', response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
        run_id = req.run_id
        try:
            MODEL.set(run_id=run_id)
            metrics.model_update_total.inc()
            metrics.model_last_update_timestamp.set(time.time())
            metrics.model_info.info({
                'run_id': run_id,
                'model_type': get_model_type(),
                'feature_names': ','.join(MODEL.features or [])
            })
            metrics.http_requests_total.labels(
                method='POST', endpoint='/updateModel', status_code='200'
            ).inc()
        except MlflowException as e:
            metrics.http_requests_total.labels(
                method='POST', endpoint='/updateModel', status_code='404'
            ).inc()
            raise HTTPException(status_code=404, detail=f"Run ID '{run_id}' not found in MLflow: {e}")
        except Exception as e:
            metrics.http_requests_total.labels(
                method='POST', endpoint='/updateModel', status_code='503'
            ).inc()
            raise HTTPException(status_code=503, detail=f"MLflow is unavailable: {e}")
        return UpdateModelResponse(run_id=run_id)

    return app


app = create_app()
