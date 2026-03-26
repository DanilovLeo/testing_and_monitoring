import threading
from typing import NamedTuple

import mlflow
from sklearn.pipeline import Pipeline

from ml_service.mlflow_utils import load_model


class ModelData(NamedTuple):
    model: object | None
    run_id: str | None
    features: list[str] | None


class Model:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.data = ModelData(model=None, run_id=None, features=None)

    def get(self) -> ModelData:
        with self.lock:
            return self.data

    def set(self, run_id: str) -> None:
        model = load_model(run_id=run_id)
        # Берём фичи из параметров MLflow run
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        features_str = run.data.params.get('features', '[]')
        # Параметр хранится как строка "['race', 'sex', ...]" — парсим
        import ast
        features = ast.literal_eval(features_str)
        with self.lock:
            self.data = ModelData(model=model, run_id=run_id, features=features)

    @property
    def features(self) -> list[str]:
        return self.data.features
