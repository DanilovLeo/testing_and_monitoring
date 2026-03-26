import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from ml_service.schemas import PredictRequest

FEATURE_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education.num',
    'marital.status', 'occupation', 'relationship', 'race', 'sex',
    'capital.gain', 'capital.loss', 'hours.per.week', 'native.country',
]

ALL_CAT_FEATURES = [
    'workclass', 'education', 'marital.status', 'occupation',
    'relationship', 'race', 'sex', 'native.country',
]


def to_dataframe(req: PredictRequest, needed_columns: list[str] = None) -> pd.DataFrame:
    columns = [
        column for column in needed_columns if column in FEATURE_COLUMNS
    ] if needed_columns is not None else FEATURE_COLUMNS

    row = {col: getattr(req, col.replace('.', '_')) for col in columns}

    missing = [col for col, val in row.items() if val is None]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    df = pd.DataFrame([row])

    # Воспроизводим предобработку из process_data.py
    cat_features = [f for f in columns if f in ALL_CAT_FEATURES]
    num_features = [f for f in columns if f not in ALL_CAT_FEATURES]

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    num_data = df[num_features].values if num_features else np.empty((1, 0))
    cat_data = encoder.fit_transform(df[cat_features]) if cat_features else np.empty((1, 0))
    
    X = np.hstack([num_data, cat_data])
    return pd.DataFrame(X)
