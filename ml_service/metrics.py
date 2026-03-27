from prometheus_client import Counter, Histogram, Gauge, Info, Summary

# Технические метрики
http_requests_total = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

http_request_duration_summary = Summary(
    'http_request_duration_summary_seconds',
    'HTTP request duration summary',
    ['method', 'endpoint']
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'Number of HTTP requests in progress',
    ['method', 'endpoint']
)

# Метрики данных
feature_preprocessing_duration_seconds = Histogram(
    'feature_preprocessing_duration_seconds',
    'Feature preprocessing duration in seconds',
    ['model_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25]
)

feature_preprocessing_summary = Summary(
    'feature_preprocessing_summary_seconds',
    'Feature preprocessing duration summary',
    ['model_type']
)

missing_features_total = Counter(
    'missing_features_total',
    'Total number of requests with missing features'
)

feature_value = Gauge(
    'feature_value',
    'Current value of numeric features',
    ['feature_name']
)

# Метрики модели
model_inference_duration_seconds = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration in seconds',
    ['model_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25]
)

model_inference_summary = Summary(
    'model_inference_summary_seconds',
    'Model inference duration summary',
    ['model_type']
)

model_prediction_total = Counter(
    'model_prediction_total',
    'Total number of model predictions',
    ['predicted_class']
)

model_prediction_probability = Histogram(
    'model_prediction_probability',
    'Distribution of model prediction probabilities',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Метрики информации о модели
model_info = Info(
    'model',
    'Information about the current model'
)

model_last_update_timestamp = Gauge(
    'model_last_update_timestamp',
    'Unix timestamp of last model update'
)

model_update_total = Counter(
    'model_update_total',
    'Total number of model updates'
)
