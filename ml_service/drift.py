import pandas as pd
import logging
import traceback
import os

logger = logging.getLogger(__name__)

EVIDENTLY_URL = "http://158.160.2.37:8000"
PROJECT_ID = "019d061f-cc08-7b5e-b932-d792a1f258e2"
REPORTS_DIR = "/tmp/drift_reports"

_reference_data: pd.DataFrame | None = None
_current_buffer: list[dict] = []
_reference_buffer: list[dict] = []
_report_counter = 0

REFERENCE_SIZE = 20
BUFFER_SIZE = 10
FEATURES = ['race', 'sex', 'native.country', 'occupation', 'education', 'capital.gain']

os.makedirs(REPORTS_DIR, exist_ok=True)


def add_to_buffer(row: dict) -> None:
    global _current_buffer, _reference_buffer, _reference_data

    filtered = {k: v for k, v in row.items() if k in FEATURES}

    if _reference_data is None:
        _reference_buffer.append(filtered)
        if len(_reference_buffer) >= REFERENCE_SIZE:
            _reference_data = pd.DataFrame(_reference_buffer)
            logger.info(f"Reference data collected: {len(_reference_data)} rows")
        return

    _current_buffer.append(filtered)
    if len(_current_buffer) >= BUFFER_SIZE:
        _flush_buffer()


def set_reference(df: pd.DataFrame) -> None:
    global _reference_data
    if len(df) > 0:
        _reference_data = df.copy()


def _flush_buffer() -> None:
    global _current_buffer, _reference_data, _report_counter

    if _reference_data is None or len(_current_buffer) < 5:
        _current_buffer = []
        return

    try:
        from evidently.report import Report
        from evidently.metrics import ColumnDriftMetric

        current_df = pd.DataFrame(_current_buffer)
        _current_buffer = []

        logger.info(f"Running drift: ref={len(_reference_data)}, current={len(current_df)}")

        metrics = [ColumnDriftMetric(column_name=col) for col in FEATURES]
        report = Report(metrics=metrics)
        report.run(reference_data=_reference_data, current_data=current_df)

        # Сохраняем HTML отчёт локально
        _report_counter += 1
        report_path = f"{REPORTS_DIR}/drift_report_{_report_counter}.html"
        report.save_html(report_path)
        logger.info(f"Drift report saved: {report_path}")

        # Пробуем отправить в Evidently через SDK
        try:
            from evidently.ui.workspace import RemoteWorkspace
            ws = RemoteWorkspace(EVIDENTLY_URL)
            ws.add_snapshot(PROJECT_ID, report.to_snapshot())
            logger.info("Drift report sent to Evidently successfully")
        except Exception as sdk_e:
            logger.warning(f"Could not send to Evidently UI (saved locally instead): {sdk_e}")

    except Exception as e:
        logger.error(f"Drift monitoring error: {e}\n{traceback.format_exc()}")
        _current_buffer = []
