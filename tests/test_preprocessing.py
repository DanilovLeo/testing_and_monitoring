import pytest
from ml_service.features import to_dataframe
from ml_service.schemas import PredictRequest


VALID_REQUEST = PredictRequest(
    race="White",
    sex="Male",
    **{"native.country": "United-States"},
    occupation="Prof-specialty",
    education="Bachelors",
    **{"capital.gain": 0},
)

NEEDED_COLUMNS = ['race', 'sex', 'native.country', 'occupation', 'education', 'capital.gain']


def test_to_dataframe_valid():
    df = to_dataframe(VALID_REQUEST, needed_columns=NEEDED_COLUMNS)
    assert df.shape == (1, 6)


def test_to_dataframe_missing_feature():
    request = PredictRequest(race="White", sex="Male")
    with pytest.raises(ValueError, match="Missing required features"):
        to_dataframe(request, needed_columns=NEEDED_COLUMNS)


def test_to_dataframe_ignores_extra_columns():
    needed = ['race', 'sex']
    df = to_dataframe(VALID_REQUEST, needed_columns=needed)
    assert df.shape == (1, 2)
