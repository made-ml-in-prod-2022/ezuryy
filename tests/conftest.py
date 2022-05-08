import pytest
from typing import List
import numpy as np
import pandas as pd

from ml_project import Features, TrainingParams, SplittingParams


@pytest.fixture(scope="session")
def target_col():
    return "condition"


@pytest.fixture(scope="session")
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "thal"
    ]


@pytest.fixture(scope="session")
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
        "ca"
    ]


@pytest.fixture()
def params(categorical_features, numerical_features, target_col, tmpdir) -> TrainingParams:
    np.random.seed(42)
    rows_number = 100
    data = pd.DataFrame()

    for col in categorical_features:
        values = [0, 1, 2, 3]
        column = np.random.choice(values, rows_number)
        data[col] = column

    for col in numerical_features:
        column = np.random.randint(200., size=rows_number)
        data[col] = column

    test_filename = tmpdir.mkdir("tmpdir").join("test_data.csv")
    train_filename = tmpdir.join("tmpdir/train_data.csv")
    data.to_csv(test_filename, index_label=False)
    data[target_col] = np.random.choice([0, 1], rows_number)
    data.to_csv(train_filename, index_label=False)

    features = Features(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col
    )
    params = TrainingParams(
        input_train_data_path=train_filename,
        input_test_data_path=test_filename,
        scaler_path="tmpdir/scaler.joblib",
        model_path="tmpdir/model.pkl",
        predict_path="tmpdir/predict.csv",
        model_type="LogisticRegression",
        features=features,
        splitting_params=SplittingParams(val_size=0.1, random_state=42, stratify=True)
    )
    return params
