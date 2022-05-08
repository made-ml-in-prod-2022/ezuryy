import os
# from py._path.local import LocalPath
# import pandas as pd

from ml_project import (
    TrainingParams,
    run_train_pipeline,
    run_test_pipeline,
    # split_train_val_data,
)


def test_train(params: TrainingParams):
    metrics = run_train_pipeline(params)
    assert metrics["Accuracy"] > 0
    assert metrics["ROC AUC score"] > 0
    assert metrics["F1-score"] > 0
    assert os.path.exists(params.model_path)
    assert os.path.exists(params.scaler_path)


def test_predict(params: TrainingParams):
    metrics = run_test_pipeline(params)
    assert metrics["Accuracy"] > 0
    assert metrics["ROC AUC score"] > 0
    assert metrics["F1-score"] > 0
    assert os.path.exists(params.predict_path)
