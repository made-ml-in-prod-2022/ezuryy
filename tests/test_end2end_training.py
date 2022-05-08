import os

from ml_project import (
    TrainingParams,
    run_train_pipeline
)


def test_train_e2e(params: TrainingParams):
    metrics = run_train_pipeline(params)
    assert metrics["Accuracy"] > 0
    assert metrics["ROC AUC score"] > 0
    assert metrics["F1-score"] > 0
    assert os.path.exists(params.model_path)
    assert os.path.exists(params.scaler_path)
