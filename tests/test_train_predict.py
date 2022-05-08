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


# def test_split_train_val_data(params: TrainingParams):
#     print("Im here!")
#     data = pd.read_csv(params.input_train_data_path)
#     target = data[params.features.target_col]
#     data.drop(columns=[params.features.target_col], inplace=True)
#     train_data, val_data, train_target, val_target = split_train_val_data(data, target, params)
#
#     expected_val_size = int(params.splitting_params.val_size * data.shape[0])
#     expected_train_size = int((1 - params.splitting_params.val_size) * data.shape[0])
#
#     print("val_data", len(val_data), expected_val_size)
#     assert len(val_data) == expected_val_size
#     print("train_data", len(train_data), expected_train_size)
#     assert len(train_data) == expected_train_size
#     print("val_target", len(val_target), expected_val_size)
#     assert len(val_target) == expected_val_size
#     print("train_target", len(train_target), expected_train_size)
#     assert len(train_target) == expected_train_size
#     print("Proportion", len(val_target == 1) / len(val_target), len(train_target == 1) / len(train_target))
#     assert len(val_target == 1) / len(val_target) == len(train_target == 1) / len(train_target)

