import pandas as pd
import numpy as np
import pickle
import logging
from typing import Union, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

from ml_project.entity import TrainingParams
from ml_project.data import split_train_val_data
from ml_project.features import (
    preprocess_train_data,
    preprocess_test_data,
    extract_target,
)

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def get_model(params: TrainingParams) -> Union[LogisticRegression, GaussianNB]:
    logger.info(f"Model type:{params.model_type}")
    if params.model_type == "LogisticRegression":
        model = LogisticRegression(random_state=0, penalty="l2", C=0.9)
    elif params.model_type == "GaussianNB":
        model = GaussianNB()
    else:
        logger.warning("No such model_type! Use LogisticRegression")
        model = LogisticRegression(random_state=0, penalty="l2", C=0.9)
    return model


def evaluate_model(predict: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    result_metrics = {
        "Accuracy": metrics.accuracy_score(target, predict),
        "ROC AUC score": metrics.roc_auc_score(target, predict),
        "F1-score": metrics.f1_score(target, predict),
    }
    for key in result_metrics:
        print(key, " : ", result_metrics[key])
    return result_metrics


def save_model(model: object, path: str):
    logger.info(f"Saving model to {path}")
    with open(path, "wb") as f:
        pickle.dump(model, f)


def open_model(path: str) -> LogisticRegression:
    logger.info(f"Opening model {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def save_predict(target: str, predict: np.ndarray, path: str):
    logger.info(f"Saving predict to {path}")
    predict_df = pd.DataFrame({target: predict})
    predict_df.to_csv(path, index=False)


def run_train_pipeline(params: TrainingParams) -> Dict[str, float]:
    data = pd.read_csv(params.input_train_data_path)
    data, target = extract_target(data, params)

    full_data = preprocess_train_data(data, params)

    train_data, val_data, train_target, val_target = split_train_val_data(
        full_data, target, params
    )

    model = get_model(params)
    model.fit(train_data, train_target)

    save_model(model, params.model_path)

    predict = model.predict(val_data)
    result_metrics = evaluate_model(predict, val_target)
    return result_metrics


def run_test_pipeline(params: TrainingParams):
    target_col = params.features.target_col
    data = pd.read_csv(params.input_test_data_path)

    test_data = preprocess_test_data(data, params)

    model = open_model(params.model_path)

    predict = model.predict(test_data)

    answer = pd.read_csv(params.input_train_data_path)[target_col]

    result_metrics = evaluate_model(predict, answer)

    save_predict(target_col, predict, params.predict_path)

    return result_metrics
