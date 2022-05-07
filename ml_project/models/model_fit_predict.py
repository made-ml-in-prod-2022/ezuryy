import pandas as pd
import numpy as np
import pickle
import logging
from typing import Union
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

from enities import TrainingParams

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def get_model(params: TrainingParams) -> Union[LogisticRegression, GaussianNB]:
    logger.info(f'Model type:{params.model_type}')
    if params.model_type == 'LogisticRegression':
        model = LogisticRegression(random_state=0, penalty='l2', C=0.9)
    elif params.model_type == 'GaussianNB':
        model = GaussianNB()
    else:
        logger.warning('No such model_type! Use LogisticRegression')
        model = LogisticRegression(random_state=0, penalty='l2', C=0.9)
    return model


def evaluate_model(predict: np.ndarray, target: np.ndarray):
    print("Accuracy: ", metrics.accuracy_score(target, predict))
    print("ROC AUC score: ", metrics.roc_auc_score(target, predict))
    print("F1-score: ", metrics.f1_score(target, predict))


def save_model(model: object, path: str):
    logger.info(f'Saving model to {path}')
    with open(path, "wb") as f:
        pickle.dump(model, f)


def open_model(path: str) -> LogisticRegression:
    logger.info(f'Opening model {path}')
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def save_predict(target: str, predict: np.ndarray, path: str):
    logger.info(f'Saving predict to {path}')
    predict_df = pd.DataFrame({target: predict})
    predict_df.to_csv(path, index=False)
