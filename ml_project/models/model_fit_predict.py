import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def evaluate_model(predict: np.ndarray, target: np.ndarray):

    print("Accuracy: ", metrics.accuracy_score(target, predict))
    print("ROC AUC score: ", metrics.roc_auc_score(target, predict))
    print("F1-score: ", metrics.f1_score(target, predict))


def save_model(model: object, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def open_model(path: str) -> LogisticRegression:
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def save_predict(target: str, predict: np.ndarray, path: str):
    predict_df = pd.DataFrame({target: predict})
    predict_df.to_csv(path, index=False)
