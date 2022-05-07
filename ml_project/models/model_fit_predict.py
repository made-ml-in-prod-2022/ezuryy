import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# from enities import read_training_params, fix_path, fix_config
# from features import preprocess_test_data


# def train_model(
#     features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
# ) -> SklearnRegressionModel:
#     if train_params.model_type == "RandomForestRegressor":
#         model = RandomForestRegressor(
#             n_estimators=100, random_state=train_params.random_state
#         )
#     elif train_params.model_type == "LinearRegression":
#         model = LinearRegression()
#     else:
#         raise NotImplementedError()
#     model.fit(features, target)
#     return model
#
#
# def predict_model(
#     model: Pipeline, features: pd.DataFrame, use_log_trick: bool = True
# ) -> np.ndarray:
#     predicts = model.predict(features)
#     if use_log_trick:
#         predicts = np.exp(predicts)
#     return predicts


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
