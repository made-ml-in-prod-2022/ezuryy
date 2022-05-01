import pandas as pd
import pickle
import joblib
from sklearn import metrics

from ml_project.read_config import read_training_params


def predict(config_path: str):
    params = read_training_params(config_path)
    target_col = params.features.target_col
    data = pd.read_csv(params.input_data_path)
    answer = data[target_col]
    data = data.drop(columns=[target_col])
    X_test = data.values
    with open(params.output_scaler_path, "rb") as f:
        scaler = joblib.load(f)
    with open(params.output_model_path, "rb") as f:
        model = pickle.load(f)

    X_test = scaler.transform(X_test)
    predict = model.predict(X_test)

    print("Accuracy", metrics.accuracy_score(answer, predict))



