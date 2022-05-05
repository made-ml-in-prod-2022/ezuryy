import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from enities import read_training_params, fix_path, fix_config
from features import preprocess_train_data


def train_model(config_path: str):
    config_path = fix_path(config_path)
    params = fix_config(read_training_params(config_path))
    target_col = params.features.target_col
    data = pd.read_csv(params.input_train_data_path)
    y = data[target_col].values
    data = data.drop(columns=[target_col])
    X = preprocess_train_data(data, params)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=0, stratify=y
    )

    model = LogisticRegression(random_state=0, penalty='l2', C=0.9).fit(X_train, y_train)
    with open(params.model_path, "wb") as f:
        pickle.dump(model, f)

    predict = model.predict(X_val)
    print("Accuracy: ", metrics.accuracy_score(y_val, predict))
    print("ROC AUC score: ", metrics.roc_auc_score(y_val, predict))
    print("F1-score: ", metrics.f1_score(y_val, predict))


if __name__ == '__main__':
    train_model('configs/train_config.yaml')
