import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from data import split_train_val_data
from enities import read_training_params, fix_path, fix_config
from features import preprocess_train_data, extract_target
from models import save_model, evaluate_model


def train_model(config_path: str):
    config_path = fix_path(config_path)
    params = fix_config(read_training_params(config_path))

    data = pd.read_csv(params.input_train_data_path)
    data, y = extract_target(data, params)

    X = preprocess_train_data(data, params)

    X_train, X_val, y_train, y_val = split_train_val_data(X, y, params)
    # train_test_split(
    #     X, y, test_size=0.1, random_state=0, stratify=y
    # )

    model = LogisticRegression(random_state=0, penalty='l2', C=0.9).fit(X_train, y_train)

    save_model(model, params.model_path)

    predict = model.predict(X_val)
    evaluate_model(predict, y_val)


if __name__ == '__main__':
    train_model('configs/train_config.yaml')
