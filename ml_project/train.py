import pandas as pd
import click
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

    model = LogisticRegression(random_state=0, penalty='l2', C=0.9).fit(X_train, y_train)

    save_model(model, params.model_path)

    predict = model.predict(X_val)
    evaluate_model(predict, y_val)


@click.command(name="train_model")
@click.argument("config_path")
def train_model_command(config_path: str):
    train_model(config_path)


if __name__ == '__main__':
    train_model_command()

# if __name__ == '__main__':
#     train_model('configs/config1.yaml')
