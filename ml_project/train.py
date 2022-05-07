import pandas as pd
import click

from data import split_train_val_data
from enities import read_training_params, fix_path, fix_config
from features import preprocess_train_data, extract_target
from models import save_model, evaluate_model, get_model


def train_model(config_path: str):
    config_path = fix_path(config_path)
    params = fix_config(read_training_params(config_path))

    data = pd.read_csv(params.input_train_data_path)
    data, target = extract_target(data, params)

    full_data = preprocess_train_data(data, params)

    train_data, val_data, train_target, val_target = split_train_val_data(full_data, target, params)

    model = get_model(params)
    model.fit(train_data, train_target)
    # if params.model_type == 'LogisticRegression':
    #     model = LogisticRegression(random_state=0, penalty='l2', C=0.9).fit(X_train, y_train)
    # elif params.model_type == 'GaussianNB':
    #     model = GaussianNB()
    # else:
    #     logger.info('error!')

    save_model(model, params.model_path)

    predict = model.predict(val_data)
    evaluate_model(predict, val_target)


@click.command(name="train_model")
@click.argument("config_path")
def train_model_command(config_path: str):
    train_model(config_path)


if __name__ == '__main__':
    train_model_command()

# if __name__ == '__main__':
#     train_model('configs/config1.yaml')
