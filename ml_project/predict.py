import pandas as pd

from enities import read_training_params, fix_path, fix_config
from features import preprocess_test_data
from models import open_model, evaluate_model, save_predict


def predict(config_path: str):
    config_path = fix_path(config_path)
    params = fix_config(read_training_params(config_path))
    target_col = params.features.target_col
    data = pd.read_csv(params.input_test_data_path)

    X_test = preprocess_test_data(data, params)

    model = open_model(params.model_path)

    predict = model.predict(X_test)

    answer = pd.read_csv(params.input_train_data_path)[target_col]

    evaluate_model(predict, answer)

    save_predict(target_col, predict, params.predict_path)


if __name__ == '__main__':
    predict('configs/train_config.yaml')
