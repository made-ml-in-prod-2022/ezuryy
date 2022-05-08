import pandas as pd
import click
from entity import read_training_params, fix_path, fix_config
from features import preprocess_test_data
from models import run_test_pipeline, open_model, evaluate_model, save_predict


def predict_model(config_path: str):
    config_path = fix_path(config_path)
    params = fix_config(read_training_params(config_path))
    run_test_pipeline(params)


@click.command(name="predict_model")
@click.argument("config_path")
def predict_model_command(config_path: str):
    predict_model(config_path)


if __name__ == '__main__':
    predict_model_command()

# if __name__ == '__main__':
#     predict_model('configs/config1.yaml')
