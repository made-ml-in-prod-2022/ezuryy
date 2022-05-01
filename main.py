from ml_project.read_config import read_training_params
import pandas as pd


if __name__ == '__main__':
    config = read_training_params('configs/train_config.yaml')
    print(config)
    path = config.input_data_path
    data = pd.read_csv(path)
    print(data.head())
