import pandas as pd
import pickle
from sklearn import metrics

from enities import read_training_params, fix_path, fix_config
from features import preprocess_test_data


def predict(config_path: str):
    config_path = fix_path(config_path)
    params = fix_config(read_training_params(config_path))
    target_col = params.features.target_col
    data = pd.read_csv(params.input_test_data_path)

    X_test = preprocess_test_data(data, params)

    with open(params.model_path, "rb") as f:
        model = pickle.load(f)

    predict = model.predict(X_test)

    answer = pd.read_csv(params.input_train_data_path)[target_col]
    print("Accuracy: ", metrics.accuracy_score(answer, predict))
    print("ROC AUC score: ", metrics.roc_auc_score(answer, predict))
    print("F1-score: ", metrics.f1_score(answer, predict))

    predict_df = pd.DataFrame({target_col: predict})
    predict_df.to_csv(params.predict_path, index=False)
    print(predict_df.head())


if __name__ == '__main__':
    predict('configs/train_config.yaml')
