import pandas as pd
import pickle
import joblib
from sklearn import metrics

from read_config import read_training_params, TrainingParams
from fix_path import fix_path, fix_config


def preprocess_test_data(data: pd.DataFrame, params: TrainingParams) -> pd.DataFrame:
    cat_features = params.features.categorical_features
    num_features = params.features.numerical_features

    df_test_cat = data[cat_features]
    df_test_num = data[num_features]

    cat_columns = df_test_cat.columns
    num_columns = df_test_num.columns

    with open(params.scaler_path, "rb") as f:
        scaler = joblib.load(f)

    df_cat = pd.get_dummies(df_test_cat, columns=cat_columns, prefix=cat_columns)

    transformed_num = scaler.transform(df_test_num.to_numpy())
    df_num = pd.DataFrame(transformed_num, columns=num_columns)

    return pd.concat([df_cat, df_num], axis=1)


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
