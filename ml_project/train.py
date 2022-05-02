import pandas as pd
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from ml_project.read_config import read_training_params, TrainingParams


def preprocess_train_data(data: pd.DataFrame, params: TrainingParams) -> pd.DataFrame:
    cat_features = params.features.categorical_features
    num_features = params.features.numerical_features

    df_train_cat = data[cat_features]
    df_train_num = data[num_features]

    cat_columns = df_train_cat.columns
    num_columns = df_train_num.columns

    df_cat = pd.get_dummies(df_train_cat, columns=cat_columns, prefix=cat_columns)

    scaler = StandardScaler()
    transformed_num = scaler.fit_transform(df_train_num.to_numpy())
    df_num = pd.DataFrame(transformed_num, columns=num_columns)
    with open(params.scaler_path, "wb") as f:
        joblib.dump(scaler, f)

    return pd.concat([df_cat, df_num], axis=1)


def train_model(config_path: str):
    params = read_training_params(config_path)
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
