import pandas as pd
import numpy as np
import joblib
import logging
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler

from ml_project.entity import TrainingParams

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def extract_target(
        data: pd.DataFrame,
        params: TrainingParams
) -> Tuple[pd.DataFrame, np.ndarray]:
    target_col = params.features.target_col
    target = data[target_col].values
    data = data.drop(columns=[target_col])
    return data, target


def extract_cat_num_features(
        params: TrainingParams
) -> Tuple[List[str], List[str]]:
    cat_features = params.features.categorical_features
    num_features = params.features.numerical_features

    logger.info(f'Preprocess train data parameters: '
                f'categorical parameters = {cat_features}, '
                f'numerical_features = {num_features}, '
                f'target_col = {params.features.target_col}')
    return cat_features, num_features


def preprocess_train_data(
        data: pd.DataFrame,
        params: TrainingParams
) -> pd.DataFrame:
    cat_features, num_features = extract_cat_num_features(params)

    df_train_cat = data[cat_features]
    df_train_num = data[num_features]

    cat_columns = df_train_cat.columns
    num_columns = df_train_num.columns

    df_cat = pd.get_dummies(
        df_train_cat,
        columns=cat_columns,
        prefix=cat_columns
    )

    scaler = StandardScaler()
    transformed_num = scaler.fit_transform(df_train_num.to_numpy())
    df_num = pd.DataFrame(transformed_num, columns=num_columns)
    with open(params.scaler_path, "wb") as f:
        joblib.dump(scaler, f)

    return pd.concat([df_cat, df_num], axis=1)


def preprocess_test_data(
        data: pd.DataFrame,
        params: TrainingParams
) -> pd.DataFrame:
    cat_features, num_features = extract_cat_num_features(params)

    df_test_cat = data[cat_features]
    df_test_num = data[num_features]

    cat_columns = df_test_cat.columns
    num_columns = df_test_num.columns

    with open(params.scaler_path, "rb") as f:
        scaler = joblib.load(f)

    df_cat = pd.get_dummies(
        df_test_cat,
        columns=cat_columns,
        prefix=cat_columns
    )

    transformed_num = scaler.transform(df_test_num.to_numpy())
    df_num = pd.DataFrame(transformed_num, columns=num_columns)

    return pd.concat([df_cat, df_num], axis=1)
