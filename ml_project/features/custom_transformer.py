from typing import NoReturn

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)

from ml_project.entity import TrainingParams


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features) -> NoReturn:
        # self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.numerical_features = numerical_features

    def fit(self, data: pd.DataFrame):
        self.scaler.fit(data[self.numerical_features])
        return self

    def transform(self, data: pd.DataFrame):
        data_copy = data[self.numerical_features].copy()
        data_copy = self.scaler.transform(data_copy)
        return data_copy


def preprocess_categorical_features(
    data: pd.DataFrame, ohe: OneHotEncoder, params: TrainingParams
) -> pd.DataFrame:
    cat_columns = params.features.categorical_features
    cat_df = pd.DataFrame(ohe.transform(data[cat_columns]).toarray())
    cat_df.rename(columns=str, inplace=True)
    return cat_df


def preprocess_numerical_features(
    data: pd.DataFrame, transformer: CustomTransformer, params: TrainingParams
) -> pd.DataFrame:
    num_features = params.features.numerical_features
    num_df = pd.DataFrame(
        columns=num_features, data=transformer.transform(data[num_features])
    )
    return num_df


def preprocess_features(
    data: pd.DataFrame,
    ohe: OneHotEncoder,
    transformer: CustomTransformer,
    params: TrainingParams,
) -> pd.DataFrame:
    cat_df = preprocess_categorical_features(data, ohe, params)
    num_df = preprocess_numerical_features(data, transformer, params)
    preprocessed_data = pd.concat([cat_df, num_df], axis=1)
    return preprocessed_data
