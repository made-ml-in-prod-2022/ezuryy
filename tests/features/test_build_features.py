import pandas as pd
from ml_project import extract_target, preprocess_train_data, preprocess_test_data


def test_extract_target(params):
    data = pd.read_csv(params.input_train_data_path)
    expected_target = data[params.features.target_col]
    got_data, got_target = extract_target(data, params)
    feature_count = len(params.features.categorical_features) + len(params.features.numerical_features)

    assert (expected_target == got_target).all()
    assert got_data.shape[0] == got_data.shape[0]
    assert got_data.shape[1] == feature_count


def test_extract_cat_num_features(params):
    cat_features = params.features.categorical_features
    num_features = params.features.numerical_features

    assert cat_features == params.features.categorical_features
    assert num_features == params.features.numerical_features


def test_preprocess_train_data(params):
    data = pd.read_csv(params.input_train_data_path)
    preprocessed_data = preprocess_train_data(data, params)

    expected_feature_count = 4 * len(params.features.categorical_features) + len(params.features.numerical_features)

    assert preprocessed_data.shape[0] == data.shape[0]
    assert preprocessed_data.shape[1] == expected_feature_count


def test_preprocess_test_data(params):
    data = pd.read_csv(params.input_test_data_path)
    preprocessed_data = preprocess_test_data(data, params)

    expected_feature_count = 4 * len(params.features.categorical_features) + len(params.features.numerical_features)

    assert preprocessed_data.shape[0] == data.shape[0]
    assert preprocessed_data.shape[1] == expected_feature_count
