from read_config import TrainingParams


def fix_path(path: str) -> str:
    return path  # sometimes we need ('../' + path)


def fix_config(params: TrainingParams) -> TrainingParams:
    params.input_train_data_path = fix_path(params.input_train_data_path)
    params.input_test_data_path = fix_path(params.input_test_data_path)
    params.scaler_path = fix_path(params.scaler_path)
    params.model_path = fix_path(params.model_path)
    params.predict_path = fix_path(params.predict_path)
    return params
