from marshmallow_dataclass import class_schema
from dataclasses import dataclass
import yaml

from .features import Features


@dataclass()
class TrainingParams:
    input_train_data_path: str
    input_test_data_path: str
    scaler_path: str
    model_path: str
    predict_path: str
    features: Features


TrainingParamsSchema = class_schema(TrainingParams)


def read_training_params(path: str) -> TrainingParams:
    with open(path, "r") as input_stream:
        schema = TrainingParamsSchema()
        return schema.load(yaml.safe_load(input_stream))


def fix_path(path: str) -> str:
    return path  # sometimes we need ('../' + path)


def fix_config(params: TrainingParams) -> TrainingParams:
    params.input_train_data_path = fix_path(params.input_train_data_path)
    params.input_test_data_path = fix_path(params.input_test_data_path)
    params.scaler_path = fix_path(params.scaler_path)
    params.model_path = fix_path(params.model_path)
    params.predict_path = fix_path(params.predict_path)
    return params
