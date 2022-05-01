from marshmallow_dataclass import class_schema
from dataclasses import dataclass
import yaml

from ml_project.features import Features


@dataclass()
class TrainingParams:
    input_data_path: str
    output_scaler_path: str
    output_model_path: str
    features: Features


TrainingParamsSchema = class_schema(TrainingParams)


def read_training_params(path: str) -> TrainingParams:
    with open(path, "r") as input_stream:
        schema = TrainingParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
