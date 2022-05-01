from dataclasses import dataclass
from typing import List


@dataclass()
class Features:
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: str
    # use_log_trick: bool = field(default=True)