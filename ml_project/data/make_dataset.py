from sklearn.model_selection import train_test_split
from typing import Tuple
import pandas as pd
import numpy as np

from enities import TrainingParams


def split_train_val_data(
        data: pd.DataFrame, target: pd.DataFrame, params: TrainingParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if params.splitting_params.stratify:
        train_data, val_data, train_target, val_target = train_test_split(
            data, target, test_size=params.splitting_params.val_size,
            random_state=params.splitting_params.random_state, stratify=target
        )
    else:
        train_data, val_data, train_target, val_target = train_test_split(
            data, target, test_size=params.splitting_params.val_size, random_state=params.splitting_params.random_state
        )

    return train_data, val_data, train_target, val_target
