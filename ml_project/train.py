import pandas as pd
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from ml_project.read_config import read_training_params


def train_model(config_path: str):
    params = read_training_params(config_path)
    target_col = params.features.target_col
    data = pd.read_csv(params.input_data_path)
    y = data[target_col].values
    data = data.drop(columns=[target_col])
    X = data.values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=0, stratify=y
    )
    scaler = StandardScaler()
    scaler.fit(X)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    # pca = PCA(n_components=10, svd_solver='randomized', random_state=123)
    # X_train = pca.fit_transform(X_train)
    # X_val = pca.transform(X_val)
    model = LogisticRegression(random_state=0, penalty='l2', C=0.9).fit(X_train, y_train)
    # clf.predict(X_val)
    with open(params.output_scaler_path, "wb") as f:
        joblib.dump(scaler, f)
    with open(params.output_model_path, "wb") as f:
        pickle.dump(model, f)

    print('Accuracy: ', model.score(X_val, y_val))
