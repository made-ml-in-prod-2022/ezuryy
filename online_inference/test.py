from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_main():
    response = client.get("/")
    assert 200 == response.status_code
    assert response.json() == 'Hello! Go to /docs to see methods :)'


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict():
    request_data = [69.0, 1.0, 0.0, 160.0, 234.0, 1.0,
                    2.0, 131.0, 0.0, 0.1, 1.0, 1.0, 0.0]
    request_features = ['age', 'sex', 'cp', 'trestbps', 'chol',
                        'fbs', 'restecg', 'thalach', 'exang',
                        'oldpeak', 'slope', 'ca', 'thal']

    print("Request: ", request_data)
    response = client.get(
        "/predict",
        json={"data": [request_data], "features": request_features},
    )
    assert response.status_code == 200
    print('response.text:', response.text)
    assert response.json() == {'condition': 0.0}
