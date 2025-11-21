from fastapi.testclient import TestClient
from main import app
from sklearn.datasets import load_breast_cancer

client=TestClient(app)

def test_api():
    data=load_breast_cancer()
    sample=data.data[0].tolist()
    response=client.post("/predict",json={'values':sample})
    assert response.status_code==200
    assert response.json()['prediction'] in [0,1]