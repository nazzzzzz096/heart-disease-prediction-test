from model import predict
from sklearn.datasets import load_breast_cancer

def test_model():
    data=load_breast_cancer()
    samp=data.data[0]
    result=predict(samp)
    assert result in [0,1]