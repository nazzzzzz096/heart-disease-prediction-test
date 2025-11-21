import numpy as np
import pickle
from preprocessing import preprocess_scale

def load_model():
    with open('model.pkl','rb')as f:
        model=pickle.load(f)
    return model


def predict_cancer(sample):
    arr=np.array(sample).reshape(1,-1)
    arr=preprocess_scale(arr)
    model=load_model()
    return int(model.predict(arr)[0])

