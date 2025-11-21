from fastapi import FastAPI
from model import predict
from pydantic import BaseModel
import numpy as np
from typing import List

app=FastAPI()

class Dieses(BaseModel):
    features:List[float]

@app.post('/predict')
def predict(data:Dieses):
    samp=np.array(data).reshape(1,-1)
    pred=predict(samp)
    return {'prediction':pred}