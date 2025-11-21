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
    
    pred=predict(data.values)
    return {'prediction':pred}