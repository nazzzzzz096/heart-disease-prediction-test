from fastapi import FastAPI
from model import predict
from pydantic import BaseModel
from typing import List

app=FastAPI()

class Dieses(BaseModel):
    values:List[float]

@app.post('/predict')
def predict(data:Dieses):
    
    pred=predict(data.values)
    return {'prediction':pred}