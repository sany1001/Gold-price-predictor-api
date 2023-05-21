from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
import pickle

app=FastAPI()

class blueprint(BaseModel):
    spx:float
    uso:float
    slv:float
    eur_usd:float

#loading regressor
regressor=pickle.load(open('Gold_Predictor.sav','rb'))

@app.post('/upload')
def upload(input:blueprint):
    #input to json to dictionary
    dict=json.loads(input.json())

    #processing
    input_tuple=(dict['spx'],dict['uso'],dict['slv'],dict['eur_usd'])
    input_data=np.asarray(input_tuple).reshape(1,-1)

    predicted_output=regressor.predict(input_data)
    return predicted_output[0]