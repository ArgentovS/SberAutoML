import pandas as pd
import datetime as dt
import dill

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, Dict


class Form(BaseModel):
    client_id: Union[None, str]
    visit_date: Union[None, str]
    visit_time: Union[None, str]
    utm_source: Union[None, str]
    utm_medium: Union[None, str]
    utm_campaign: Union[None, str]
    utm_adcontent: Union[None, str]
    utm_keyword: Union[None, str]
    device_category: Union[None, str]
    device_os: Union[None, str]
    device_brand: Union[None, str]
    device_screen_resolution: Union[None, str]
    device_browser: Union[None, str]
    geo_country: Union[None, str]
    geo_city: Union[None, str]


class Prediction(BaseModel):
    Client_id: str
    Result: str




app = FastAPI()
with open('model.pickle', 'rb') as file:
    model = dill.load(file)


@app.get('/status')
def status():
    return 'Я в порядке'


@app.get('/version')
def version():
    return model['metadata']

@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    with open('model.pickle', 'rb') as file:
        model = dill.load(file)
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['best_model'].predict(df)
    return {
            'Client_id': form.client_id,
            'Result': str(y[0])
          }