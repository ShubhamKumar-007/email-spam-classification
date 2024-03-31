import joblib
import pickle
import pandas as pd
import numpy as np
import re
import uvicorn
from fastapi import FastAPI, Body, Form
from pydantic import BaseModel

class finalresult(BaseModel):
    email:str

application=FastAPI(
    title="Email classification model",
    description="Simple api that use NLP model to predict email is ham or spam",
    version="1.0"
)

@app.get('/')

def index():
    return {'message':'this is home page of api'}


with open(r"D:\email_spam_model\email_classification_model.pkl",'rb') as model_file:
    linear_svc_classification_model=joblib.load(model_file)
with open(r"D:\email_spam_model\email_tfidf_vectorizer.pkl",'rb') as model_file:
    tfidf_=joblib.load(model_file)
with open(r"D:\email_spam_model\email_lbl_encoder.pkl",'rb') as model_file:
    lbl_encoder=joblib.load(model_file)

@app.post('/emailClassification')
def get_prediction(description:str):
    try:
        tfidf_val=tfidf_.transform([description])
        prediction=linear_svc_classification_model.predict(tfidf_val)
        inverse=lbl_encoder.inverse_transform(prediction)

        final_results=finalresult(
                email=inverse[0]
                )

        return {"email is":final_results}

    except Exception as e:
       print(e)

# if __name__=='__main__':
#     uvicorn.run(application, host='127.0.0.1',port=5000)