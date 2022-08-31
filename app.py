from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import uvicorn
import joblib
import numpy as np
from extract import emailToFeature

from pydantic import BaseModel


class Email(BaseModel):
    email: str

    class Config:
        schema_extra = {
            "example": {
                "email": "this is a test email"}
        }


app = FastAPI(title='Spam Email Detector API',
              description='Accurately detecting spam mails')


@app.get("/", response_class=PlainTextResponse)
def home():
    return "Welcome! API is working perfectly well. Use /docs to proceed to check if it's a spam mail."


@app.post("/predict")
def pred(mail: Email):
    email = mail.email

    emailTof = emailToFeature('vocabulary.csv')
    email_ = emailTof.fea_vector(email)
    email1 = np.concatenate((np.ones((email_.shape[0], 1)), email_), 1)
    email_final = email1.reshape(-1, 1)

    loaded_model = joblib.load(open('model.pkl', 'rb'))
    prediction = loaded_model.predict(email_final.T)

    if prediction == 0:
        return {"Outcome": "Not a spam message!"}
    else:
        return {"Outcome": "Spam message!"}


# run
if __name__ == '__main__':
    uvicorn.run(app)
