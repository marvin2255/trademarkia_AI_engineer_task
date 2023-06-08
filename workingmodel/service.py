from fastapi import FastAPI

from model import train_model, predict

app = FastAPI()


@app.get("/")
def start():
    return "App Starting"


@app.post("/train")
def train():
    return train_model()


@app.post("/predict")
def predict_data(user_input: str):
    return predict(user_input)
