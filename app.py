#for running this project type uvicorn app:app --reload
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from PIL import Image
import os
import numpy as np
import io
from uuid import uuid4

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

templates = Jinja2Templates(directory="templates")

model = load_model("waste.h5")

labels = {
    0: "cardbord",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash"
}

def preprocess(img: Image.Image):
    img = img.resize((300, 300))
    img_array = np.array(img, dtype="uint8") / 255.0
    return img_array[np.newaxis, ...]

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_pre = preprocess(img)

    prediction = model.predict(img_pre)
    prediction_in = int(np.argmax(prediction))
    prediction_la = labels[prediction_in]
    confidence = round(float(np.max(prediction[0])) * 100, 2)

    filename = f"{uuid4().hex}_{file.filename}"
    filepath = os.path.join("uploads", filename)

    with open(filepath, 'wb') as buffer:
        buffer.write(contents)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction_class": prediction_la,
        "confidence": confidence,
        "img_url": f"/uploads/{filename}"
    })