from fastapi import FastAPI, File, UploadFile, Request
import tensorflow as tf
import numpy as np
from PIL import Image
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")
model = tf.keras.models.load_model('pneumoniaAndCovid.h5')

def preprocess_image(image_bytes):
    image = Image.open(image_bytes)
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, image_file: UploadFile = File(...)):
    image_bytes = await image_file.read()
    image = preprocess_image(image_bytes)
    prediction = model.predict(image)
    return templates.TemplateResponse("result.html", {"request": request, "prediction": str(prediction[0][0])})
