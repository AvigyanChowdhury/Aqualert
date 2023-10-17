from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from pymongo import MongoClient
import json
from bson import ObjectId
from fastapi.responses import JSONResponse
from bson import ObjectId
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS



app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("classifier_model2.h5")

CLASS_NAMES = ["Flooding", "Drainage", "Lake pollution","Flash Flood","River Flooding","Coastal Flooding","Drain Blockage","Drought"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    # Store the prediction in MongoDB
    prediction_data = {
        'class': predicted_class,
        'confidence': confidence
    }
    



   

    # Return the prediction dictionary as a JSON response
    return  prediction_data



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=9000)