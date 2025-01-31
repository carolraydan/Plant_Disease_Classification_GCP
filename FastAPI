from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

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


MODEL = tf.keras.models.load_model("/Users/carolistical/Desktop/ML_ops/Project_Potato/my_project/api/Model.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/request")
async def request():
    return "Hello, I am alive"

def read_files_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(files: UploadFile = File(...)):
    # Read the file and process it
    image_data = await files.read()  # Read the uploaded image
    image = read_files_as_image(image_data)  # Convert to image format using the helper function

    img_batch = np.expand_dims(image,0)
    


    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

