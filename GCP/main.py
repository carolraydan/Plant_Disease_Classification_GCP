from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
from google.cloud import storage



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

# Function to download the model from GCP bucket
def download_model_from_gcp(bucket_name, model_path, destination_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.download_to_filename(destination_path)
    print(f"Model downloaded from GCP to {destination_path}")

# Define model path and GCP bucket details
GCP_BUCKET_NAME = "carols_bucket_model"  # GCP bucket name
MODEL_PATH_IN_BUCKET = "models/Model.h5"  # Path to the model in GCP bucket
MODEL_LOCAL_PATH = "/tmp/Model.h5"  # Temporary local path to store the model

# Download the model from GCP (you can do this once and then load it)
download_model_from_gcp(GCP_BUCKET_NAME, MODEL_PATH_IN_BUCKET, MODEL_LOCAL_PATH)

# Load the model into memory
MODEL = tf.keras.models.load_model(MODEL_LOCAL_PATH)
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

    img_batch = np.expand_dims(image, 0)  # Add batch dimension
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
