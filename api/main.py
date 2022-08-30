from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
CORSMiddleware,
allow_origins=["*"], # Allows all origins
allow_credentials=True,
allow_methods=["*"], # Allows all methods
allow_headers=["*"], # Allows all headers
)

file_path = '../saved_models/1'
MODEL = tf.keras.models.load_model(file_path)
CLASS_NAMES = ["Early_Blight", "Late_Blight", "Healthy"]


def read_file_as_img(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/prediction")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_img(await file.read())
    img_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(prediction[0])] # Look for maximum value in array
    confidence = np.max(prediction[0])

    return {
        'class' : predicted_class,
        'confidence' : float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
