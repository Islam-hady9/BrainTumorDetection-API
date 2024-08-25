from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model (Accuracy: 92.16%)
model = tf.keras.models.load_model('Model/brain_tumor_cnn_model.h5')

app = FastAPI()

# Add CORS middleware
orig_cors = ["*"]  # Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=orig_cors,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((128, 128))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Open and preprocess the image
        image = Image.open(file.file)
        image = preprocess_image(image)

        # Make prediction
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        class_names = ["No Tumor", "Tumor"]
        result = class_names[predicted_class]

        return JSONResponse(content={"prediction": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
