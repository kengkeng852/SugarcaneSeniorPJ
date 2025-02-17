import os
import pkgutil
if not hasattr(pkgutil, 'ImpImporter'):
    from importlib.machinery import SourceFileLoader
    pkgutil.ImpImporter = SourceFileLoader
import tensorflow as tf

from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from PIL import Image
import numpy as np
import uvicorn
import io
import requests

LINE_CHANNEL_ACCESS_TOKEN = "/4XBNehql4cunpj5crTaGmihVODV6IthLfxnljCzuTcpyaZwh+T47kjTMPK6Tsvtadb6mImMkL3HrUUF8rzUeDQanHctllgGlBa75jBizCzOGbRI43hjiHdrX8nQC8SCEkTmDBQRlYqoLeCrJr9nlwdB04t89/1O/w1cDnyilFU="
LINE_REPLY_API = "https://api.line.me/v2/bot/message/reply"

# Model and Class Labels
MODEL_PATH = "./best_model9.keras"
CLASS_LABELS = ['Banded Chlorosis', 'Brown Spot', 'BrownRust', 'Dried Leaves', 'Grassy shoot', 'Healthy Leaves', 'Pokkah Boeng', 'RedRot', 'Sett Rot', 'Viral Disease', 'Yellow Leaf', 'smut']

try:
    model = load_model(MODEL_PATH, compile=False)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

app = FastAPI()

def preprocess_image(img, image_size=(256, 256)):
    img = img.resize(image_size)
    img = img.convert("L").convert("RGB")
    img = tf.image.adjust_contrast(img, contrast_factor=1.5) 
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def home():
    return {"message": "Welcome to the Sugarcane disease detection API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions))
        predicted_label = CLASS_LABELS[predicted_index]
        confidence = predictions[0][predicted_index]

        return {
            "predicted_label": predicted_label,
            "confidence": round(float(confidence), 2),
            "all_probabilities": [round(float(prob), 4) for prob in predictions[0]]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

@app.post("/webhook")
async def webhook(request: Request):
    try:
        body = await request.json()
        for event in body.get("events", []):
            if event["type"] == "message":
                reply_token = event["replyToken"]
                message_type = event["message"]["type"]

                if message_type == "text":
                    reply_message = {
                        "replyToken": reply_token,
                        "messages": [{"type": "text", "text": "Please send an image for disease prediction."}],
                    }
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
                    }
                    requests.post(LINE_REPLY_API, json=reply_message, headers=headers)

                elif message_type == "image":
                    message_id = event["message"]["id"]
                    # Retrieve image content from LINE
                    headers = {
                        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
                    }
                    image_response = requests.get(
                        f"https://api-data.line.me/v2/bot/message/{message_id}/content", headers=headers
                    )
                    if image_response.status_code == 200:
                        img_bytes = image_response.content
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        img_array = preprocess_image(img)
                        predictions = model.predict(img_array)
                        predicted_index = int(np.argmax(predictions))
                        predicted_label = CLASS_LABELS[predicted_index]
                        confidence = predictions[0][predicted_index]

                        reply_text = f"Prediction: {predicted_label} with confidence {confidence:.2f}"
                    else:
                        reply_text = "Failed to retrieve the image. Please try again."

                    reply_message = {
                        "replyToken": reply_token,
                        "messages": [{"type": "text", "text": reply_text}],
                    }
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
                    }
                    requests.post(LINE_REPLY_API, json=reply_message, headers=headers)
        return {"status": "success"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

#run local
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
