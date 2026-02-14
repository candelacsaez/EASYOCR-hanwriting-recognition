from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import easyocr

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

reader = easyocr.Reader(['en'], gpu=False)

class ImagePayload(BaseModel):
    image: str  # base64 image

@app.post("/ocr")
def ocr(payload: ImagePayload):
    # Decode base64 image
    img_data = base64.b64decode(payload.image.split(",")[1])
    np_img = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)

    # Preprocess (important for handwriting)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # OCR
    result = reader.readtext(img)

    text = [t for _, t, c in result]
    
    result = reader.readtext(img)
    print("OCR result:", result)
    cv2.imwrite("debug_from_canvas.png", img)

    
    

    return {"text": " ".join(text)}


