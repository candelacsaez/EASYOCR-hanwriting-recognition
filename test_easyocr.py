from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import binascii
import os
import numpy as np
import cv2
import easyocr

app = FastAPI()

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:8000,http://localhost:8000",
    ).split(",")
    if origin.strip()
]

max_image_bytes = int(os.getenv("MAX_IMAGE_BYTES", "3000000"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

reader = easyocr.Reader(["en"], gpu=False)


class ImagePayload(BaseModel):
    image: str


@app.post("/ocr")
def ocr(payload: ImagePayload):
    if "," not in payload.image:
        raise HTTPException(status_code=400, detail="Invalid image payload")

    encoded = payload.image.split(",", 1)[1]

    # Fast size check before decode.
    estimated_size = (len(encoded) * 3) // 4
    if estimated_size > max_image_bytes:
        raise HTTPException(status_code=413, detail="Image too large")

    try:
        img_data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Image is not valid base64")

    if len(img_data) > max_image_bytes:
        raise HTTPException(status_code=413, detail="Image too large")

    np_img = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Image decode failed")

    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    result = reader.readtext(img)
    text = [value for _, value, _ in result]
    return {"text": " ".join(text)}