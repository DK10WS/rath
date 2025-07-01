from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import cv2
import numpy as np
import shutil
import os
from uuid import uuid4
import base64

from model import segment_image, setup_model
from input import stitch_object, save_image

app = FastAPI()
predictor = setup_model()

mask_store = {}

def read_imagefile(file: UploadFile) -> np.ndarray:
    contents = file.file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


@app.post("/segment/")
async def segment_objects(image: UploadFile = File(...)):
    img = read_imagefile(image)
    masks = segment_image(predictor, img)

    if masks.shape[0] == 0:
        return JSONResponse(content={"message": "No objects detected."}, status_code=200)

    session_id = uuid4().hex
    mask_store[session_id] = {"masks": masks, "image": img}

    object_previews = []
    for i, mask in enumerate(masks):
        object_pixels = img.copy()
        object_pixels[mask == 0] = 0

        _, buffer = cv2.imencode('.png', object_pixels)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        object_previews.append({
            "id": i,
            "preview": b64_str
        })

    return {
        "session_id": session_id,
        "objects": object_previews,
        "count": len(object_previews)
    }


@app.post("/stitch/")
async def stitch_selected_objects(
    session_id: str = Form(...),
    selected_ids: str = Form(...),
    background: UploadFile = File(...)
):
    if session_id not in mask_store:
        return JSONResponse(content={"error": "Invalid session ID"}, status_code=400)

    selected_indices = list(map(int, selected_ids.split(',')))
    data = mask_store[session_id]
    img, masks = data["image"], data["masks"]

    bg = read_imagefile(background)
    if bg.shape != img.shape:
        bg = cv2.resize(bg, (img.shape[1], img.shape[0]))

    for i in selected_indices:
        if 0 <= i < len(masks):
            stitch_object(bg, masks[i], img)

    output_path = f"stitched_{uuid4().hex}.png"
    save_image(bg, output_path)

    return FileResponse(output_path, media_type="image/png", filename="stitched.png")
