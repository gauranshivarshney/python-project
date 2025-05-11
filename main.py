from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from typing import List
import os
import shutil
import uuid
import cv2
import numpy as np

from utils import extract_frames, compute_histogram
from qdrant_setup import init_qdrant, insert_vectors, search_vectors

app = FastAPI()
FRAME_DIR = "frames"
COLLECTION_NAME = "video_frames"

os.makedirs(FRAME_DIR, exist_ok=True)
init_qdrant(COLLECTION_NAME)

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...), interval: int = Form(1)):
    video_id = str(uuid.uuid4())
    video_path = f"temp_{video_id}.mp4"
    
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    frames = extract_frames(video_path, FRAME_DIR, interval, video_id)
    vectors = []
    payloads = []

    for i, frame_path in enumerate(frames):
        vector = compute_histogram(frame_path)
        vectors.append(vector)
        payloads.append({
            "video_id": video_id,
            "frame_path": frame_path,
        })

    insert_vectors(COLLECTION_NAME, vectors, payloads)
    os.remove(video_path)

    return {"message": f"{len(frames)} frames processed and stored.", "video_id": video_id}

@app.post("/search/")
async def search_similar(vector: List[float], top_k: int = 5):
    results = search_vectors(COLLECTION_NAME, vector, top_k)

    output = []
    for result in results:
        payload = result.payload
        output.append({
            "score": result.score,
            "frame_path": payload["frame_path"],
            "vector": result.vector
        })

    return JSONResponse(content=output)