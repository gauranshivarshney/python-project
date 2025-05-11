import cv2
import numpy as np
import os

def extract_frames(video_path, output_dir, interval=1, prefix=""):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    
    count = 0
    saved_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_name = os.path.join(output_dir, f"{prefix}_frame_{count}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_frames.append(frame_name)
        count += 1

    cap.release()
    return saved_frames

def compute_histogram(image_path):
    image = cv2.imread(image_path)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.tolist()
