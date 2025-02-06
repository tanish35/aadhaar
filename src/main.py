# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import numpy as np
import pickle
import os
import tempfile
import cv2
import gc
from face_verify import verify_faces, preprocess_image, extract_aadhaar_face

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to Aadhaar Face Verification API"}

@app.head("/")
async def head_root():
    return {"message": "Aadhaar Face Verification API is running"}

@app.post("/verify-face/")
async def verify_face_endpoint(
    webcam_image: UploadFile = File(...),
    aadhaar_image: UploadFile = File(...)
):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            webcam_path = os.path.join(temp_dir, "webcam_original.jpg")
            aadhaar_path = os.path.join(temp_dir, "aadhaar_original.jpg")
            
            with open(webcam_path, "wb") as buffer:
                shutil.copyfileobj(webcam_image.file, buffer)
            with open(aadhaar_path, "wb") as buffer:
                shutil.copyfileobj(aadhaar_image.file, buffer)
            
            webcam_img = cv2.imread(webcam_path)
            aadhaar_img = cv2.imread(aadhaar_path)
            
            webcam_img = webcam_img.astype(np.float32) / 255.0
            aadhaar_img = aadhaar_img.astype(np.float32) / 255.0
            
            webcam_img = cv2.cvtColor(webcam_img, cv2.COLOR_BGR2RGB)
            aadhaar_img = cv2.cvtColor(aadhaar_img, cv2.COLOR_BGR2RGB)
            
            processed_webcam_path = os.path.join(temp_dir, "webcam_processed.jpg")
            processed_aadhaar_path = os.path.join(temp_dir, "aadhaar_processed.jpg")
            
            cv2.imwrite(processed_webcam_path, (webcam_img * 255).astype(np.uint8))
            cv2.imwrite(processed_aadhaar_path, (aadhaar_img * 255).astype(np.uint8))
            
            result = verify_faces(processed_webcam_path, processed_aadhaar_path)
            
            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])
            
            return result
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
