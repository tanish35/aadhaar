from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import io

def extract_aadhaar_face(aadhaar_image_path):
   
    img = cv2.imread(aadhaar_image_path)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        raise Exception("No face detected on Aadhaar card")
    
    x, y, w, h = faces[0]
    
    
    padding = 20
    face_img = img[max(0, y-padding):min(img.shape[0], y+h+padding), 
                   max(0, x-padding):min(img.shape[1], x+w+padding)]
    
    return face_img

def verify_faces(webcam_image_path, aadhaar_image_path):
    try:
        
        aadhaar_face = extract_aadhaar_face(aadhaar_image_path)
        
        
        result = DeepFace.verify(
            img1_path=webcam_image_path,
            img2_path=aadhaar_image_path,
            detector_backend="retinaface",
            model_name="ArcFace",
            distance_metric="cosine",
            enforce_detection=True
        )
        
       
        metrics = {
            "verified": result["verified"],
            "distance_score": result["distance"],
            "threshold": result["threshold"],
            "model": "ArcFace",
            "detector": "RetinaFace"
        }
        
        return metrics
        
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import os
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    webcam_path = os.path.join(BASE_DIR, "images", "curr1.jpg")
    aadhaar_path = os.path.join(BASE_DIR, "images", "aadhaar.png") 

    if not os.path.exists(webcam_path):
        print(f"Error: Webcam image not found at {webcam_path}")
    if not os.path.exists(aadhaar_path):
        print(f"Error: Aadhaar image not found at {aadhaar_path}") 
    
    result = verify_faces(webcam_path, aadhaar_path)
    print("\nVerification Results:")
    print("-------------------")
    for key, value in result.items():
        print(f"{key}: {value}")
