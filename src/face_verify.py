from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import pickle
import io
import os
import gc  

def preprocess_image(image_path, max_size=800):
    """Optimize image size before processing"""
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Could not read image at {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width = img.shape[:2]
    if height > max_size or width > max_size:
        scale = max_size / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    
    return img

def extract_aadhaar_face(aadhaar_image_path):
    """Extract face from Aadhaar card with optimized processing"""
    try:
        img = preprocess_image(aadhaar_image_path)
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if not os.path.exists(cascade_path):
            raise Exception(f"Cascade classifier file not found at {cascade_path}")
            
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            raise Exception("Error loading cascade classifier")
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            raise Exception("No face detected on Aadhaar card")
        
        x, y, w, h = faces[0]
        padding = 20
        face_img = img[max(0, y-padding):min(img.shape[0], y+h+padding), 
                      max(0, x-padding):min(img.shape[1], x+w+padding)]
        
        del img, gray
        gc.collect()
        
        return face_img
        
    except Exception as e:
        raise Exception(f"Face extraction failed: {str(e)}")


from deepface import DeepFace
import numpy as np

def verify_faces(webcam_image_path, aadhaar_image_path):
    try:
        # Configure DeepFace to use local weights
        DeepFace.build_model("Facenet")
        
        result = DeepFace.verify(
            img1_path=webcam_image_path,
            img2_path=aadhaar_image_path,
            detector_backend="ssd",
            model_name="Facenet",    # Use model_name instead of model
            distance_metric="euclidean",
            enforce_detection=False,
            align=True
        )
        
        return {
            "verified": result["verified"],
            "distance_score": result["distance"],
            "threshold": result["threshold"],
            "model": "Facenet",
            "detector": "SSD"
        }
        
    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    webcam_path = os.path.join(BASE_DIR, "images", "curr1.jpg")
    aadhaar_path = os.path.join(BASE_DIR, "images", "aadhaar.png")
    
    try:
        for path in [webcam_path, aadhaar_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found at {path}")
        
        result = verify_faces(webcam_path, aadhaar_path)
        
        print("\nVerification Results:")
        print("-------------------")
        for key, value in result.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        gc.collect()
