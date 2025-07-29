import cv2
import numpy as np

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en {path}")
    img = cv2.resize(img, (80, 80))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces