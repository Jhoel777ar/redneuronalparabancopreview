import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import queue
import os
import json
import time
from camera import Camera, list_cameras
from model import load_or_train_model, predict, train_model
from utils import detect_faces
from audio import play_alert

class App:
    def __init__(self, window):
        self.window = window
        self.camera = None
        self.model = load_or_train_model()
        self.cameras = list_cameras()
        if not self.cameras or "No cameras found" in self.cameras:
            messagebox.showerror("Error", "No se encontraron cámaras disponibles.")
            self.window.destroy()
            return
        self.camera_index = self.cameras[0]
        self.frame_queue = queue.Queue()
        self.running = False
        self.current_frame = None
        self.create_widgets()
        self.processing_thread = threading.Thread(target=self.process_frames_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.update_frame()

    def create_widgets(self):
        self.camera_var = tk.StringVar(value=self.camera_index)
        camera_menu = tk.OptionMenu(self.window, self.camera_var, *self.cameras, command=self.select_camera)
        camera_menu.pack()
        self.video_label = tk.Label(self.window)
        self.video_label.pack()
        self.start_button = tk.Button(self.window, text="Iniciar", command=self.start_capture)
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = tk.Button(self.window, text="Parar", command=self.stop_capture)
        self.stop_button.pack(side=tk.LEFT)
        tk.Button(self.window, text="Capturar para entrenar", command=self.capture_for_training).pack(side=tk.LEFT)
        tk.Button(self.window, text="Confirmar predicción", command=self.confirm_prediction).pack(side=tk.LEFT)
        tk.Button(self.window, text="Reentrenar modelo", command=self.retrain_model).pack(side=tk.LEFT)
        tk.Button(self.window, text="Resetear modelo", command=self.reset_model).pack(side=tk.LEFT)
        tk.Button(self.window, text="Predicción incorrecta", command=self.prediction_incorrect).pack(side=tk.LEFT)

    def select_camera(self, index):
        if self.camera:
            self.camera.release()
            self.camera = None
        try:
            self.camera_index = index
            self.camera = Camera(index)
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def start_capture(self):
        self.running = True

    def stop_capture(self):
        self.running = False

    def process_frames_thread(self):
        while True:
            if self.running and self.camera:
                frame = self.camera.get_frame()
                if frame is not None:
                    self.current_frame = frame.copy()
                    faces = detect_faces(frame)
                    for (x, y, w, h) in faces:
                        face_img = frame[y:y+h, x:x+w]
                        if face_img.size == 0:
                            continue
                        face_img_resized = cv2.resize(face_img, (80, 80))
                        face_img_processed = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2RGB)
                        prediction = predict(self.model, face_img_processed)
                        if prediction[0] > 0.5:  # Gafas
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, "Gafas", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        if prediction[1] > 0.5:  # Gorra
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, "Gorra", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    frame_resized = cv2.resize(frame, (80, 80))
                    frame_processed = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    prediction = predict(self.model, frame_processed)
                    if prediction[2] > 0.5:  # Celular
                        play_alert()
                    self.frame_queue.put(frame)
            time.sleep(0.01)

    def update_frame(self):
        try:
            frame = self.frame_queue.get_nowait()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        except queue.Empty:
            pass
        self.window.after(33, self.update_frame)

    def capture_for_training(self):
        if self.current_frame is not None:
            img_path = os.path.join("confirmed_data/images", f"img_{int(time.time())}.jpg")
            os.makedirs("confirmed_data/images", exist_ok=True)
            cv2.imwrite(img_path, self.current_frame)
            labels = simpledialog.askstring("Etiquetas", "Ingrese etiquetas correctas (ej. gafas,gorra,celular) o deje vacío si no hay nada")
            labels = [label.strip() for label in labels.split(",")] if labels else []
            with open("confirmed_data/labels.json", "a") as f:
                json.dump({"image_path": img_path, "labels": labels}, f)
                f.write("\n")
            self.retrain_model_async()

    def confirm_prediction(self):
        if self.current_frame is not None:
            response = messagebox.askyesno("Confirmar", "¿La predicción es correcta?")
            if not response:
                self.prediction_incorrect()
            else:
                self.capture_for_training()

    def prediction_incorrect(self):
        self.capture_for_training()

    def retrain_model_async(self):
        threading.Thread(target=self.retrain_model).start()

    def retrain_model(self):
        self.running = False
        self.model = train_model()
        self.running = True

    def reset_model(self):
        self.running = False
        if os.path.exists("models/model.h5"):
            os.remove("models/model.h5")
        self.model = load_or_train_model()
        self.running = True

    def on_closing(self):
        self.running = False
        if self.camera:
            self.camera.release()
        self.window.destroy()