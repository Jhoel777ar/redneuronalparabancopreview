import cv2

class Camera:
    def __init__(self, index):
        index = int(index.split()[1]) if isinstance(index, str) and "Camera" in index else int(index)
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise ValueError(f"No se puede abrir la cámara {index}")

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()

def list_cameras():
    index = 0
    cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            cameras.append(f"Camera {index}")
        cap.release()
        index += 1
    return cameras if cameras else ["No cameras found"]