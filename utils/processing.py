import cv2
import numpy as np

def read_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, 1)

def read_video(file):
    temp_filename = f"temp_{file.name}"
    with open(temp_filename, "wb") as f:
        f.write(file.read())
    return temp_filename
