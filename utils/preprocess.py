import cv2
import numpy as np

def preprocess_image(image_path, size=224):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (size, size))
    img = img / 255.0
    return img