import torch
from model.model import ForgeryDetector
import cv2

model = ForgeryDetector()
model.eval()

img = cv2.imread("test.jpg")
img = cv2.resize(img, (224, 224))
img = torch.tensor(img).permute(2,0,1).float().unsqueeze(0) / 255.0

with torch.no_grad():
    output = model(img)

print("Forgery Probability:", output.item())