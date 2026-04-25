from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io

from model.model import OptimizedForgeryDetector
from utils.handcrafted import extract_dct_features, extract_zernike_features, extract_lbp_features

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = OptimizedForgeryDetector(use_handcrafted=True).to(device)
model.load_state_dict(torch.load("checkpoints/best.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

def get_handcrafted(img):
    img = img.resize((224,224))
    arr = np.array(img) / 255.0
    try:
        dct = extract_dct_features(arr)
        zernike = extract_zernike_features(arr)
        lbp = extract_lbp_features(arr)
        feat = np.concatenate([dct, zernike, lbp])
    except:
        feat = np.zeros(39)
    feat = np.pad(feat, (0, 39 - len(feat)))[:39]
    return torch.tensor(feat, dtype=torch.float32)

def predict(img):
    x = transform(img).unsqueeze(0).to(device)
    h = get_handcrafted(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(x, h)).item()

    print("DEBUG PROB:", prob)

    if prob > 0.7:
        label = "FORGED"
    elif prob < 0.4:
        label = "AUTHENTIC"
    else:
        label = "UNCERTAIN (low confidence)"

    return label, prob

@app.route("/")
def home():
    return "Forgery Detection API Running"

@app.route("/predict", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        label, prob = predict(image)

        return jsonify({
            "prediction": label,
            "confidence": round(prob, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)