from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from sentinelhub import SHConfig
import base64
import io
from PIL import Image

import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import joblib

app = Flask(__name__)
CORS(app)

# =========================
# SENTINEL CONFIG (IMPORTANT: apni original keys daalna)
# =========================
config = SHConfig()
config.sh_client_id = "sh-0aec34ba-5045-4d69-9eb1-4253bbc08022"
config.sh_client_secret = "oZf3Fjc4UZLc2kTzKWAyRGJJZLYY4jSz"
config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

# =========================
# DEFORESTATION MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    classes=1,
    activation=None
)
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Kaatyayani/earth-guardian-model",
    filename="segmentation_model.pth"
)

model.load_state_dict(torch.load("segmentation_model.pth", map_location=device))
model.to(device)
model.eval()

# =========================
# GLACIER MODELS
# =========================
ala_model = joblib.load("ALA_linear_regression_model.pkl")
acs_model = joblib.load("ACS_linear_regression_model.pkl")
acn_model = joblib.load("ACN_linear_regression_model.pkl")

# =========================
# MODEL RUN (DEFORESTATION)
# =========================
def run_model(img):
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.array(img, dtype=np.float32)

    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)

    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.sigmoid(output)

    mask = (pred.squeeze().cpu().numpy() > 0.2).astype('uint8')
    return mask

def calculate_percentage(mask):
    forest = np.sum(mask == 1)
    total = mask.size
    return (forest/total)*100, 100 - (forest/total)*100

def encode_image(img_array):
    if img_array.max() <= 1:
        img_array = img_array * 255

    img_array = img_array.astype(np.uint8)

    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)

    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return jsonify({"status": "Backend running 🚀"})

# =========================
# DEFORESTATION
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    lat = data.get("lat")
    lon = data.get("lon")

    if lat is None or lon is None:
        return jsonify({"error": "Missing coordinates"}), 400

    auth_payload = {
        "grant_type": "client_credentials",
        "client_id": config.sh_client_id,
        "client_secret": config.sh_client_secret
    }

    auth_response = requests.post(config.sh_token_url, data=auth_payload)

    if auth_response.status_code != 200:
        return jsonify({"error": "Auth failed"}), 500

    access_token = auth_response.json()["access_token"]

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    url = "https://sh.dataspace.copernicus.eu/api/v1/process"

    payload = {
        "input": {
            "bounds": {
                "bbox": [lon-0.1, lat-0.1, lon+0.1, lat+0.1],
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                }
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": "2023-01-01T00:00:00Z",
                        "to": "2024-12-31T23:59:59Z"
                    },
                    "maxCloudCoverage": 50
                }
            }]
        },
        "output": {
            "width": 512,
            "height": 512,
            "responses": [
                {"identifier": "default", "format": {"type": "image/png"}}
            ]
        },
        "evalscript": """
        //VERSION=3
        function setup() {
          return {
            input: ["B04", "B03", "B02"],
            output: { bands: 3 }
          };
        }
        function evaluatePixel(sample) {
          return [sample.B04, sample.B03, sample.B02];
        }
        """
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        return jsonify({"error": "Sentinel failed"}), 500

    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    image = np.array(img)

    mask = run_model(image)
    forest, deforest = calculate_percentage(mask)

    return jsonify({
        "forest": round(forest, 2),
        "deforestation": round(deforest, 2),
        "image": encode_image(image),
        "mask": encode_image(mask)
    })

# =========================
# GLACIER
# =========================
@app.route("/glacier", methods=["POST"])
def glacier():
    data = request.get_json()

    region = data.get("region")

    temp = data.get("temperature")
    snow = data.get("snow")
    elev = data.get("elevation")

    input_data = np.array([[temp, snow, elev]])

    if region == "ALA":
        model = ala_model
    elif region == "ACS":
        model = acs_model
    elif region == "ACN":
        model = acn_model
    else:
        return jsonify({"error": "Invalid region"}), 400

    pred = model.predict(input_data)[0]

    return jsonify({"prediction": float(pred)})

# =========================
# GLACIER IMPACT
# =========================
@app.route("/glacier-impact", methods=["POST"])
def glacier_impact():
    data = request.get_json()

    region = data.get("region")

    temp = data.get("temperature")
    snow = data.get("snow")
    elev = data.get("elevation")

    cars = data.get("cars")
    trees = data.get("trees")

    co2 = cars * 4.6 - trees * 0.02
    new_temp = temp + (co2 * 0.01)

    input_data = np.array([[new_temp, snow, elev]])

    if region == "ALA":
        model = ala_model
    elif region == "ACS":
        model = acs_model
    elif region == "ACN":
        model = acn_model
    else:
        return jsonify({"error": "Invalid region"}), 400

    new_pred = model.predict(input_data)[0]

    return jsonify({
        "new_prediction": float(new_pred),
        "adjusted_temp": new_temp
    })

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True, port=5001)