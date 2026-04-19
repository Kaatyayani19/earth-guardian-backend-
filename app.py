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

app = Flask(__name__)
CORS(app)

# =========================
# SENTINEL CONFIG
# =========================
config = SHConfig()
config.sh_client_id = "sh-0aec34ba-5045-4d69-9eb1-4253bbc08022"
config.sh_client_secret = "oZf3Fjc4UZLc2kTzKWAyRGJJZLYY4jSz"
config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

# =========================
# MODEL LOAD
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    classes=1,
    activation=None
)

model.load_state_dict(torch.load("segmentation_model.pth", map_location=device))
model.to(device)
model.eval()

# =========================
# MODEL RUN (NO PREPROCESS)
# =========================
def run_model(img):
    # resize only (model requirement)
    img = cv2.resize(img, (256, 256))

    # normalize
    img = img / 255.0
    img = np.array(img, dtype=np.float32)

    # ensure 3 channel
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)

    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.sigmoid(output)

    print("MIN:", pred.min().item(), "MAX:", pred.max().item())

    mask = (pred.squeeze().cpu().numpy() > 0.2).astype('uint8')

    return mask

# =========================
# PERCENTAGE
# =========================
def calculate_percentage(mask):
    forest = np.sum(mask == 1)
    total = mask.size
    return (forest/total)*100, 100 - (forest/total)*100

# =========================
# ENCODE FUNCTION
# =========================
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

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    lat = data.get("lat")
    lon = data.get("lon")

    print("📍 Lat:", lat, "Lon:", lon)

    if lat is None or lon is None:
        return jsonify({"error": "Missing coordinates"}), 400

    # AUTH
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

    # SENTINEL REQUEST
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

    # IMAGE
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    image = np.array(img)

    # MODEL
    mask = run_model(image)
    forest, deforest = calculate_percentage(mask)

    # ENCODE
    original_encoded = encode_image(image)
    mask_encoded = encode_image(mask)

    return jsonify({
        "forest": round(forest, 2),
        "deforestation": round(deforest, 2),
        "image": original_encoded,
        "mask": mask_encoded
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)