from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import requests
import tempfile

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STANDARD_FOLDER = 'standard_components'
BASE_URL = "https://pcb-component-matcher-production.up.railway.app"  # Change to your deployed URL

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- Component Metadata ----------
COMPONENT_INFO = {
    "esp32_board.jpg": {
        "name": "ESP32 Board",
        "description": "A microcontroller module with built-in Wi-Fi and Bluetooth, used for IoT projects, embedded systems, and robotics."
    },
    "switch.jpg": {
        "name": "Switch",
        "description": "An electrical component that opens or closes a circuit, controlling the flow of electricity to a device."
    },
    "usb_port.jpg": {
        "name": "USB Power Port",
        "description": "A connector used to supply power and data transfer, commonly for charging devices or powering electronics."
    }
}


# ---------- Image Matching Logic ----------
def load_image_gray(path):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def identify_component(upload_path, standard_folder):
    uploaded = load_image_gray(upload_path)
    best_score = -1
    best_match = None

    for filename in os.listdir(standard_folder):
        standard_path = os.path.join(standard_folder, filename)
        standard = load_image_gray(standard_path)

        try:
            h, w = min(uploaded.shape[0], standard.shape[0]), min(uploaded.shape[1], standard.shape[1])
            uploaded_resized = cv2.resize(uploaded, (w, h))
            standard_resized = cv2.resize(standard, (w, h))

            score = ssim(uploaded_resized, standard_resized)
            if score > best_score:
                best_score = score
                best_match = filename
        except Exception:
            continue

    if best_match:
        info = COMPONENT_INFO.get(best_match, {
            "name": os.path.splitext(best_match)[0],
            "description": "No detailed description available."
        })
        return {
            "component": info["name"],
            "description": info["description"],
            "match_image": best_match,
            "match_image_url": f"{BASE_URL}/{STANDARD_FOLDER}/{best_match}",
            "similarity_score": round(best_score, 3)
        }
    else:
        return {"error": "No match found"}


# ---------- Routes ----------
@app.route('/')
def index():
    return """
    <h1>PCB Component Matcher</h1>
    <form action="/identify" method="post" enctype="multipart/form-data">
        <p><input type="file" name="image" required></p>
        <p><input type="submit" value="Upload & Identify"></p>
    </form>
    """


@app.route('/identify', methods=['POST'])
def identify():
    filepath = None

    # Handle form-data upload
    if 'image' in request.files:
        file = request.files['image']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

    # Handle JSON with image URL
    elif request.is_json:
        data = request.get_json()
        if 'image' in data:
            image_url = data['image']
            try:
                resp = requests.get(image_url, timeout=10)
                if resp.status_code == 200:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                    temp_file.write(resp.content)
                    temp_file.close()
                    filepath = temp_file.name
                else:
                    return jsonify({'error': 'Failed to fetch image from URL'}), 400
            except Exception as e:
                return jsonify({'error': f'Error downloading image: {str(e)}'}), 400

    # No valid input
    if not filepath:
        return jsonify({'error': 'No image provided'}), 400

    match_result = identify_component(filepath, STANDARD_FOLDER)
    return jsonify(match_result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
