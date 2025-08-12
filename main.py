from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import requests

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STANDARD_FOLDER = 'standard_components'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Serve standard components folder
@app.route('/standard_components/<path:filename>')
def serve_standard(filename):
    return send_from_directory(STANDARD_FOLDER, filename)

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
        component_name = os.path.splitext(best_match)[0]
        explanation = f"This appears to be '{component_name}'. Please let the LLM describe its function."
        description = match_info.get("description", "Description not available.")
        match_url = f"{request.url_root}standard_components/{best_match}"
        return {
            "component": component_name,
            "match_image": best_match,
            "match_image_url": match_url,
            "similarity_score": round(best_score, 3),
            "explanation": explanation,
            "description": description
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

    # Handle standard file upload
    if 'image' in request.files:
        file = request.files['image']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

    # Handle JSON body with image URL
    elif request.is_json:
        data = request.get_json()
        if 'image' in data:
            image_url = data['image']
            try:
                resp = requests.get(image_url)
                filename = secure_filename(image_url.split('/')[-1])
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                with open(filepath, 'wb') as f:
                    f.write(resp.content)
            except Exception as e:
                return jsonify({'error': f'Failed to download image: {str(e)}'}), 400
        else:
            return jsonify({'error': 'No image provided in JSON'}), 400
    else:
        return jsonify({'error': 'No image uploaded'}), 400

    match_result = identify_component(filepath, STANDARD_FOLDER)
    return jsonify(match_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
