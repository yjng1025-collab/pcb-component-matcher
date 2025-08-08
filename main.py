from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STANDARD_FOLDER = 'standard_components'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------- Image Matching Logic (formerly compare_component.py) ----------
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
        return {
            "component": component_name,
            "match_image": best_match,
            "similarity_score": round(best_score, 3),
            "explanation": explanation
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
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    match_result = identify_component(filepath, STANDARD_FOLDER)
    return jsonify(match_result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
