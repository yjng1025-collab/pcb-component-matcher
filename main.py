from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from compare_component import identify_component

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STANDARD_FOLDER = 'standard_components'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

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
