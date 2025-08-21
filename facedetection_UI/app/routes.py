import os
import base64
from datetime import datetime
from flask import Blueprint, render_template, request
from werkzeug.utils import secure_filename
from .models import detect_and_recognize

main = Blueprint("main", __name__)

UPLOAD_FOLDER = "dataset/uploads"
LIVE_FOLDER = "dataset/live_captures"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LIVE_FOLDER, exist_ok=True)

@main.route('/')
def index():
    return render_template("index.html")

@main.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # File upload case
        if 'upload_image' in request.files:
            image = request.files['upload_image']
            if image and image.filename != "":
                filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image.filename}")
                image_path = os.path.join(UPLOAD_FOLDER, filename)
                image.save(image_path)
                output_path, results = detect_and_recognize(image_path)
                return render_template("result.html", image_path=output_path, faces=results)

        # Live camera capture case
        if 'captured_image' in request.form:
            data_url = request.form['captured_image']
            if data_url:
                header, encoded = data_url.split(",", 1)
                img_data = base64.b64decode(encoded)
                filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_live.jpg"
                image_path = os.path.join(LIVE_FOLDER, filename)
                with open(image_path, "wb") as f:
                    f.write(img_data)
                output_path, results = detect_and_recognize(image_path)
                return render_template("result.html", image_path=output_path, faces=results)

    return render_template("predict.html")

@main.route('/result')
def result():
    return render_template("result.html")
