from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = YOLO("yolov8n.pt")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    results = model(filepath)

    annotated_frame = results[0].plot()

    output_path = os.path.join(
        "static",
        "output.jpg"
    )

    cv2.imwrite(output_path, annotated_frame)

    return render_template(
        "index.html",
        uploaded=True,
        output_image=output_path
    )

if __name__ == "__main__":
    app.run(debug=True)