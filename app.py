from flask import Flask, render_template, request
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded", 400

    image = request.files["image"]
    if image.filename == "":
        return "No selected file", 400

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)

    # 🧪 Placeholder predictions (replace with Deep Learning model output)
    gender = "Male"
    age = "25"

    return render_template("index.html", gender=gender, age=age, image_url=image_path)

if __name__ == "__main__":
    app.run(debug=True)
