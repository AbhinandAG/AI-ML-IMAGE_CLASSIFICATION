from flask import Flask, render_template, request
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

app = Flask("Cat_Dog_Classifier")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once
model = tf.keras.models.load_model("cat_dog_classify.keras")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template("index.html", error="No image selected.")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", error="Please choose an image.")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Load Image
    image = tf.keras.utils.load_img(filepath, target_size=(160, 160))
    image_array = tf.keras.utils.img_to_array(image)

    image_array = tf.expand_dims(image_array, 0)

    prediction = model.predict(image_array)

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        result = "🐶 Dog"
        confidence = confidence * 100
    else:
        result = "🐱 Cat"
        confidence = (1 - confidence) * 100

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence, 2),
        image=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)