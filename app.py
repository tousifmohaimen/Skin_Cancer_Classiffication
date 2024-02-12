from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load the trained model
model = load_model("Skin_Cancer.h5")

# Function to preprocess the image
def preprocess_image(image_path):
    # Load and resize the image
    img = image.load_img(image_path, target_size=(28, 28))
    # Convert to a numpy array and normalize
    img_array = image.img_to_array(img) / 255.0
    # Expand dimension for batch prediction
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Main route
@app.route("/")
def index():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get the uploaded image
        image_file = request.files["image"]
        # Check if image is uploaded
        if image_file:
            # Save the image temporarily
            image_path = os.path.join(app.root_path, "static", "tmp", image_file.filename)
            image_file.save(image_path)
            # Preprocess the image
            preprocessed_image = preprocess_image(image_path)
            # Make prediction
            prediction = model.predict(preprocessed_image)[0]
            # Apply threshold
            threshold = 0.5  # Adjust as needed
            predicted_class = np.argmax(prediction)
            # Map predicted class to cancer type
            cancer_types = {
                0: 'Actinic keratoses and intraepithelial carcinomae',
                1: 'Basal cell carcinoma',
                2: 'Benign keratosis-like lesions',
                3: 'Dermatofibroma',
                4: 'Melanocytic nevi',
                5: 'Pyogenic granulomas and hemorrhage',
                6: 'Melanoma'
            }
            # Get the corresponding cancer type for the predicted class
            predicted_cancer_type = cancer_types.get(predicted_class, 'Unknown')
            # Remove temporary image
            os.remove(image_path)
            # Return prediction as JSON
            return jsonify({"prediction": int(predicted_class), "cancer_type": predicted_cancer_type})
        else:
            return jsonify({"error": "No image uploaded"})

if __name__ == "__main__":
    app.run(debug=True)
