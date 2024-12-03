from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

app = Flask("COVID_PREDICTION")

path = r'C:\Users\levia\OneDrive\Desktop\ML\Covid.h5'
try:
    model = tf.keras.models.load_model(path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

if model is None:
    print("Model could not be loaded. Exiting application.")
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')
  
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded image
    os.makedirs('uploads', exist_ok=True)
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)[0][0]
        result = "COVID" if prediction < 0.5 else "Non-COVID"

        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": f"Error processing image: e"}), 500
    finally:
        # Clean up uploaded image
        try:
            os.remove(img_path)
        except Exception as e:
            print("Error removing file:", e)

if __name__ == '__main__':
    app.run(debug=True)
