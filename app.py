"""
Flask Web Application for Diabetic Retinopathy Detection
"""

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Load the trained model
MODEL_PATH = 'models/xception_final.h5'
model = None

# DR severity levels
DR_LEVELS = {
    0: "No Diabetic Retinopathy",
    1: "Mild Non-proliferative DR",
    2: "Moderate Non-proliferative DR",
    3: "Severe Non-proliferative DR",
    4: "Proliferative DR"
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained model"""
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully!")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}")
            print("Please train the model first using train_model.py")
    return model

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

def predict_dr(image_path):
    """Predict DR level from fundus image"""
    model = load_model()
    if model is None:
        return None, "Model not available. Please train the model first."
    
    try:
        # Preprocess image
        img_array = preprocess_image(image_path)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class] * 100)
        
        return predicted_class, confidence
    except Exception as e:
        return None, f"Error during prediction: {str(e)}"

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/info')
def info():
    """Information page about Diabetic Retinopathy"""
    return render_template('info.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected. Please upload an image.', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected. Please upload an image.', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Secure filename and save
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            predicted_class, result = predict_dr(filepath)
            
            if predicted_class is not None:
                dr_level = DR_LEVELS[predicted_class]
                return render_template('result.html', 
                                     image_path=filepath,
                                     dr_level=dr_level,
                                     level_num=predicted_class,
                                     confidence=result)
            else:
                flash(f'Prediction error: {result}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, BMP).', 'error')
            return redirect(request.url)
    
    return render_template('predict.html')

@app.route('/result')
def result():
    """Result page (redirects to predict if accessed directly)"""
    return redirect(url_for('predict'))

if __name__ == '__main__':
    # Load model on startup
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
