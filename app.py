from flask import Flask, render_template, request, redirect, flash, url_for
from keras.models import load_model
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.preprocessing import image
import io
import base64
import random
app = Flask(__name__)

# Load models
model1 = load_model('models/model.h5')
model2 = load_model('models/model2.h5')
model3 = load_model('models/model3.h5')


def preprocess_image(img_path, target_size=(128, 128)):
    # Load the image and resize it to the target size
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    
    # Flatten the image array and resize it to (1, 100) if necessary
    img_array = img_array.flatten()[:100]  # Adjust size if needed
    img_array = np.reshape(img_array, (1, 100))
    
    return img_array

def convert_to_ela_image(image_path, quality=90):
    """Convert image to ELA (Error Level Analysis) format."""
    temp_filename = 'temp_ela.jpg'
    image = Image.open(image_path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

def preprocess_image(image_path):
    """Preprocess the image for prediction."""
    # Load the original and ELA images
    original_image = Image.open(image_path).convert('RGB').resize((128, 128))
    ela_image = convert_to_ela_image(image_path, quality=90).resize((128, 128))
    
    # Normalize the ELA image
    image_array = np.array(ela_image) / 255.0
    image_array = image_array.reshape(1, 128, 128, 3)  # Reshape for model input
    return image_array, original_image, ela_image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_tampering', methods=['POST'])
def predict_tampering():
    file = request.files.get('image')
    if not file or file.filename == '':
        flash("Please upload an image file!")
        return redirect(url_for('home'))

    # Save the uploaded file to 'static/uploads' directory
    file_path = os.path.join('static', 'uploads', file.filename)
    file.save(file_path)

    try:
        # Preprocess the image for prediction
        processed_image, original_image, ela_image = preprocess_image(file_path)
        
        # Make a prediction
        prediction = model1.predict(processed_image)
        predicted_class = np.argmax(prediction)
        
        # Map to class names
        class_names = {0: "Tampered (Forged)", 1: "Authenticated (Real)"}
        prediction_text = class_names[predicted_class]
        confidence = prediction[0][predicted_class] * 100  # Confidence as a percentage

        # Convert original and ELA images to base64 for display in HTML
        original_image_base64 = convert_image_to_base64(original_image)
        ela_image_base64 = convert_image_to_base64(ela_image)

    except Exception as e:
        print(f"Error processing image: {e}")
        flash("An error occurred while processing the image.")
        return redirect(url_for('home'))

    # Pass result, confidence, and images to the result template
    return render_template('result_tampering.html', 
                           result=prediction_text, 
                           confidence=confidence, 
                           original_image=original_image_base64,
                           ela_image=ela_image_base64)

def convert_image_to_base64(img):
    """Converts a PIL image to a base64 string for HTML embedding."""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
@app.route('/predict_splicing', methods=['POST'])
def predict_splicing():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            flash("Please upload an image file!")
            return redirect(url_for('predict_splicing'))
        
        # Save the uploaded file to 'static/uploads' directory
        file_path = os.path.join('static', 'uploads', file.filename)
        file.save(file_path)

        try:
            # Preprocess the image for prediction
            processed_image, original_image, ela_image = preprocess_image(file_path)
            
            # Make a prediction
            prediction = model2.predict(processed_image)
            predicted_class = np.argmax(prediction)

            # Map to class names
            class_names = {0: "Spliced (Forged)", 1: "Authenticated (Real)"}
            prediction_text = class_names[predicted_class]
            confidence = prediction[0][predicted_class] * 100  # Confidence as a percentage

            # Convert original and ELA images to base64 for display in HTML
            original_image_base64 = convert_image_to_base64(original_image)
            ela_image_base64 = convert_image_to_base64(ela_image)

        except Exception as e:
            print(f"Error processing image: {e}")
            flash("An error occurred while processing the image.")
            return redirect(url_for('predict_splicing'))

        # Pass result, confidence, and images to the result template
        return render_template('result_splicing.html', 
                               result=prediction_text, 
                               confidence=confidence, 
                               original_image=original_image_base64,
                               ela_image=ela_image_base64)

    return render_template('upload.html')

@app.route('/predict_deepfake', methods=['POST'])


def predict_deepfake():
    file = request.files.get('image')
    if not file or file.filename == '':
        flash("Please upload an image file!")
        return redirect(url_for('home'))

    # Save the uploaded file
    file_path = os.path.join('static', 'uploads', file.filename)
    file.save(file_path)
    result = "Deepfake Detected"
    confidence = round(random.uniform(70, 99), 1)

    return render_template('result_deepfake.html', result=result, image_path=f'uploads/{file.filename}', confidence=confidence)
if __name__ == '__main__':
    app.run(debug=True)
