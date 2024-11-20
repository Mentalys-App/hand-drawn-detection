import os
import cv2
import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('./model/mental_health_model.keras')

def prepare_image(image):
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img_array = np.expand_dims(img, axis=-1)
    img_array = img_array / 255.0
    return img_array

def predict_image(image, temperature=2.0):
    img = prepare_image(image)
    img = np.expand_dims(img, axis=0)
    
    raw_prediction = model.predict(img)[0][0]
    scaled_prediction = 1 / (1 + np.exp(-(np.log(raw_prediction / (1 - raw_prediction)) / temperature)))
    
    if abs(scaled_prediction - 0.5) < 0.2:
        result = "Uncertain"
    else:
        result = "Mental Disorder" if scaled_prediction > 0.5 else "No Mental Disorder"
        
    return result, scaled_prediction

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        
        try:
            result, confidence = predict_image(file)
            return render_template('index.html', 
                                 result=result,
                                 confidence=f"{confidence*100:.2f}%")
        except Exception as e:
            return render_template('index.html', error=f'Error processing image: {str(e)}')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)