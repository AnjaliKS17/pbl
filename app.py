from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('models/densenet_model.h5')

classes = ['Benign', 'Malignant', 'Normal']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    file.save(filepath)

    img = image.load_img(filepath, target_size=(224,224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    result = classes[np.argmax(prediction)]

    return render_template(
        'result.html',
        prediction=result,
        image_path=filepath
    )

if __name__ == '__main__':
    app.run(debug=True)