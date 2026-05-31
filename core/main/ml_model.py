import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('../models/cancer_model.h5')

classes = ['Benign', 'Malignant', 'Normal']

def predict_cancer(img):

    img_path = 'media/' + img.name

    test_image = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(
        test_image,
        axis=0
    )

    test_image = test_image / 255.0

    prediction = model.predict(test_image)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    result = classes[predicted_class]

    return result, round(confidence, 2)