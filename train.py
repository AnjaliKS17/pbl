from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

valid_data = train_datagen.flow_from_directory(
    'dataset/valid',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

test_data = train_datagen.flow_from_directory(
    'dataset/test',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False

)

model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dense(3,activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history=model.fit(
    train_data,
    validation_data=valid_data,
    epochs=10
)
import matplotlib.pyplot as plt

# Accuracy Graph

plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['Train', 'Validation'])

plt.savefig('accuracy_graph.png')

plt.show()


# Loss Graph

plt.figure(figsize=(8,5))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(['Train', 'Validation'])

plt.savefig('loss_graph.png')

plt.show()

test_loss, test_accuracy = model.evaluate(test_data)

print("Test Accuracy:", test_accuracy)

pred = model.predict(test_data)

predicted_classes = np.argmax(pred, axis=1)

true_classes = test_data.classes

class_labels = list(test_data.class_indices.keys())

# Classification Report

print("\nClassification Report\n")

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_labels
)

print(report)

# Confusion Matrix

cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(6,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.title('Confusion Matrix')

plt.show()

model.save('models/cancer_model.h5')

print("Model Trained Successfully")