import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight


# ==========================================
# RANDOM SEED
# ==========================================

SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ==========================================
# DATASET PATHS
# ==========================================

train_dir = "dataset/train"
test_dir  = "dataset/test"


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=10,

    zoom_range=0.1,

    horizontal_flip=True,

    width_shift_range=0.05,

    height_shift_range=0.05,

    shear_range=0.05

)

test_datagen = ImageDataGenerator(

    rescale=1./255

)


# ==========================================
# LOAD DATASET
# ==========================================

train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=(224,224),

    batch_size=8,

    class_mode='categorical'

)

test_generator = test_datagen.flow_from_directory(

    test_dir,

    target_size=(224,224),

    batch_size=8,

    class_mode='categorical',

    shuffle=False

)


# ==========================================
# CLASS WEIGHTS
# ==========================================

class_weights = compute_class_weight(

    class_weight='balanced',

    classes=np.unique(train_generator.classes),

    y=train_generator.classes

)

class_weights = dict(enumerate(class_weights))


# ==========================================
# DENSENET169 MODEL
# ==========================================

base_model = DenseNet169(

    weights='imagenet',

    include_top=False,

    input_shape=(224,224,3)

)


# ==========================================
# FINE TUNING
# ==========================================

for layer in base_model.layers[:-40]:

    layer.trainable = False

for layer in base_model.layers[-40:]:

    layer.trainable = True


# ==========================================
# BUILD MODEL
# ==========================================

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(1024, activation='relu'),

    Dropout(0.5),

    Dense(512, activation='relu'),

    Dropout(0.3),

    Dense(3, activation='softmax')

])


# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(

    optimizer=Adam(learning_rate=0.00005),

    loss=tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.1
    ),

    metrics=['accuracy']

)


# ==========================================
# CALLBACKS
# ==========================================

early_stop = EarlyStopping(

    monitor='val_loss',

    patience=6,

    restore_best_weights=True

)

reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.2,

    patience=3,

    verbose=1

)


# ==========================================
# TRAIN MODEL
# ==========================================

history = model.fit(

    train_generator,

    validation_data=test_generator,

    epochs=50,

    callbacks=[early_stop, reduce_lr],

    class_weight=class_weights

)


# ==========================================
# SAVE MODEL
# ==========================================

model.save("densenet_model.h5")

print("\nModel Saved Successfully")


# ==========================================
# PREDICTIONS
# ==========================================

pred = model.predict(test_generator)

y_pred = np.argmax(pred, axis=1)

y_true = test_generator.classes


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix\n")

print(cm)


# ==========================================
# CONFUSION MATRIX GRAPH
# ==========================================

plt.figure(figsize=(8,6))

sns.heatmap(

    cm,

    annot=True,

    fmt='d',

    cmap='Blues',

    xticklabels=test_generator.class_indices.keys(),

    yticklabels=test_generator.class_indices.keys()

)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("DenseNet Confusion Matrix")

plt.show()


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report\n")

print(

    classification_report(

        y_true,

        y_pred,

        target_names=list(

            test_generator.class_indices.keys()

        )

    )

)


# ==========================================
# ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(

    history.history['accuracy']

)

plt.plot(

    history.history['val_accuracy']

)

plt.title(

    "Model Accuracy"

)

plt.xlabel(

    "Epoch"

)

plt.ylabel(

    "Accuracy"

)

plt.legend(

    ['Train','Validation']

)

plt.show()


# ==========================================
# LOSS GRAPH
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(

    history.history['loss']

)

plt.plot(

    history.history['val_loss']

)

plt.title(

    "Model Loss"

)

plt.xlabel(

    "Epoch"

)

plt.ylabel(

    "Loss"

)

plt.legend(

    ['Train','Validation']

)

plt.show()