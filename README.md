# Medical Image Classification (Chest X-Rays)

## 📌 Objective
Build a deep learning model to classify chest X-ray images into **Normal** and **Pneumonia** classes.

## 🛠️ Tools & Skills
- Python, TensorFlow/Keras
- Image Preprocessing (resizing, normalization, augmentation)
- Convolutional Neural Networks (CNNs)
- Model Evaluation (accuracy, confusion matrix, ROC curve)
- Explainability (Grad-CAM visualization)

## 📂 Dataset
- **Kaggle Dataset**: [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

## 🚀 Approach
1. Load & preprocess X-ray images.
2. Train a CNN model for binary classification.
3. Evaluate accuracy, precision, recall, F1.
4. Apply **Grad-CAM** to visualize which parts of the lung influenced predictions.

## 📊 Expected Output
- Classification accuracy > 80%.
- Visualizations showing CNN attention maps.


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Paths (adjust to dataset location)
train_path = "chest_xray/train"
test_path = "chest_xray/test"

# Data generators with augmentation
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_path, 
                                               target_size=(150,150),
                                               batch_size=32,
                                               class_mode='binary')

test_data = test_datagen.flow_from_directory(test_path, 
                                             target_size=(150,150),
                                             batch_size=32,
                                             class_mode='binary')

# CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, validation_data=test_data, epochs=5)

# Plot accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
