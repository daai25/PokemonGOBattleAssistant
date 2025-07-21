import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# We will convert our images to a standard size
IMG_SIZE = (64, 64)

# Where are the image directories?
DATASET_DIR = "../../../data_acquisition/image_dataset/final_pokemon_dataset/train"
# Load the images
X, y = [], []
# Images are labeled by the directory they are in
labels = sorted(os.listdir(DATASET_DIR))
for label_idx, label_name in enumerate(labels):
    folder = os.path.join(DATASET_DIR, label_name)
    if not os.path.isdir(folder):
        continue
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        try:
            img = load_img(img_path, target_size=IMG_SIZE)
            # Normalize colors
            img_array = img_to_array(img) / 255.0
            X.append(img_array)
            y.append(label_idx)
        except Exception as e:
            # If we can't load an image (perhaps it is a type the library
            # doesn't understand), then skip it but print a message.
            print(f"Error loading {img_path}: {e}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)
# Convert labels to one-hot encoding
y_categorical = to_categorical(y, num_classes=len(labels))

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model!
history = model.fit(X, y_categorical, epochs=50, validation_split=0.0)

# Save the model so we can use it later
model.save("pokemon_classifier_50.h5")

# Show statistics on the training:
plt.plot(history.history['accuracy'], label='Accuracy')
plt.title('Accuracy per epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Example code for making predictions with the saved model
'''
# How to use the model for prediction
model = load_model("pokemon_classifier.h5")
img_path = "path_to_your_image.jpg"  # Path to your test image

# Preprocess the image
img = load_img(img_path, target_size=IMG_SIZE)
img_array = img_to_array(img) / 255.0
img_array_expanded = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array_expanded)
predicted_index = np.argmax(prediction)
predicted_label = labels[predicted_index]
confidence = prediction[0][predicted_index]

print(f"Predicted Pokemon: {predicted_label}")
print(f"Confidence: {confidence:.2f}")
'''