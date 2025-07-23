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
TRAIN_DIR = "../../../data_acquisition/image_dataset/final_pokemon_dataset/train"
TEST_DIR = "../../../data_acquisition/image_dataset/final_pokemon_dataset/test"

# Load the training images
X_train, y_train = [], []
# Images are labeled by the directory they are in
labels = sorted(os.listdir(TRAIN_DIR))
for label_idx, label_name in enumerate(labels):
    folder = os.path.join(TRAIN_DIR, label_name)
    if not os.path.isdir(folder):
        continue
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        try:
            img = load_img(img_path, target_size=IMG_SIZE)
            # Normalize colors
            img_array = img_to_array(img) / 255.0
            X_train.append(img_array)
            y_train.append(label_idx)
        except Exception as e:
            # If we can't load an image (perhaps it is a type the library
            # doesn't understand), then skip it but print a message.
            print(f"Error loading {img_path}: {e}")

# Load the test images
X_test, y_test = [], []
print("Loading test images...")
for label_idx, label_name in enumerate(labels):
    folder = os.path.join(TEST_DIR, label_name)
    if not os.path.isdir(folder):
        continue
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        try:
            img = load_img(img_path, target_size=IMG_SIZE)
            # Normalize colors
            img_array = img_to_array(img) / 255.0
            X_test.append(img_array)
            y_test.append(label_idx)
        except Exception as e:
            print(f"Error loading test image {img_path}: {e}")

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Convert labels to one-hot encoding
y_train_categorical = to_categorical(y_train, num_classes=len(labels))
y_test_categorical = to_categorical(y_test, num_classes=len(labels))

print(f"Training set: {X_train.shape[0]} images")
print(f"Test set: {X_test.shape[0]} images")

# Define the model
model = Sequential([
    # First convolutional block
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2, 2),
    
    # Second convolutional block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Third convolutional block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Fully connected layers
    Flatten(),
    Dense(256, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with batch size of 32 and validation after each epoch using test data
history = model.fit(
    X_train, y_train_categorical, 
    epochs=30, 
    batch_size=32,
    validation_data=(X_test, y_test_categorical),
    verbose=1
)

# Save the model in Keras format
model_name = f"pokemon_classifier_cfar_{int(history.history['val_accuracy'][-1]*100)}.keras"
model.save(model_name, save_format='keras')
print(f"Model saved as '{model_name}'")

# Also save in h5 format for compatibility
model.save("pokemon_classifier_50.h5")
print("Model also saved as 'pokemon_classifier_50.h5' for compatibility")

# Show statistics on the training:
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history_cfar.png')
print("Training history saved as 'training_history_cfar.png'")
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