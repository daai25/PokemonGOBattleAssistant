import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# We will convert our images to a standard size
IMG_SIZE = (64, 64)

# Where are the image directories?
DATASET_DIR = "images"
# Load the images
X, y = [], []
# Images are labeled by the directory they are in
labels = sorted(os.listdir(DATASET_DIR))
for label_idx, label_name in enumerate(labels):
folder = os.path.join(DATASET_DIR, label_name)
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

# Split into train/test. We will use 20% of them for testing, and 80% for training.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
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
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# Save the model so we can use it later
model.save("pokemon_classifier.h5")

# Show statistics on the training:
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy per epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

IMG_SIZE = (64, 64)
labels = ["Abra","Aerodactyl",... YOU FILL IN THE REST ... ]
# Load and preprocess img = load_img(img_path, target_size=IMG_SIZE) img_array = img_to_array(img) / 255.0 img_array_expanded = np.expand_dims(img_array, axis=0) # Predict prediction = model.predict(img_array_expanded) predicted_index = np.argmax(prediction) predicted_label = labels[predicted_index] confidence = prediction[0][predicted_index]