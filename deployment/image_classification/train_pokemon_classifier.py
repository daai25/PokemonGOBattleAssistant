import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

# === 1. Dataset Pfad ===
dataset_path = r"C:\Users\dylan\Documents\Dev\PokemonGOBattleAssistant\data_acquisition\image_dataset\last_dataset"

# === 2. Parameter ===
img_height, img_width = 224, 224
batch_size = 8  # kleiner wegen wenig Daten

# === 3. Dataset laden ===
train_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Pokémon Klassen:", class_names)

# === 4. Data Augmentation ===
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
])

# === 5. Transfer Learning Modell ===
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Base einfrieren

# === 6. Modellaufbau ===
inputs = layers.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = models.Model(inputs, outputs)

# === 7. Kompilieren ===
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === 8. Early Stopping Callback ===
early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# === 9. Training ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,  # mehr Epochen wegen kleinen Datensatz
    callbacks=[early_stop]
)

# === 10. Plotten ===
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.legend()
plt.title("Accuracy Verlauf")
plt.show()

# === 11. Modell speichern ===
model.save("pokemon_small_dataset_model")