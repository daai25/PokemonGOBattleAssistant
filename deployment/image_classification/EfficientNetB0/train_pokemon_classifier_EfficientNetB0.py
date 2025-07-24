import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
import matplotlib.pyplot as plt
import os
import json

# === 1. Parameters ===
dataset_folder = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data_acquisition", "image_dataset", "dataset", "final_pokemon_dataset")
img_height, img_width = 224, 224
batch_size = 8
seed = 123

# === 2. Load dataset ===
train_ds = image_dataset_from_directory(
    os.path.join(dataset_folder, "train"),
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    os.path.join(dataset_folder, "test"),
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Further split validation dataset into validation and test datasets
val_size = int(0.2 * len(val_ds))  # 20% for validation, 80% for test
val_ds = val_ds.take(val_size)
test_ds = val_ds.skip(val_size)

class_names = train_ds.class_names
print("Pokémon classes:", class_names)

# === 3. Prefetch & Cache ===
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# === 4. Data augmentation ===
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.4),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.3)
])

# === 5. Transfer learning model ===
base_model = EfficientNetB0(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze base model initially

# === 6. Build model ===
inputs = layers.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)  # preprocess for EfficientNet
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = models.Model(inputs, outputs)

# === 7. Compile ===
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === 8. Callbacks ===
early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

class SaveEveryN(tf.keras.callbacks.Callback):
    def __init__(self, n, folder):
        super().__init__()
        self.n = n
        self.folder = folder

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.n == 0:
            filename = os.path.join(self.folder, f"checkpoint_epoch_{epoch+1:02d}.keras")
            self.model.save(filename, save_format='keras')
            print(f"Saved checkpoint: {filename}")

save_every_10_cb = SaveEveryN(10, checkpoint_dir)

# === 9. Train frozen base model ===
initial_epochs = 40
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=initial_epochs,
    callbacks=[early_stop, save_every_10_cb]
)

# === 10. Unfreeze & fine-tune ===-
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1,
    callbacks=[early_stop, save_every_10_cb]
)

# === 11. Save raw data for plot ===
plot_data = {
    "accuracy": history.history['accuracy'] + history_fine.history['accuracy'],
    "val_accuracy": history.history['val_accuracy'] + history_fine.history['val_accuracy']
}

with open("plot_data.json", "w") as f:
    json.dump(plot_data, f)

# === 11. Plot accuracy ===
acc = plot_data['accuracy']
val_acc = plot_data['val_accuracy']

plt.plot(acc, label='Train')
plt.plot(val_acc, label='Validation')
plt.legend()
plt.title("Accuracy over epochs")
plt.show()

# === 12. Save final model ===
final_model_path = "pokemon_final_model.keras"
model.save(final_model_path, save_format='keras')
print(f"✅ Final model saved to: {final_model_path}")