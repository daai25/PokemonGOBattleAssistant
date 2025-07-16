import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
import matplotlib.pyplot as plt
import os
from PIL import Image

# === 1. Fix PNG iCCP warnings ===
dataset_folder = r"C:\Users\dylan\Documents\Dev\PokemonGOBattleAssistant\data_acquisition\image_dataset\last_dataset"
for root, _, files in os.walk(dataset_folder):
    for f in files:
        if f.lower().endswith('.png'):
            full_path = os.path.join(root, f)
            img = Image.open(full_path)
            img.save(full_path, icc_profile=None)

# === 2. Parameters ===
img_height, img_width = 224, 224
batch_size = 16
seed = 123

# === 3. Load dataset ===
train_ds = image_dataset_from_directory(
    dataset_folder,
    validation_split=0.3,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    dataset_folder,
    validation_split=0.3,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Pokémon classes:", class_names)

# === 4. Prefetch & Cache ===
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# === 5. Data augmentation ===
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.4),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.3)
])

# === 6. Transfer learning model ===
base_model = EfficientNetB0(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # zunächst einfrieren

# === 7. Build model ===
inputs = layers.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)  # wichtig für EfficientNet
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = models.Model(inputs, outputs)

# === 8. Compile ===
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === 9. Callbacks ===
# 9.1 Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# 9.2 Save every 10 epochs
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

# === 10. Train frozen base model ===
initial_epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=initial_epochs,
    callbacks=[early_stop, save_every_10_cb]
)

# === 11. Unfreeze & fine-tune ===
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # kleinere LR
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

# === 12. Plot accuracy ===
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

plt.plot(acc, label='Train')
plt.plot(val_acc, label='Validation')
plt.legend()
plt.title("Accuracy over epochs")
plt.show()

# === 13. Save final model ===
final_model_path = "pokemon_small_dataset_model.keras"
model.save(final_model_path, save_format='keras')
print(f"✅ Final model saved to: {final_model_path}")