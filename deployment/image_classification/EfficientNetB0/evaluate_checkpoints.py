import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# === Parameters ===
checkpoint_dir = "checkpoints"
dataset_folder = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data_acquisition", "image_dataset", "final_pokemon_dataset")
img_height, img_width = 224, 224
batch_size = 8

# === Load test dataset ===
test_ds = image_dataset_from_directory(
    os.path.join(dataset_folder, "test"),
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# === Evaluate each checkpoint for each Pokémon class ===
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".keras")]
checkpoint_files.sort()  # Ensure checkpoints are evaluated in order

class_names = test_ds.class_names
results = {class_name: {} for class_name in class_names}

for checkpoint in checkpoint_files:
    print(f"Evaluating checkpoint: {checkpoint}")
    model = tf.keras.models.load_model(os.path.join(checkpoint_dir, checkpoint))

    for class_name in class_names:
        print(f"  Evaluating class: {class_name}")
        class_test_ds = test_ds.filter(lambda x, y: tf.reduce_any(y == class_names.index(class_name)))
        loss, accuracy = model.evaluate(class_test_ds)
        results[class_name][checkpoint] = {"loss": loss, "accuracy": accuracy}

# === Print results ===
print("\nEvaluation Results:")
for class_name, checkpoints in results.items():
    print(f"Class: {class_name}")
    for checkpoint, metrics in checkpoints.items():
        print(f"  {checkpoint}: Loss = {metrics['loss']:.4f}, Accuracy = {metrics['accuracy']:.4f}")

# === Save results to a file ===
results_file = "checkpoint_evaluation_results_per_class.txt"
with open(results_file, "w") as f:
    for class_name, checkpoints in results.items():
        f.write(f"Class: {class_name}\n")
        for checkpoint, metrics in checkpoints.items():
            f.write(f"  {checkpoint}: Loss = {metrics['loss']:.4f}, Accuracy = {metrics['accuracy']:.4f}\n")

print(f"\n✅ Results saved to {results_file}")
