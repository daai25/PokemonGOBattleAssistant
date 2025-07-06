import os
import shutil
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Folder paths
script_dir = os.path.dirname(os.path.abspath(__file__))
source_folder = os.path.join(script_dir, "compressed_dataset_pokemon_images")
clean_folder = os.path.join(script_dir, "compressed_dataset_pokemon_images_cleaned")
os.makedirs(clean_folder, exist_ok=True)

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Multi-Prompts (positive and negatives)
texts = [
    "a Pokémon trading card",  # positive (we SKIP these)
    "a Pokémon artwork",       # negative
    "a Pokémon screenshot"     # negative
]

# Thresholds
score_threshold = 0.9       # confident it's a card
diff_threshold = 0.2        # card score must exceed others by this much

# Process all images recursively
for root, dirs, files in os.walk(source_folder):
    for filename in files:
        file_path = os.path.join(root, filename)
        relative_path = os.path.relpath(file_path, source_folder)

        try:
            image = Image.open(file_path).convert("RGB")

            # Optional: Aspect ratio filter (skip images with card-like shape)
            width, height = image.size
            aspect_ratio = width / height
            if not (0.7 < aspect_ratio < 1.5):
                target_path = os.path.join(clean_folder, relative_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy(file_path, target_path)
                print(f"✅ Copied (non-card shape): {relative_path}")
                continue

            # CLIP inference
            inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()[0]

            # Scores
            score_card = probs[0]
            score_artwork = probs[1]
            score_screenshot = probs[2]

            # Find the highest scoring class
            scores = {
                "card": score_card,
                "artwork": score_artwork,
                "screenshot": score_screenshot
            }
            predicted_class = max(scores, key=scores.get)
            predicted_score = scores[predicted_class]

            print(f"{filename}: Card={score_card:.2f}, Artwork={score_artwork:.2f}, Screenshot={score_screenshot:.2f} → Predicted: {predicted_class} ({predicted_score:.2f})")

            # Decision: only copy if NOT a card
            if predicted_class == "card" and predicted_score >= score_threshold:
                print(f"🛑 Skipped (is a card): {relative_path}")
            else:
                target_path = os.path.join(clean_folder, relative_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy(file_path, target_path)
                print(f"✅ Copied: {relative_path}")

        except Exception as e:
            print(f"⚠️ Error processing {file_path}: {e}")

print("✅ Non-card images saved in 'compressed_dataset_pokemon_images_cleaned'")