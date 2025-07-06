import os
import shutil
import torch
import requests
import hashlib
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from duckduckgo_search import DDGS

# Folder paths
script_dir = os.path.dirname(os.path.abspath(__file__))
source_folder = os.path.join(script_dir, "compressed_dataset_pokemon_images")
clean_folder = os.path.join(script_dir, "compressed_dataset_pokemon_images_cleaned")
os.makedirs(clean_folder, exist_ok=True)

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Threshold
score_threshold = 0.90  # Only accept if confidence ≥ 90%
max_images_per_folder = 30

def get_image_hash(image):
    """Compute SHA256 hash of an image."""
    return hashlib.sha256(image.tobytes()).hexdigest()

def download_images(query, limit=50):
    """Search and download image URLs using DuckDuckGo."""
    urls = []
    with DDGS() as ddgs:
        for r in ddgs.images(query, max_results=limit):
            urls.append(r['image'])
    return urls

def save_image(url, save_path):
    """Download and save an image from URL as JPG."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.save(save_path, "JPEG")
        return img
    except Exception:
        return None

def is_target_pokemon(image, folder_name):
    """Check if image matches target Pokémon using CLIP."""
    positive_prompts = [
        f"a photo of {folder_name} Pokémon",
        f"close-up photo of {folder_name} Pokémon",
        f"official artwork of {folder_name} Pokémon",
        f"3D render of {folder_name} Pokémon",
        f"anime screenshot of {folder_name} Pokémon",
        f"realistic drawing of {folder_name} Pokémon",
        f"high quality image of {folder_name} Pokémon",
        f"full body photo of {folder_name} Pokémon",
        f"portrait of {folder_name} Pokémon",
        f"{folder_name} Pokémon standing alone"
    ]
    negative_prompts = [
        f"a photo of an evolved form of {folder_name}",
        f"a photo of a pre-evolved form of {folder_name}",
        "a photo of a different Pokémon"
    ]
    all_prompts = positive_prompts + negative_prompts

    inputs = processor(text=all_prompts, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).cpu().detach().numpy()[0]

    score_target = probs[0]
    score_other = max(probs[1:])

    return score_target >= score_threshold and score_target > score_other

# Process all folders
for root, dirs, files in os.walk(source_folder):
    folder_name = os.path.basename(root)
    if not files:
        continue  # skip empty folders

    print(f"\n📁 Checking folder: {folder_name}")

    # Get unique hashes of existing images
    existing_hashes = set()
    cleaned_folder_path = os.path.join(clean_folder, folder_name)
    os.makedirs(cleaned_folder_path, exist_ok=True)
    for f in os.listdir(cleaned_folder_path):
        try:
            img = Image.open(os.path.join(cleaned_folder_path, f)).convert("RGB")
            existing_hashes.add(get_image_hash(img))
        except:
            continue

    # Copy original images after filtering
    for filename in files:
        file_path = os.path.join(root, filename)
        target_path = os.path.join(clean_folder, folder_name, filename)

        try:
            img = Image.open(file_path).convert("RGB")
            img_hash = get_image_hash(img)
            if img_hash in existing_hashes:
                continue  # skip duplicates

            if is_target_pokemon(img, folder_name):
                img.save(target_path, "JPEG")
                existing_hashes.add(img_hash)
                print(f"✅ Copied: {folder_name}/{filename}")

        except Exception as e:
            print(f"⚠️ Error processing {file_path}: {e}")

    # Download more images if needed
    current_count = len(existing_hashes)
    if current_count >= max_images_per_folder:
        print(f"🎉 Folder {folder_name} already has {current_count} unique images.")
        continue

    print(f"⬇️ Downloading more images for {folder_name}...")
    image_urls = download_images(f"{folder_name} Pokémon", limit=100)

    for idx, url in enumerate(tqdm(image_urls, desc=f"Downloading {folder_name}")):
        if len(existing_hashes) >= max_images_per_folder:
            break
        try:
            img = Image.open(requests.get(url, timeout=5).content).convert("RGB")
            img_hash = get_image_hash(img)
            if img_hash in existing_hashes:
                continue  # duplicate

            if is_target_pokemon(img, folder_name):
                save_path = os.path.join(clean_folder, folder_name, f"web_{idx}.jpg")
                img.save(save_path, "JPEG")
                existing_hashes.add(img_hash)
                print(f"✅ Saved new image: {save_path}")
            else:
                print(f"🛑 Skipped (not {folder_name}): {url}")

        except Exception:
            continue

print("\n✅ All folders now contain up to 30 unique images.")