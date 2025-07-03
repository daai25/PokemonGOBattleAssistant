import os
import json
import time
import hashlib
from pathlib import Path
from duckduckgo_search import DDGS
import requests

# Target directory (e.g., Desktop/pokemon_sprites)
base_path = Path.home() / "Desktop" / "pokemon_sprites"
os.makedirs(base_path, exist_ok=True)

# Load Pokémon names (excluding Shadow forms)
json_path = Path(__file__).parent / "pokemon_name_rank.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    all_pokemon_names = [
        entry["name"]
        for entry in data
        if "shadow" not in entry["name"].lower()
    ]


def hash_url_to_filename(url):
    """Generate a unique filename from the image URL"""
    return hashlib.sha256(url.encode()).hexdigest() + ".jpg"


def download_image(url, folder):
    """Download a single image and save it to the folder"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            filename = hash_url_to_filename(url)
            file_path = folder / filename
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"🖼️ Saved image: {filename}")
    except Exception as e:
        print(f"⚠️ Error downloading {url}: {e}")


def fetch_images(name, folder, max_results=30):
    """Search and download images for a specific Pokémon"""
    print(f"\n🔍 Searching images for: {name}")

    try:
        with DDGS() as ddgs:
            results = ddgs.images(f"{name} pokemon", max_results=max_results)
            for result in results:
                if "image" in result:
                    download_image(result["image"], folder)
                time.sleep(0.5)  # Prevent hitting rate limits
    except Exception as e:
        print(f"⚠️ Error while fetching images for {name}: {e}")
        time.sleep(30)


# ▶ Main logic
for folder in base_path.iterdir():
    if folder.is_dir():
        file_count = len(list(folder.iterdir()))
        print(f"📂 Folder: {folder.name} → {file_count} files")

        if file_count < 15:
            print(f"⚡ Less than 15 files in {folder.name}. Starting image download...")
            pokemon_name = folder.name

            # Check if folder name matches a known Pokémon name
            if pokemon_name in all_pokemon_names:
                fetch_images(pokemon_name, folder, max_results=30)
                time.sleep(10)  # Delay between Pokémon
            else:
                print(f"❓ {pokemon_name} is not a known Pokémon name. Skipping.")

print("\n✅ All folders checked. Missing images downloaded.")