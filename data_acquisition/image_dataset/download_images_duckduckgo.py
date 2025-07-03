import os
import json
import time
import hashlib
from pathlib import Path
from duckduckgo_search import DDGS
import requests

# Load Pokémon names (excluding Shadow forms)
json_path = Path(__file__).parent / "pokemon_name_rank.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    pokemon_names = [
        entry["name"]
        for entry in data
        if "shadow" not in entry["name"].lower()
    ]

# 🗂 Target directory on Desktop
desktop_path = Path.home() / "Desktop" / "pokemon_sprites"
os.makedirs(desktop_path, exist_ok=True)

def hash_url_to_filename(url):
    """Generate a unique filename from the image URL"""
    return hashlib.sha256(url.encode()).hexdigest() + ".jpg"

def download_image(url, folder):
    """Download a single image and save it to the folder"""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            filename = hash_url_to_filename(url)
            path = folder / filename
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"🖼️ Saved image: {filename}")
    except Exception as e:
        print(f"[!] Error downloading {url}: {e}")

def fetch_images(name, max_results=30):
    folder = desktop_path / name
    os.makedirs(folder, exist_ok=True)

    query = f"{name} pokemon"
    print(f"\n🔍 Searching images for: {query}")
    try:
        with DDGS() as ddgs:
            results = ddgs.images(query, max_results=max_results)
            for result in results:
                if "image" in result:
                    download_image(result["image"], folder)
                time.sleep(0.5)
    except Exception as e:
        print(f"[!] Skipping {name} due to error: {e}")
        time.sleep(30)  # wait before next Pokémon

start_from = "Marowak"
start = False

for name in pokemon_names:
    if not start:
        if name.lower() == start_from.lower():
            start = True
        else:
            continue  # skip until we reach the start_from Pokémon

    fetch_images(name, max_results=30)
    time.sleep(10)