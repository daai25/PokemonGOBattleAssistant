import requests
import os
import json
import hashlib
import time
from pathlib import Path

# 📁 Ordner des Skripts ermitteln
script_dir = Path(__file__).resolve().parent

# 📦 Pokémon-Namen laden (ohne Shadow-Formen)
json_path = script_dir / "pokemon_name_rank.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    pokemon_names = [
        entry["name"]
        for entry in data
        if "shadow" not in entry["name"].lower()
    ]

# 🗂 Speicherort: compressed_dataset_pokemon_images_cleaned_v2
clear_folder = script_dir / "compressed_dataset_pokemon_images_cleaned_v2"
clear_folder.mkdir(parents=True, exist_ok=True)

sprite_sets = [
    "red-blue", "yellow", "silver", "bank", "go", "home", "x-y", "sun-moon",
    "sword-shield", "brilliant-diamond-shining-pearl", "legends-arceus",
    "scarlet-violet", "yellow", "black-white", "heartgold-soulsilver",
    "ultra-sun-ultra-moon"
]
variants = ["normal"]
base_url = "https://img.pokemondb.net/sprites/{set}/{variant}/{name}.png"
headers = {"User-Agent": "Mozilla/5.0"}

def format_name_for_url(name):
    """Formatiere Pokémon-Namen passend zur URL"""
    return (
        name.lower()
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("’", "")
        .replace("'", "")
        .replace("é", "e")
        .replace(" ", "-")
    )

def hash_url_to_filename(url):
    """Erzeuge Dateinamen aus SHA256-Hash der URL"""
    return hashlib.sha256(url.encode()).hexdigest() + ".png"

def download_sprite(name, sprite_set, variant):
    """Lade Sprite herunter und speichere es"""
    formatted_name = format_name_for_url(name)
    url = base_url.format(set=sprite_set, variant=variant, name=formatted_name)
    folder = clear_folder / name  # 📂 Speicherort pro Pokémon
    folder.mkdir(parents=True, exist_ok=True)
    filename = hash_url_to_filename(url)
    path = folder / filename

    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"✅ {name}: {sprite_set}/{variant} gespeichert als {filename}")
        elif r.status_code == 404:
            print(f"❌ {name}: {sprite_set}/{variant} nicht gefunden (404)")
        else:
            print(f"⚠️ {name}: HTTP {r.status_code} für {sprite_set}/{variant}")
    except requests.Timeout:
        print(f"⏱️ Timeout bei {name}: {sprite_set}/{variant}")
    except requests.RequestException as e:
        print(f"⚠️ Fehler bei {name}: {sprite_set}/{variant} -> {e}")

# ▶ Hauptlogik
for idx, name in enumerate(pokemon_names, 1):
    print(f"\n🔄 [{idx}/{len(pokemon_names)}] Verarbeite: {name}")
    for sprite_set in sprite_sets:
        for variant in variants:
            download_sprite(name, sprite_set, variant)
    # Kurze Pause, um Server zu schonen
    time.sleep(0.2)