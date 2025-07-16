import os
import hashlib
from PIL import Image, UnidentifiedImageError

# 📂 Pfad zu deinem Dataset (ANPASSEN!)
dataset_dir = "C:/Users/dylan/Documents/Dev/PokemonGOBattleAssistant/data_acquisition/image_dataset/last_dataset"

# 🎯 Ziel-Format für alle Bilder
target_format = "jpeg"

def calculate_image_hash(img, hash_algo="sha256"):
    """
    Berechne einen Hash für den Bildinhalt.
    """
    hash_func = hashlib.new(hash_algo)
    # Bild in Bytes schreiben
    with Image.open(img) as image:
        image_bytes = image.tobytes()
        hash_func.update(image_bytes)
    return hash_func.hexdigest()

def convert_image_to_jpg_and_rename(file_path):
    try:
        with Image.open(file_path) as img:
            # PNGs mit Transparenz in RGB umwandeln und Hintergrund weiß füllen
            if img.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])  # Nutze Alpha-Channel als Maske
                img = background
            else:
                img = img.convert("RGB")  # Alle anderen in RGB konvertieren

            # Temporär im Speicher speichern, um Hash zu berechnen
            temp_path = f"{file_path}.temp.jpg"
            img.save(temp_path, format="JPEG", quality=95)

            # Hash des Bildes berechnen
            image_hash = calculate_image_hash(temp_path)
            new_file_name = f"{image_hash}.jpg"
            new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)

            # Endgültig unter neuem Namen speichern
            os.replace(temp_path, new_file_path)

            # Alte Datei löschen, falls notwendig
            if file_path != new_file_path and os.path.exists(file_path):
                os.remove(file_path)

            print(f"✔ Converted & Renamed: {file_path} -> {new_file_path}")

    except (UnidentifiedImageError, OSError) as e:
        print(f"✘ Corrupted or unsupported file removed: {file_path} ({e})")
        os.remove(file_path)

def process_dataset_to_jpg_and_rename(dataset_dir):
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            file_path = os.path.join(root, file)
            convert_image_to_jpg_and_rename(file_path)

if __name__ == "__main__":
    print(f"🔄 Starting conversion and renaming of all images in: {dataset_dir}")
    process_dataset_to_jpg_and_rename(dataset_dir)
    print("✅ All images converted and renamed with hash.")