import os
from PIL import Image
import imagehash

script_dir = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.join(script_dir, "compressed_dataset_pokemon_images")

# Hash-Datenbank
hashes = {}

# Schwellenwert für Ähnlichkeit (0 = identisch, bis ca. 5-10 für sehr ähnlich)
threshold = 0

# Alle Bilder in Subfoldern durchlaufen
for dirpath, _, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(dirpath, filename)
            try:
                img = Image.open(filepath)
                img_hash = imagehash.phash(img)

                # Ähnliche Bilder finden
                found_duplicate = False
                for existing_hash, existing_file in hashes.items():
                    if abs(img_hash - existing_hash) <= threshold:
                        print(f"similar: {filepath} ≈ {existing_file}")
                        # Zum Löschen:
                        os.remove(filepath)
                        found_duplicate = True
                        break

                if not found_duplicate:
                    hashes[img_hash] = filepath

            except Exception as e:
                print(f"Error {filepath}: {e}")