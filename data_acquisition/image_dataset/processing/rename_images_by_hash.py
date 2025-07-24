import os
from PIL import Image
import imagehash

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, "last_dataset")

image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def rename_images_with_hash(root_directory):
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            if is_image_file(filename):
                full_path = os.path.join(dirpath, filename)
                try:
                    with Image.open(full_path) as img:
                        # Berechne phash
                        hash_value = str(imagehash.phash(img))
                        # Erstelle neuen Dateinamen mit gleicher Extension
                        new_filename = f"{hash_value}{os.path.splitext(filename)[1]}"
                        new_full_path = os.path.join(dirpath, new_filename)
                        
                        # Prüfe ob Datei mit Hash bereits existiert
                        if not os.path.exists(new_full_path):
                            os.rename(full_path, new_full_path)
                            print(f"Umbenannt: {filename} -> {new_filename}")
                        else:
                            print(f"Übersprungen (existiert schon): {new_filename}")
                except Exception as e:
                    print(f"Fehler bei {full_path}: {e}")

if __name__ == "__main__":
    rename_images_with_hash(root_dir)