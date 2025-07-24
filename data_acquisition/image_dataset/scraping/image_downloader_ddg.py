from duckduckgo_search import DDGS
import requests
import os
import time
import random
import hashlib
from PIL import Image
from io import BytesIO
import imagehash
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
BASE_FOLDER = "last_dataset"             # Folder with Pokémon subfolders
TARGET_IMAGE_COUNT = 30            # Images per subfolder
JPEG_QUALITY = 50                  # JPEG quality (percent)
SIMILARITY_THRESHOLD = 5           # Max Hamming distance for perceptual hash
# ----------------------

def calculate_sha256(image_content):
    """Calculate SHA256 hash of image bytes."""
    return hashlib.sha256(image_content).hexdigest()

def calculate_perceptual_hash(image_content):
    """Calculate perceptual hash (phash) for image bytes."""
    img = Image.open(BytesIO(image_content)).convert("RGB")
    return imagehash.phash(img)

def hamming_distance(hash1, hash2):
    """Compute Hamming distance between two perceptual hashes."""
    return hash1 - hash2

def get_existing_hashes_and_phashes(folder_path):
    """Load SHA256 and perceptual hashes for all existing images in a folder."""
    sha256_hashes = set()
    phashes = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.jpg'):
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                    sha256_hashes.add(calculate_sha256(content))
                    img = Image.open(BytesIO(content)).convert("RGB")
                    phashes.append(imagehash.phash(img))
            except Exception as e:
                print(f"[!] Error reading {file_name}: {e}")
    return sha256_hashes, phashes

def save_as_jpeg(image_content, save_path, quality=JPEG_QUALITY):
    """Convert image to JPEG and save with specified quality."""
    try:
        img = Image.open(BytesIO(image_content)).convert("RGB")
        img.save(save_path, "JPEG", quality=quality)
        return True
    except Exception as e:
        print(f"[!] Error converting/saving as JPEG: {e}")
        return False

def process_folder(folder_path, folder_name, target_count, similarity_threshold):
    """Process one folder: download until it contains target_count unique images."""
    print(f"[THREAD] Starting processing for '{folder_name}'")
    existing_sha256, existing_phashes = get_existing_hashes_and_phashes(folder_path)
    current_count = len(existing_sha256)

    with DDGS() as ddgs:
        while current_count < target_count:
            needed = target_count - current_count
            print(f"[>] {folder_name}: {current_count}/{target_count} unique images. Downloading {needed} more...")

            results = ddgs.images(folder_name, max_results=needed * 3)
            next_index = current_count + 1

            for idx, result in enumerate(results, 1):
                if len(existing_sha256) >= target_count:
                    break  # Enough images

                image_url = result["image"]
                print(f"[.] {folder_name} image {idx}: {image_url}")

                try:
                    # Random delay (1–3 seconds)
                    wait_time = random.uniform(1, 3)
                    print(f"[-] {folder_name}: Waiting {wait_time:.2f}s before download...")
                    time.sleep(wait_time)

                    response = requests.get(image_url, timeout=10)
                    response.raise_for_status()
                    image_content = response.content

                    # Check SHA256 for duplicates
                    sha256_hash = calculate_sha256(image_content)
                    if sha256_hash in existing_sha256:
                        print(f"[=] {folder_name}: Image {idx} already exists (SHA256 match). Skipping.")
                        continue

                    # Check perceptual hash similarity
                    new_phash = calculate_perceptual_hash(image_content)
                    if any(hamming_distance(new_phash, ph) <= similarity_threshold for ph in existing_phashes):
                        print(f"[=] {folder_name}: Image {idx} too similar to existing images. Skipping.")
                        continue

                    # Save image as JPEG
                    filename = os.path.join(folder_path, f"{folder_name}_{next_index}.jpg")
                    if save_as_jpeg(image_content, filename, quality=JPEG_QUALITY):
                        print(f"[+] {folder_name}: Saved JPEG (quality={JPEG_QUALITY}%): {filename}")
                        existing_sha256.add(sha256_hash)
                        existing_phashes.append(new_phash)
                        next_index += 1
                        current_count += 1
                    else:
                        print(f"[!] {folder_name}: Failed to save image {idx}")

                except Exception as e:
                    print(f"[!] {folder_name}: Error downloading image {idx}: {e}")

            # Random delay (5–10 seconds) between rounds
            if current_count < target_count:
                wait_time = random.uniform(5, 10)
                print(f"[-] {folder_name}: Waiting {wait_time:.2f}s before next round...")
                time.sleep(wait_time)

    print(f"[✓] {folder_name}: Complete – {target_count} unique images now available.")

def fill_folders_with_images(base_folder, target_count, similarity_threshold):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, base_folder)

    if not os.path.exists(images_dir):
        print(f"[!] Folder '{images_dir}' does not exist.")
        return

    print(f"[+] Scanning '{images_dir}' for subfolders...")
    subfolders = [name for name in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, name))]
    if not subfolders:
        print("[!] No subfolders found. Please create folders for Pokémon names.")
        return

    max_workers = os.cpu_count()
    print(f"[+] Using up to {max_workers} threads based on CPU cores.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for folder_name in subfolders:
            folder_path = os.path.join(images_dir, folder_name)
            futures.append(executor.submit(process_folder, folder_path, folder_name, target_count, similarity_threshold))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[!] Error in thread: {e}")

    print("\n[✓] All folders now contain at least {target_count} unique and diverse JPEG images.")

# --- MAIN ---
if __name__ == "__main__":
    fill_folders_with_images(BASE_FOLDER, TARGET_IMAGE_COUNT, SIMILARITY_THRESHOLD)