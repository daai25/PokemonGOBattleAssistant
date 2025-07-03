from PIL import Image
import os
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_dir, "dataset_pokemon_images")
output_folder = os.path.join(script_dir, "compressed_dataset_pokemon_images")

compression_quality = 60  # JPEG quality (0-100, lower = smaller file)

for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            input_path = os.path.join(root, file)

            relative_path = os.path.relpath(root, input_folder)
            target_dir = os.path.join(output_folder, relative_path)
            os.makedirs(target_dir, exist_ok=True)

            # Change extension to .jpg if necessary
            output_filename = os.path.splitext(file)[0] + ".jpg"
            output_path = os.path.join(target_dir, output_filename)

            try:
                img = Image.open(input_path)

                # Convert RGBA/P to RGB (handle transparency)
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGBA")
                    # Fill transparent background with white
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.getchannel("A") if "A" in img.getbands() else None)
                    img = background
                else:
                    img = img.convert("RGB")

                # Save as JPEG
                img.save(output_path, "JPEG", optimize=True, quality=compression_quality)

                print(f"✅ {input_path} → {output_path}")

            except Exception as e:
                print(f"⚠️ Error compressing {input_path}: {e}")
                try:
                    shutil.copy2(input_path, output_path)
                    print(f"📁 Copied without compression: {input_path} → {output_path}")
                except Exception as copy_err:
                    print(f"❌ Failed to copy {input_path}: {copy_err}")