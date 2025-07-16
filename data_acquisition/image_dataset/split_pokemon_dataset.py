import os
import shutil
import random

# Get the current directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Set source and target folders
source_folder = os.path.join(base_dir, "last_dataset") # Replace 'pokemon_folder' with your actual folder name
target_folder = os.path.join(base_dir, "final_pokemon_dataset")

train_ratio = 0.8 # 80% for training, 20% for testing

# Create target train and test folders
train_folder = os.path.join(target_folder, "train")
test_folder = os.path.join(target_folder, "test")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Loop over all Pokémon subfolders
for pokemon_name in os.listdir(source_folder):
    pokemon_path = os.path.join(source_folder, pokemon_name)
    if not os.path.isdir(pokemon_path):
        continue # Skip files, only process folders

    # Get list of all image files
    images = [f for f in os.listdir(pokemon_path) if os.path.isfile(os.path.join(pokemon_path, f))]
    random.shuffle(images)

    # Split into train and test sets
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    # Create destination folders for this Pokémon
    train_pokemon_folder = os.path.join(train_folder, pokemon_name)
    test_pokemon_folder = os.path.join(test_folder, pokemon_name)
    os.makedirs(train_pokemon_folder, exist_ok=True)
    os.makedirs(test_pokemon_folder, exist_ok=True)

    # Copy images to their respective folders
    for img in train_images:
        shutil.copy(os.path.join(pokemon_path, img), os.path.join(train_pokemon_folder, img))
    for img in test_images:
        shutil.copy(os.path.join(pokemon_path, img), os.path.join(test_pokemon_folder, img))

print("Dataset successfully split into training and test sets!")