import os

# 📂 Pfad zum Dataset (ANPASSEN!)
dataset_dir = "C:/Users/dylan/Documents/Dev/PokemonGOBattleAssistant/data_acquisition/image_dataset/last_dataset"

# 🖼 Optional: Zähle nur bestimmte Bildformate
valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

# ⬆️ Sortierreihenfolge: True = absteigend (größte zuerst), False = aufsteigend
sort_descending = True

def count_files_in_subfolders(directory, extensions=None):
    folder_counts = {}

    for root, dirs, files in os.walk(directory):
        # Nur direkt im aktuellen Ordner zählen (nicht rekursiv)
        if root == directory:
            for subfolder in dirs:
                subfolder_path = os.path.join(root, subfolder)
                file_count = 0

                for file in os.listdir(subfolder_path):
                    if os.path.isfile(os.path.join(subfolder_path, file)):
                        if extensions:
                            if os.path.splitext(file)[1].lower() in extensions:
                                file_count += 1
                        else:
                            file_count += 1

                folder_counts[subfolder] = file_count

    return folder_counts

if __name__ == "__main__":
    counts = count_files_in_subfolders(dataset_dir, extensions=valid_extensions)

    # 📊 Sortiere nach Anzahl Dateien
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=sort_descending)

    print("\n📊 Anzahl Dateien pro Subfolder (sortiert):")
    for folder, count in sorted_counts:
        print(f"📁 {folder}: {count} Dateien")

    total_files = sum(counts.values())
    print(f"\n✅ Gesamtzahl aller Dateien: {total_files}")