import os
import requests
from bs4 import BeautifulSoup
import time
import re

# Basis-URL der Website
BASE_URL = "https://www.coloringpages101.com"
START_URL = f"{BASE_URL}/Video-Games/Pokemon-coloring-pages/pg-"

# Dein Zielordner mit den Pokémon-Unterordnern
TARGET_DIR = r"C:\Users\dylan\Documents\Dev\PokemonGOBattleAssistant\data_acquisition\image_dataset\last_dataset"

# Alle Unterordnernamen aus dem Zielpfad holen
pokemon_folders = {folder for folder in os.listdir(TARGET_DIR) if os.path.isdir(os.path.join(TARGET_DIR, folder))}
print(f"{len(pokemon_folders)} Pokémon-Ordner gefunden.")

# Hilfsfunktion: Seite laden und parsen
def fetch_soup(url):
    response = requests.get(url)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")

# Hilfsfunktion: Alle PNGs von einer Detailseite laden
def download_images(pokemon_name, detail_url):
    if pokemon_name not in pokemon_folders:
        print(f"Überspringe {pokemon_name}: Kein Ordner vorhanden.")
        return

    folder_path = os.path.join(TARGET_DIR, pokemon_name)
    print(f"✔ Lade Bilder für {pokemon_name}")

    detail_soup = fetch_soup(detail_url)
    png_links = detail_soup.find_all('a', string="Download PNG")

    for idx, link in enumerate(png_links, start=1):
        img_url = BASE_URL + link.get('href')
        filename = os.path.join(folder_path, f"{idx}.png")

        if os.path.exists(filename):
            print(f"   Bild {idx} existiert schon, überspringe.")
            continue

        try:
            img_data = requests.get(img_url).content
            with open(filename, "wb") as f:
                f.write(img_data)
            print(f"   Gespeichert: {filename}")
            time.sleep(1)  # kleine Pause zwischen Downloads
        except Exception as e:
            print(f"   Fehler beim Speichern von {filename}: {e}")

# Scraper-Logik
def scrape_pokemon_pages(max_pages=50):
    for page_num in range(1, max_pages + 1):
        page_url = START_URL + str(page_num)
        print(f"Lade Index-Seite {page_num}")
        soup = fetch_soup(page_url)

        links = soup.select('a[href*="/Pokemon-coloring-pages/"]')
        for link in links:
            href = link.get('href')
            if not href: continue

            # Pokémon-Name aus Linktext extrahieren
            name_match = re.search(r"-([^-]+?)-Pokemon-coloring-page", href)
            if not name_match: continue

            pokemon_name = name_match.group(1).replace("-", " ").title()

            # Spezialfälle: Galarian, Hisuian, etc. erkennen
            if "galarian" in href.lower():
                pokemon_name += " (Galarian)"
            if "hisuian" in href.lower():
                pokemon_name += " (Hisuian)"

            # Detailseite besuchen und Bilder laden
            detail_url = BASE_URL + href
            download_images(pokemon_name, detail_url)

if __name__ == "__main__":
    scrape_pokemon_pages(max_pages=50)  # passe Anzahl Seiten bei Bedarf an