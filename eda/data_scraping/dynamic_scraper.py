from selenium import webdriver
from bs4 import BeautifulSoup
import time
import json

def scrape_rankings_names_scores_to_json():
    # Set up the Selenium WebDriver
    driver = webdriver.Chrome()

    try:
        # Navigate to the rankings page
        url = 'https://pvpoke.com/rankings/'
        driver.get(url)

        # Wait for the page to load completely
        time.sleep(15)

        # Get the page source and parse it with BeautifulSoup
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # Extract Pokémon rankings data
        pokemon_blocks = soup.find_all('div', class_='rank')
        data = []

        for pokemon in pokemon_blocks:
            name = pokemon.find('span', class_='name').text if pokemon.find('span', class_='name') else "No name"
            score = pokemon.find('div', class_='rating score-rating').text if pokemon.find('div', class_='rating score-rating') else "No score"

            data.append({"name": name, "score": score})

        # Filter out entries with 'No score'
        data = [entry for entry in data if entry['score'] != "No score"]

        # Save data to a JSON file in the raw_data folder
        output_path = 'raw_data/pokemon_name_rank.json'
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"Data has been saved to {output_path}")

    finally:
        # Quit the driver
        driver.quit()

# Example usage
scrape_rankings_names_scores_to_json()