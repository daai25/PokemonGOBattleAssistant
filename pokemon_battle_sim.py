from selenium import webdriver
import time
import random
import os
import csv
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

from pokemon_list import pokemon_go  # your Pokémon list

filename = "poke_battles.csv"

def write_battle_result(winner, loser):
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Winner", "Loser"])
        writer.writerow([winner, loser])

driver = webdriver.Chrome()
driver.get("https://pvpoke.com/battle/")

# Accept cookies only once
try:
    accept_button = WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable((By.CLASS_NAME, "ncmp__btn"))
    )
    accept_button.click()

    accept_all_button = WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable((By.XPATH, "//button[text()='Accept All']"))
    )
    accept_all_button.click()
except TimeoutException:
    print("No cookie popup or already accepted.")

for i in range(200):
    pokeOne = random.choice(pokemon_go).lower()
    pokeTwo = random.choice(pokemon_go).lower()
    print(f"Battle {i+1}: {pokeOne} vs {pokeTwo}")

    driver.get("https://pvpoke.com/battle/")
    time.sleep(3)  # wait for page to load

    # Enter first Pokémon
    search1 = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[placeholder='Search name']"))
    )
    search1.clear()
    search1.send_keys(pokeOne)
    time.sleep(1)
    search1.send_keys("\n")

    # Enter second Pokémon (second input)
    search2 = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "input[placeholder='Search name']"))
    )[1]
    search2.clear()
    search2.send_keys(pokeTwo)
    time.sleep(1)
    search2.send_keys("\n")

    # Click battle button
    battle_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button.battle-btn.button"))
    )
    battle_button.click()

    time.sleep(5)  # wait for result to load

    html = driver.page_source

    if html.find('wins') == -1:  # pokeOne lost
        loser = pokeOne
        winner = pokeTwo
    else:
        loser = pokeTwo
        winner = pokeOne

    write_battle_result(winner, loser)

driver.quit()
