from selenium import webdriver
import time
import random
import csv
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import TimeoutException

from pokemon_list import pokemon_go  # pokemon_go (list)
# pokemon_data = open("pokemon_data.csv", mode='r')
# battle_data = open("battle_data.csv", mode='a', newline="")
# writer = csv.writer("battle_data.csv")
driver = webdriver.Chrome()
driver.get("https://pvpoke.com/battle/")

def write_battle_data(winner, loser):
    # Find data from winner and loser in pokemon_data
    # copy it and paste it into battle data spots
    ### CHECK IF WINNER HAS ALL THE DATA ACTUALLY NEEDED. IF NOT, WE WILL NEED TO SCRAPE SOME DATA FROM THE HTML.
    
    df = pd.read_csv("cp1500_all_overall_rankings.csv")

    winner_row = df[df["Pokemon"].str.lower() == winner.lower()]
    loser_row = df[df["Pokemon"].str.lower() == loser.lower()]

        # Extract first matching row
    w = winner_row.iloc[0]
    l = loser_row.iloc[0]

    output_row = [
        w["Type 1"],
        w["Type 2"],
        w["Fast Move"],
        w["Charged Move 1"],
        w["Charged Move 2"],
        None,
        None,
        None,
        w["Dex"],
        w["Attack"],
        w["Defense"],
        w["Stamina"],
        w["Score"],

        l["Type 1"],
        l["Type 2"],
        l["Fast Move"],
        l["Charged Move 1"],
        l["Charged Move 2"],
        None,
        None,
        None,
        l["Dex"],
        l["Attack"],
        l["Defense"],
        l["Stamina"],
        l["Score"]
    ]

    with open('poke_battles.csv', 'a', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(output_row)
        print(f"Battle data between {winner} and {loser} written to poke_battles.csv")

def battle_simulator(pokeOne, pokeTwo):
    if count == 0:
        driver.get("https://pvpoke.com/battle/")
        driver.implicitly_wait(10) # Wait for page to load

    # Enter first Pokémon
    search1 = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[placeholder='Search name']"))
    )
    search1.clear()
    search1.send_keys(pokeOne)
    WebDriverWait(driver, 10).until(
        lambda d: d.find_elements(By.CLASS_NAME, 'poke-stats')[0].value_of_css_property("display") == "block"
    )

    # Enter second Pokémon (second input)
    search2 = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "input[placeholder='Search name']"))
    )[1]
    search2.clear()
    search2.send_keys(pokeTwo)
    WebDriverWait(driver, 10).until(
        lambda d: d.find_elements(By.CLASS_NAME, 'poke-stats')[1].value_of_css_property("display") == "block"
    )

    # Click battle button
    battle_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button.battle-btn.button"))
    )
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button.battle-btn.button"))
    )
    try:
        battle_button.click()
    except:
        print('excepted1')
        time.sleep(1)
        battle_button.click()

    html = driver.page_source

    if html.find('wins') != -1:  # pokeOne wins
        return pokeOne, pokeTwo
    else: # pokeTwo wins
        return pokeTwo, pokeOne

#Accept cookies only once
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
count = 0
for i in range(15000):
    print('battle_',count)
    pokeOne = random.choice(pokemon_go).lower()
    pokeTwo = random.choice(pokemon_go).lower()
    winner, loser = battle_simulator(pokeOne, pokeTwo)
    write_battle_data(winner, loser)
    # time.sleep(2)
    count += 1

time.sleep(5)
driver.quit()
# battle_data.close()
# pokemon_data.close()