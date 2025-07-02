from selenium import webdriver
from bs4 import BeautifulSoup
import time
import random
import pandas
import os
import csv

from pokemon_list import pokemon_go #imports pokemon list

# The CSV filename
filename = "poke_battles.csv"
# with open(filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Winner', 'Loser', 'Battle Time'])

# Function to write a row to CSV (creates file with header if it doesn't exist)
def write_battle_result(winner, loser):
    # Check if file exists to decide whether to write header
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # Write header only if file is new
        if not file_exists:
            writer.writerow(["Winner", "Loser"])
        
        # Write the data row
        writer.writerow([winner, loser])




# print(len(pokemon_go))

driver = webdriver.Chrome()
pokeOne = random.choice(pokemon_go).lower()
pokeTwo = random.choice(pokemon_go).lower()
print(pokeOne)
print(pokeTwo)

poke_url = 'https://pvpoke.com/battle/'
print(poke_url)
driver.get(poke_url)
# time.sleep(3)  # wait for page to load
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

# Find the first Pokemon search input (adjust the selector to match the real element)
search1 = driver.find_element("css selector", "input[placeholder='Search name']")  
search1.send_keys(pokeOne)
time.sleep(1)  # pause to let suggestions load if any
search1.send_keys("\n")  # hit Enter to select the Pokemon

# Find the second Pokemon search input (you’ll need to adjust selector if there are multiple inputs with same placeholder)
search2 = driver.find_elements("css selector", "input[placeholder='Search name']")[1]
search2.send_keys(pokeTwo)
time.sleep(1)
search2.send_keys("\n")
# Find and click the Battle button (example selector, adjust as needed)


#SOMETHING IS WRONG HERE
time.sleep(15) 
battle_button = driver.find_element("css selector", "button.battle-btn.button")
battle_button.click()
time.sleep(15)  # wait for battle result page to load




# Now soup includes JavaScript-rendered content
name = soup.find('div', class_='battle-summary-line')


if name.find('wins') == -1: #pokeOne has lost
    loser = pokeOne
    winner = pokeTwo
else: #pokeOne has Won
    loser = pokeTwo
    winner = pokeOne

#<div class="battle-summary-line"><span class="name">Mamoswine</span> wins in <span class="time">20.5s</span> with a battle rating of <span class="rating close-loss" style="background-color: rgb(136, 43, 145);"><span></span>354</span></div>
print(f'\n{name}\n')

write_battle_result(winner, loser)

driver.quit()


