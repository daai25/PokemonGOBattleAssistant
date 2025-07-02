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

poke_url = 'https://pvpoke.com/battle/1500/' + pokeOne + '/' + pokeTwo + '/11/0-1-1/0-1-1/'
print(poke_url)
driver.get(poke_url)
time.sleep(5)

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')


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


