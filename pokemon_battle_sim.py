from selenium import webdriver
import time
import pandas as pd
import ast
import random
import os
import csv
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import TimeoutException
import re


from pokemon_list import pokemon_go  # pokemon_go (list)
filename = "poke_battles.csv"

# Edit the data inside the dictionaries in the csv file
def edit_moves_data(pokemon, series, new_dict):
    df = pd.read_csv(filename)
    row_index = df[df['Pokemon'] == pokemon].index[0]
    move_dict = ast.literal_eval(df.at[row_index, series])
    move_dict.update(new_dict)
    df.at[row_index, series] = str(move_dict)
    df.to_csv(filename, index=False)

def create_move_dicts(win_fast_move, lose_fast_move, win_chg_moves, lose_chg_moves, bonus_index):
    # Turn moves and bonus index into dictionaries 
        win_chg_dict = {}
        win_fast_dict = {}
        lose_chg_dict = {}
        lose_fast_dict = {}
        if win_chg_moves:
            for move in win_chg_moves:
                win_chg_dict[move] = bonus_index
            win_fast_dict = {win_fast_move: bonus_index}
        else:
            for move in lose_chg_moves:
                lose_chg_dict[move] = bonus_index
            lose_fast_dict = {lose_fast_move: bonus_index}
        return win_fast_dict, lose_fast_dict, win_chg_dict, lose_chg_dict

def write_battle_data(pokemon, win_types, lose_types, win_fast_move, lose_fast_move, win_chg_moves, lose_chg_moves, bonus_index):
    dicts = create_move_dicts(win_fast_move, lose_fast_move, win_chg_moves, lose_chg_moves, bonus_index)
    df = pd.read_csv(filename)
    row_index = df[df['Pokemon'] == pokemon].index[0]
    if win_types:
        old_list = ast.literal_eval(df.at[row_index, 'WinTypes'])
        updated_list = old_list + win_types
        df.at[row_index, 'WinTypes'] = str(updated_list)
        df.to_csv(filename, index=False)
    else:
        old_list = ast.literal_eval(df.at[row_index, 'LoseTypes'])
        updated_list = old_list + lose_types
        df.at[row_index, 'LoseTypes'] = str(updated_list)
        df.to_csv(filename, index=False)

    if win_fast_move:
        edit_moves_data(pokemon, "WinFastMoves", dicts[0])
    else:
        edit_moves_data(pokemon, "LoseFastMoves", dicts[1])

    if win_chg_moves:
        edit_moves_data(pokemon, "WinChargedMoves", dicts[2])
    else:
        edit_moves_data(pokemon, "LoseChargedMoves", dicts[3])              

def write_battle_data_new_pokemon(pokemon, win_types, lose_types, win_fast_move, lose_fast_move, win_chg_moves, lose_chg_moves, bonus_index):
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Pokemon", "WinTypes", "LoseTypes", "WinFastMoves", "LoseFastMoves", "WinChargedMoves", "LoseChargedMoves"])
        dicts = create_move_dicts(win_fast_move, lose_fast_move, win_chg_moves, lose_chg_moves, bonus_index)
        writer.writerow([pokemon, win_types, lose_types, dicts[0], dicts[1], dicts[2], dicts[3]])
            
def battle_simulator(pokeOne, pokeTwo):
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

    time.sleep(5)

    html = driver.page_source


    # Get the types of both pokemon
    types = driver.find_elements(By.CLASS_NAME, 'types')
    pokeOne_types = []
    pokeTwo_types = []

    match = re.match(r'([A-Z][a-z]+)([A-Z].*)', types[0].text)
    if match:
        type1, type2 = match.groups()
        pokeOne_types = [type1, type2]
    else:
        pokeOne_types = [types[0].text]

    match = re.match(r'([A-Z][a-z]+)([A-Z].*)', types[1].text)
    if match:
        type3, type4 = match.groups()
        pokeTwo_types = [type3, type4]
    else:
        pokeTwo_types = [types[1].text]

    #print(pokeOne_types)
    #print(pokeTwo_types)

    # Get stats (atk, def, sta) from both pokemon
    stats = driver.find_elements(By.CLASS_NAME, 'stat-label')
    pokeOne_atk = stats[0].find_element(By.CLASS_NAME, 'stat').text
    pokeOne_def = stats[1].find_element(By.CLASS_NAME, 'stat').text
    pokeOne_sta = stats[2].find_element(By.CLASS_NAME, 'stat').text
    
    pokeTwo_atk = stats[4].find_element(By.CLASS_NAME, 'stat').text
    pokeTwo_def = stats[5].find_element(By.CLASS_NAME, 'stat').text
    pokeTwo_sta = stats[6].find_element(By.CLASS_NAME, 'stat').text

    #print(pokeOne_atk, pokeOne_def, pokeOne_sta)
    #print(pokeTwo_atk, pokeTwo_def, pokeTwo_sta)

    # Get fast moves for both pokemon
    fast_moves = driver.find_elements(By.CLASS_NAME, 'fast')
    pokeOne_fast = Select(fast_moves[0]).first_selected_option.text
    pokeTwo_fast = Select(fast_moves[1]).first_selected_option.text

    #print(pokeOne_fast)
    #print(pokeTwo_fast)

    # Get charged moves for both pokemon
    charged_moves = driver.find_elements(By.CLASS_NAME, 'charged')
    pokeOne_chg = [Select(charged_moves[0]).first_selected_option.text, Select(charged_moves[1]).first_selected_option.text]
    pokeTwo_chg = [Select(charged_moves[2]).first_selected_option.text, Select(charged_moves[3]).first_selected_option.text]

    #print(pokeOne_chg)
    #print(pokeTwo_chg)   

    if html.find('wins') != -1:  # pokeOne wins
        bonus_index = 1
        if pokeOne_atk <= pokeTwo_atk:
            bonus_index += 1
        if pokeOne_def <= pokeTwo_def:
            bonus_index += 1
        if pokeOne_sta <= pokeTwo_sta:
            bonus_index += 1
        
        return pokeOne, pokeOne_types, pokeOne_fast, pokeOne_chg, pokeTwo_types, pokeTwo_fast, pokeTwo_chg, bonus_index
    else: # pokeTwo wins
        bonus_index = 1
        if pokeTwo_atk <= pokeOne_atk:
            bonus_index += 1
        if pokeTwo_def <= pokeOne_def:
            bonus_index += 1
        if pokeTwo_sta <= pokeOne_sta:
            bonus_index += 1

        return pokeTwo, pokeOne_types, pokeOne_fast, pokeOne_chg, pokeTwo_types, pokeTwo_fast, pokeTwo_chg, bonus_index


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


for i in range(0, len(pokemon_go), 2):
    pokeOne = pokemon_go[i]
    pokeTwo = pokemon_go[(i + 1) % len(pokemon_go)]
    battle_data = battle_simulator(pokeOne, pokeTwo)
    if battle_data[0] == pokeOne:
        write_battle_data_new_pokemon(pokeOne, battle_data[4], [], battle_data[5], None, battle_data[6], None, battle_data[7])
        write_battle_data_new_pokemon(pokeTwo, [], battle_data[1], None, battle_data[2], None, battle_data[3], battle_data[7])
    else:
        write_battle_data_new_pokemon(pokeTwo, battle_data[1], [], battle_data[2], None, battle_data[3], None, battle_data[7])
        write_battle_data_new_pokemon(pokeOne, [], battle_data[4], None, battle_data[5], None, battle_data[6], battle_data[7])

for i in range(len(pokemon_go) * 20):
    pokeOne = random.choice(pokemon_go).lower()
    pokeTwo = random.choice(pokemon_go).lower()
    battle_data = battle_simulator(pokeOne, pokeTwo)
    if battle_data[0] == pokeOne:
        write_battle_data(pokeOne, battle_data[4], [], battle_data[5], None, battle_data[6], None, battle_data[7])
        write_battle_data(pokeTwo, [], battle_data[1], None, battle_data[2], None, battle_data[3], battle_data[7])
    else:
        write_battle_data(pokeTwo, battle_data[1], [], battle_data[2], None, battle_data[3], None, battle_data[7])
        write_battle_data(pokeOne, [], battle_data[4], None, battle_data[5], None, battle_data[6], battle_data[7])
  
driver.quit()

# Old code for getting the elemental types for moves
'''
# Get fast move types for both pokemon
move_select = driver.find_elements(By.CLASS_NAME, 'fast')
type_one = move_select[0].get_attribute("class").split()
pokeOne_fast_type = type_one[-1]
type_two = move_select[1].get_attribute("class").split()
pokeTwo_fast_type = type_two[-1]

print(pokeOne_fast_type)
print(pokeTwo_fast_type)

# Get charged move types for both pokemon
move_select = driver.find_elements(By.CLASS_NAME, 'charged')
type_one = move_select[0].get_attribute("class").split()
pokeOne_chg_type = type_one[-1]
type_two = move_select[1].get_attribute("class").split()
pokeTwo_fast_type = type_two[-1]

print(pokeOne_fast_type)
print(pokeTwo_fast_type)
''' 
