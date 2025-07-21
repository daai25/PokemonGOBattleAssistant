from selenium import webdriver
import time
import random
import csv
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import StaleElementReferenceException
import os

from pokemon_list import pokemon_go

class BattleTimeout(Exception):
    pass

# === Load mapping CSVs ===
type_map = pd.read_csv("type_to_number.csv")
type_map_dict = dict(zip(type_map["Type"].str.strip().str.lower(), type_map["Number"]))

fast_move_map = pd.read_csv("fast_move_to_number.csv")
fast_move_map_dict = dict(zip(
    fast_move_map["Fast_Move"].str.strip().str.lower(),
    fast_move_map["Number"]
))

charged_move_map = pd.read_csv("charged_move_to_number.csv")
charged_move_map_dict = dict(zip(
    charged_move_map["Charged_Move"].str.strip().str.lower(),
    charged_move_map["Number"]
))

# === Load pvpoke_moves for move type correction ===
pvpoke_moves = pd.read_csv('pvpoke_moves.csv', header=None, names=[
    "Move","Type","Category","Power","Energy","DPE","Time","Cooldown","BuffChance"
])
move_to_type = {
    str(row['Move']).strip().lower(): str(row['Type']).strip().lower()
    for _, row in pvpoke_moves.iterrows()
}

# === Load Pokémon stats CSV ===
stats_df = pd.read_csv("cp1500_all_overall_rankings.csv")
stats_df["Pokemon"] = stats_df["Pokemon"].str.strip().str.lower()
stats_dict = stats_df.set_index("Pokemon").to_dict(orient="index")

def get_type_num(type_name):
    if pd.isna(type_name) or type_name == "":
        return 0
    return type_map_dict.get(type_name.strip().lower(), 0)

def get_fast_move_num(move_name):
    if pd.isna(move_name) or move_name == "":
        return 0
    return fast_move_map_dict.get(move_name.strip().lower(), 0)

def get_charged_move_num(move_name):
    if pd.isna(move_name) or move_name == "":
        return 0
    return charged_move_map_dict.get(move_name.strip().lower(), 0)

def get_move_type_num(move_name):
    if pd.isna(move_name) or move_name == "":
        return 0
    base = move_name.replace("*","").replace("†","").strip().lower()
    t = move_to_type.get(base, 'none')
    return type_map_dict.get(t, 0)

def get_pokemon_stats(poke_name):
    key = poke_name.strip().lower()
    row = stats_dict.get(key)
    if row is None:
        raise ValueError(f"Pokémon '{poke_name}' not found in stats CSV.")
    return {
        "type_1": get_type_num(row["Type 1"]),
        "type_2": get_type_num(row["Type 2"]),
        "fast_move": get_fast_move_num(row["Fast Move"]),
        "charge_move_1": get_charged_move_num(row["Charged Move 1"]),
        "charge_move_2": get_charged_move_num(row["Charged Move 2"]),
        "fast_move_name": row["Fast Move"],
        "charge_move_1_name": row["Charged Move 1"],
        "charge_move_2_name": row["Charged Move 2"],
        "dex": row["Dex"],
        "attack": row["Attack"],
        "defense": row["Defense"],
        "stamina": row["Stamina"],
        "overall": row["Score"]
    }

def get_battle_moves_types(driver):
    fast_els = driver.find_elements(By.CLASS_NAME, 'fast')
    fast_types = [e.get_attribute("class").split()[-1] for e in fast_els]
    chg_els = driver.find_elements(By.CLASS_NAME, 'charged')
    chg_types = [e.get_attribute("class").split()[-1] for e in chg_els]
    return fast_types, chg_types

def battle_simulator(pokeOne, pokeTwo):
    driver.get("https://pvpoke.com/battle/")
    driver.implicitly_wait(10)

    # pick pokeOne
    inp = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR,"input[placeholder='Search name']"))
    )
    inp.clear(); inp.send_keys(pokeOne)
    WebDriverWait(driver,10).until(
        lambda d: d.find_elements(By.CLASS_NAME,'poke-stats')[0]
                  .value_of_css_property("display")=="block"
    )

    # pick pokeTwo
    inp2 = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR,"input[placeholder='Search name']"))
    )[1]
    inp2.clear(); inp2.send_keys(pokeTwo)
    WebDriverWait(driver,10).until(
        lambda d: d.find_elements(By.CLASS_NAME,'poke-stats')[1]
                  .value_of_css_property("display")=="block"
    )

    # click battle
    btn = WebDriverWait(driver,10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,"button.battle-btn.button"))
    )
    try:
        btn.click()
    except:
        time.sleep(1)
        btn.click()

    # — wait for the battle summary line to appear —
    try:
        summary_el = WebDriverWait(driver, 20).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR,
                ".battle-results.single .summary .battle-summary-line"))
        )
    except TimeoutException:
        # failed to get a summary → skip this battle
        raise BattleTimeout(f"Battle {pokeOne} vs {pokeTwo} timed out waiting for summary")
    
    for _ in range(3):
        try:
            name_el = summary_el.find_element(By.CSS_SELECTOR, ".name")
            break
        except StaleElementReferenceException:
            time.sleep(0.1)
            summary_el = driver.find_element(By.CSS_SELECTOR, ".battle-summary-line")
    else:
        raise BattleTimeout(f"Battle {pokeOne} vs {pokeTwo} stale summary element")
    
    # get summary text and name
    summary_text = summary_el.text.strip()
    name_el = summary_el.find_element(By.CSS_SELECTOR, ".name")
    name = name_el.text.strip()

    # determine winner/loser
    lower = summary_text.lower()
    if "wins" in lower:
        winner, loser = name, (pokeTwo if name.lower()==pokeOne.lower() else pokeOne)
    elif "loses" in lower:
        loser, winner = name, (pokeTwo if name.lower()==pokeOne.lower() else pokeOne)
    else:
        raise RuntimeError(f"Unexpected summary text: '{summary_text}'")

    # now grab move‐type classes
    fast_types, chg_types = get_battle_moves_types(driver)

    if winner.lower() == pokeOne.lower():
        w_fast, l_fast = fast_types[0], fast_types[1]
        w_chg,   l_chg   = chg_types[0:2], chg_types[2:4]
    else:
        w_fast, l_fast = fast_types[1], fast_types[0]
        w_chg,   l_chg   = chg_types[2:4], chg_types[0:2]

    return {
        "winner":            winner,
        "loser":             loser,
        "winner_fast_type":  w_fast,
        "winner_chg_types":  w_chg,
        "loser_fast_type":   l_fast,
        "loser_chg_types":   l_chg
    }

def make_output_row(left_stats, right_stats, left_types, right_types, winner_side, winner_name, loser_name):
    return [
        winner_name, loser_name,
        left_stats["type_1"], left_stats["type_2"],
        left_stats["fast_move"], left_stats["charge_move_1"], left_stats["charge_move_2"],
        left_types[0], left_types[1], left_types[2],
        left_stats["dex"], left_stats["attack"], left_stats["defense"],
        left_stats["stamina"], left_stats["overall"],
        right_stats["type_1"], right_stats["type_2"],
        right_stats["fast_move"], right_stats["charge_move_1"], right_stats["charge_move_2"],
        right_types[0], right_types[1], right_types[2],
        right_stats["dex"], right_stats["attack"], right_stats["defense"],
        right_stats["stamina"], right_stats["overall"],
        winner_side
    ]

def get_move_types(stats):
    return [
        get_move_type_num(stats["fast_move_name"]),
        get_move_type_num(stats["charge_move_1_name"]),
        get_move_type_num(stats["charge_move_2_name"])
    ]

# setup driver & accept cookies
driver = webdriver.Chrome()
driver.get("https://pvpoke.com/battle/")
try:
    btn = WebDriverWait(driver,5).until(EC.element_to_be_clickable((By.CLASS_NAME,"ncmp__btn")))
    btn.click()
    btn2 = WebDriverWait(driver,5).until(EC.element_to_be_clickable((By.XPATH,"//button[text()='Accept All']")))
    btn2.click()
except TimeoutException:
    pass

# prepare CSV
header = [
    "pokemon_winner","pokemon_loser",
    "left_pokemon_type_1","left_pokemon_type_2",
    "left_pokemon_fast_move","left_pokemon_charge_move_1","left_pokemon_charge_move_2",
    "left_pokemon_fast_move_type","left_pokemon_charge_move_1_type","left_pokemon_charge_move_2_type",
    "left_pokemon_dex","left_pokemon_attack","left_pokemon_defense","left_pokemon_stamina","left_pokemon_overall",
    "right_pokemon_type_1","right_pokemon_type_2",
    "right_pokemon_fast_move","right_pokemon_charge_move_1","right_pokemon_charge_move_2",
    "right_pokemon_fast_move_type","right_pokemon_charge_move_1_type","right_pokemon_charge_move_2_type",
    "right_pokemon_dex","right_pokemon_attack","right_pokemon_defense","right_pokemon_stamina","right_pokemon_overall",
    "winner"
]
file_exists = os.path.isfile('poke_battles.csv')
with open('poke_battles.csv','a',newline='',encoding='utf-8') as f:
    w = csv.writer(f)
    if not file_exists or os.path.getsize('poke_battles.csv') == 0:
        w.writerow(header)

    for count in range(10000):
        pokeOne = random.choice(pokemon_go)
        pokeTwo = random.choice(pokemon_go)
        while pokeOne == pokeTwo:
            pokeTwo = random.choice(pokemon_go)

        # handle potential battle simulation errors by skipping battle if fails
        try:
            battle = battle_simulator(pokeOne, pokeTwo)
        except BattleTimeout as e:
            print("⚠️", e)
            continue
        except Exception as e:
            print(f"⚠️ Unexpected Error in battle simulation: {e}")
            continue

        left_stats  = get_pokemon_stats(pokeOne)
        right_stats = get_pokemon_stats(pokeTwo)
        left_types  = get_move_types(left_stats)
        right_types = get_move_types(right_stats)

        # manual check for first 10
        if count < 5:
            print(f"\n=== Battle {count}: {pokeOne} vs {pokeTwo} ===")
            print(" Winner:", battle["winner"])
            print("  Left stats:", left_stats, "move‑types:", left_types)
            print("  Right stats:", right_stats, "move‑types:", right_types)
            ok = input("Does this look right? (y/n): ")
            if not ok.lower().startswith('y'):
                print(" SKIPPING this battle.\n")
                continue

        # pokeOne is always left, pokeTwo always right
        winner_col = 1 if battle["winner"] == pokeOne else 0

        row = make_output_row(
            left_stats, right_stats,
            left_types, right_types,
            winner_col,
            battle["winner"], battle["loser"]
        )
        w.writerow(row)
        time.sleep(1)
        print(f"Battle {count} completed: {pokeOne} vs {pokeTwo} - Winner: {battle['winner']}")

driver.quit()
