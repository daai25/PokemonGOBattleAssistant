from selenium import webdriver
from bs4 import BeautifulSoup
import time
import random
import pandas
import os
import csv


pokemon_go = [
    "Clodsire", "Diggersby", "Forretress", "Lapras", "Jellicent", "Corsola (Galarian)",
    "Dusclops", "Samurott", "Gligar", "Clodsire", "Feraligatr", "Golisopod", "Mandibuzz",
    "Corviknight", "Gastrodon", "Steelix", "Stunfisk", "Dusknoir", "Carbink", "Dedenne",
    "Morpeko (Full Belly)", "Togedemaru", "Cradily", "Marowak", "Primeape", "Furret",
    "Pangoro", "Bastiodon", "Emolga", "Runerigus", "Azumarill", "Grumpig", "Quagsire",
    "Sableye", "Bellibolt", "Dragalge", "Malamar", "Araquanid", "Claydol", "Metang",
    "Moltres (Galarian)", "Tinkaton", "Toxapex", "Aurorus", "Togetic", "Weezing (Galarian)",
    "Golurk", "Jumpluff", "Marowak (Alolan)", "Scizor", "Blastoise", "Clefable", "Lickitung",
    "Annihilape", "Bronzong", "Sandslash (Alolan)", "Victreebel", "Golisopod", "Lurantis",
    "Swampert", "Whiscash", "Greninja", "Machamp", "Vespiquen", "Guzzlord", "Registeel",
    "Tinkatuff", "Dewott", "Armarouge", "Linoone", "Drapion", "Tentacruel", "Gliscor",
    "Hakamo-o", "Raichu", "Wigglytuff", "Dunsparce", "Medicham", "Goodra", "Dachsbun",
    "Stunfisk (Galarian)", "Swalot", "Talonflame", "Froslass", "Pachirisu", "Wormadam (Trash)",
    "Lanturn", "Lickilicky", "Hippopotas", "Lairon", "Nidoqueen", "Spiritomb", "Ninetales",
    "Regirock", "Qwilfish", "Thievul", "Skeledirge", "Electrode (Hisuian)", "Ninetales (Alolan)",
    "Barbaracle", "Hippowdon", "Gallade", "Sealeo", "Serperior", "Zapdos", "Amoonguss", "Klefki",
    "Toxicroak", "Dewgong", "Farfetch'd (Galarian)", "Donphan", "Dragonair", "Toxtricity",
    "Magnezone", "Walrein", "Farfetch'd", "Florges", "Magcargo", "Bellossom", "Mew",
    "Sirfetch'd", "Ledian", "Electivire", "Kommo-o", "Skarmory", "Dragonite",
    "Castform (Sunny)", "Genesect", "Zweilous", "Arctibax", "Charjabug", "Qwilfish (Hisuian)",
    "Urshifu (Single Strike)", "Chesnaught", "Gogoat", "Ursaring", "Venusaur", "Gourgeist (Super)",
    "Marshtomp", "Meganium", "Umbreon", "Glalie", "Piloswine", "Crustle", "Salazzle",
    "Machoke", "Miltank", "Altaria", "Castform (Rainy)", "Giratina (Origin)", "Spinda",
    "Gallade", "Sandslash", "Cofagrigus", "Tropius", "Magneton", "Raichu (Alolan)", "Dragapult",
    "Mightyena", "Cetoddle", "Nidorina", "Oranguru", "Wailmer", "Whimsicott", "Ferrothorn",
    "Gourgeist (Large)", "Mantine", "Suicune", "Cresselia", "Trevenant", "Aggron", "Relicanth",
    "Wailord", "Gengar", "Lileep", "Bombirdier", "Fletchinder", "Gourgeist (Average)", "Greedent",
    "Overqwil", "Tauros (Aqua)", "Wartortle", "Abomasnow", "Castform", "Cetitan", "Pawniard",
    "Typhlosion", "Lokix", "Roserade", "Samurott (Hisuian)", "Decidueye", "Haunter", "Leavanny",
    "Kingambit", "Perrserker", "Skuntank", "Beedrill", "Empoleon", "Ariados", "Hariyama",
    "Hydreigon", "Poliwrath", "Kricketune", "Venomoth", "Drampa", "Gourgeist (Small)", "Scrafty",
    "Starmie", "Grimer (Alolan)", "Gyarados", "Avalugg", "Ferrothorn", "Raikou", "Scyther",
    "Drifloon", "Galvantula", "Litleo", "Servine", "Castform (Snowy)", "Pelipper", "Dugtrio (Alolan)",
    "Magnemite", "Flygon", "Pawmot", "Turtonator", "Golett", "Frillish", "Furfrou", "Grimer",
    "Hawlucha", "Pidgeot", "Oinkologne (Female)", "Dubwool", "Pumpkaboo (Large)", "Raticate (Alolan)",
    "Regice", "Arcanine", "Bewear", "Sneasler", "Charizard", "Machop", "Lucario", "Pawmo",
    "Zangoose", "Mantyke", "Rapidash (Galarian)", "Sylveon", "Tapu Fini", "Ampharos",
    "Aromatisse", "Rapidash", "Rhyperior", "Grafaiai", "Politoed", "Mawile", "Typhlosion (Hisuian)",
    "Kangaskhan", "Sceptile", "Vaporeon", "Escavalier", "Bibarel", "Kleavor", "Magmar",
    "Munchlax", "Centiskorch", "Golem (Alolan)", "Pinsir", "Armaldo", "Celesteela", "Crocalor",
    "Mamoswine", "Noctowl", "Druddigon", "Heliolisk", "Staravia", "Dugtrio", "Magmortar",
    "Klang", "Muk", "Passimian", "Rhydon", "Slowking (Galarian)", "Parasect", "Excadrill",
    "Primarina", "Tauros (Combat)", "Fraxure", "Heatran", "Kecleon", "Mienshao", "Obstagoon",
    "Seaking", "Tyrunt", "Articuno (Galarian)", "Palossand", "Sandshrew (Alolan)", "Togekiss",
    "Vullaby", "Deoxys (Defense)", "Zygarde", "Golbat", "Staraptor", "Sandshrew",
    "Dhelmise", "Braixen", "Gholdengo", "Girafarig", "Drilbur", "Hypno", "Ivysaur", "Lunatone",
    "Jirachi", "Incineroar", "Slowbro (Galarian)", "Hitmontop", "Muk (Alolan)", "Pyroar",
    "Bruxish", "Golem", "Lugia", "Scolipede", "Baxcalibur", "Solrock", "Chansey", "Chimecho",
    "Dartrix", "Granbull", "Quilladin", "Alomomola"
]



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


