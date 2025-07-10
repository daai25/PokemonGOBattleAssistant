import pandas as pd

# === Load main battle data ===
df = pd.read_csv("poke_battles_organized_not_digit.csv")

# === Load mapping CSVs into dictionaries ===
type_map = pd.read_csv("type_to_number.csv")
type_map_dict = dict(zip(type_map["Type"].str.strip().str.lower(), type_map["Number"]))

fast_move_map = pd.read_csv("fast_move_to_number.csv")
fast_move_map_dict = dict(zip(fast_move_map["Fast_Move"].str.strip().str.lower(), fast_move_map["Number"]))

charged_move_map = pd.read_csv("charged_move_to_number.csv")
charged_move_map_dict = dict(zip(charged_move_map["Charged_Move"].str.strip().str.lower(), charged_move_map["Number"]))

# === Clean and lower-case all string columns for consistent mapping ===
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.strip().str.lower()

# === Define which columns map to which dictionaries ===
type_columns = [
    "left_pokemon_type_1", "left_pokemon_type_2",
    "left_pokemon_fast_move_type", "left_pokemon_charge_move_1_type", "left_pokemon_charge_move_2_type",
    "right_pokemon_type_1", "right_pokemon_type_2",
    "right_pokemon_fast_move_type", "right_pokemon_charge_move_1_type", "right_pokemon_charge_move_2_type"
]

fast_move_columns = ["left_pokemon_fast_move", "right_pokemon_fast_move"]
charged_move_columns = [
    "left_pokemon_charge_move_1", "left_pokemon_charge_move_2",
    "right_pokemon_charge_move_1", "right_pokemon_charge_move_2"
]

# === Apply mapping to each group of columns ===
for col in type_columns:
    df[col] = df[col].map(type_map_dict)

for col in fast_move_columns:
    df[col] = df[col].map(fast_move_map_dict)

for col in charged_move_columns:
    df[col] = df[col].map(charged_move_map_dict)

# === Optional: Save the transformed data ===
df.to_csv("battle_data_numeric.csv", index=False)

# === Check sample output ===
print(df.head())
