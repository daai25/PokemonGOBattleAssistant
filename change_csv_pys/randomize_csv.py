import pandas as pd
import random

df = pd.read_csv("poke_battles.csv") #change this to your file with the old format of scraping
new_rows = []

for _, row in df.iterrows():
    if random.randint(0, 1) == 0:
        row_dict = {
            "left_pokemon_type_1": row["winner_type_1"],
            "left_pokemon_type_2": row["winner_type_2"],
            "left_pokemon_fast_move": row["winner_fast_move"],
            "left_pokemon_charge_move_1": row["winner_charge_move_1"],
            "left_pokemon_charge_move_2": row["winner_charge_move_2"],
            "left_pokemon_fast_move_type": row["winner_fast_move_type"],
            "left_pokemon_charge_move_1_type": row["winner_charge_move_1_type"],
            "left_pokemon_charge_move_2_type": row["winner_charge_move_2_type"],
            "left_pokemon_dex": row["winner_dex"],
            "left_pokemon_attack": row["winner_attack"],
            "left_pokemon_defense": row["winner_defense"],
            "left_pokemon_stamina": row["winner_stamina"],
            "left_pokemon_overall": row["winner_overall"],

            "right_pokemon_type_1": row["loser_type_1"],
            "right_pokemon_type_2": row["loser_type_2"],
            "right_pokemon_fast_move": row["loser_fast_move"],
            "right_pokemon_charge_move_1": row["loser_charge_move_1"],
            "right_pokemon_charge_move_2": row["loser_charge_move_2"],
            "right_pokemon_fast_move_type": row["loser_fast_move_type"],
            "right_pokemon_charge_move_1_type": row["loser_charge_move_1_type"],
            "right_pokemon_charge_move_2_type": row["loser_charge_move_2_type"],
            "right_pokemon_dex": row["loser_dex"],
            "right_pokemon_attack": row["loser_attack"],
            "right_pokemon_defense": row["loser_defense"],
            "right_pokemon_stamina": row["loser_stamina"],
            "right_pokemon_overall": row["loser_overall"],

            "winner": 1
        }
    else:
        row_dict = {
            "left_pokemon_type_1": row["loser_type_1"],
            "left_pokemon_type_2": row["loser_type_2"],
            "left_pokemon_fast_move": row["loser_fast_move"],
            "left_pokemon_charge_move_1": row["loser_charge_move_1"],
            "left_pokemon_charge_move_2": row["loser_charge_move_2"],
            "left_pokemon_fast_move_type": row["loser_fast_move_type"],
            "left_pokemon_charge_move_1_type": row["loser_charge_move_1_type"],
            "left_pokemon_charge_move_2_type": row["loser_charge_move_2_type"],
            "left_pokemon_dex": row["loser_dex"],
            "left_pokemon_attack": row["loser_attack"],
            "left_pokemon_defense": row["loser_defense"],
            "left_pokemon_stamina": row["loser_stamina"],
            "left_pokemon_overall": row["loser_overall"],

            "right_pokemon_type_1": row["winner_type_1"],
            "right_pokemon_type_2": row["winner_type_2"],
            "right_pokemon_fast_move": row["winner_fast_move"],
            "right_pokemon_charge_move_1": row["winner_charge_move_1"],
            "right_pokemon_charge_move_2": row["winner_charge_move_2"],
            "right_pokemon_fast_move_type": row["winner_fast_move_type"],
            "right_pokemon_charge_move_1_type": row["winner_charge_move_1_type"],
            "right_pokemon_charge_move_2_type": row["winner_charge_move_2_type"],
            "right_pokemon_dex": row["winner_dex"],
            "right_pokemon_attack": row["winner_attack"],
            "right_pokemon_defense": row["winner_defense"],
            "right_pokemon_stamina": row["winner_stamina"],
            "right_pokemon_overall": row["winner_overall"],

            "winner": 0
        }

    new_rows.append(row_dict)

final_df = pd.DataFrame(new_rows)
final_df.to_csv("poke_battles_final.csv", index=False)
print("Final dataset saved to poke_battles_final.csv")
