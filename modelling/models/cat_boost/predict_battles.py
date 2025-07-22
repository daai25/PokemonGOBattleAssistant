"""
Predict a battle outcome between two Pokémon using the trained CatBoost model.
Works with all_overall_rankings_full_gl.csv (Gamemaster-derived, ALL CAPS moves).
"""

import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# ---------- Paths ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'catboost_model.cbm')
RANKINGS_CSV = os.path.join(ROOT, 'data_acquisition', 'vectorized_data', 'all_overall_rankings_full_gl.csv')

FAST_MOVE_MAP_CSV    = os.path.join(ROOT, 'data_acquisition', 'dictionarie', 'fast_move_to_number.csv')
CHARGED_MOVE_MAP_CSV = os.path.join(ROOT, 'data_acquisition', 'dictionarie', 'charged_move_to_number.csv')
TYPE_MAP_CSV         = os.path.join(ROOT, 'data_acquisition', 'dictionarie', 'type_to_number.csv')
PVPOKE_MOVES_CSV     = os.path.join(ROOT, 'data_acquisition', 'vectorized_data', 'pvpoke_moves.csv')

# ---------- Helpers ----------
def load_dict(path, key_col, value_col):
    df = pd.read_csv(path)
    return df.set_index(key_col)[value_col].to_dict()

def normalize_move_for_lookup(m):
    """
    Convert "TACKLE" or "POWER_WHIP" to "Tackle" / "Power Whip" so it matches the mapping CSVs / pvpoke_moves.csv.
    """
    if pd.isna(m):
        return 'none'
    m = str(m).strip()
    if m.lower() == 'none':
        return 'none'
    # from UPPER_UNDERSCORE to Title Case
    return m.replace('_', ' ').title()

def build_move_to_type_num():
    """move_name -> move_type_number (using pvpoke_moves.csv + type_to_number.csv)"""
    type_map = pd.read_csv(TYPE_MAP_CSV)
    type2num = {row['Type'].capitalize(): int(row['Number']) for _, row in type_map.iterrows()}

    mv = pd.read_csv(PVPOKE_MOVES_CSV)
    mv['Type'] = mv['Type'].str.capitalize()
    # Normalize move names the same way we’ll normalize incoming names
    mv['MoveNorm'] = mv['Move'].apply(lambda x: normalize_move_for_lookup(x))
    return {r['MoveNorm']: type2num.get(r['Type'], 0) for _, r in mv.iterrows()}

def get_row(df, name):
    row = df[df['Pokemon'].str.lower() == name.lower()]
    if row.empty:
        raise ValueError(f"Pokémon '{name}' not found in rankings CSV.")
    return row.iloc[0]

def build_feature_row(left_row, right_row, fast_map, charged_map, move2type_num):
    # Types are already capitalized in CSV; keep as-is but handle 'None'
    def fix_type(t):
        return 'None' if pd.isna(t) or t == 'None' else str(t)

    l_t1, l_t2 = fix_type(left_row['Type 1']), fix_type(left_row['Type 2'])
    r_t1, r_t2 = fix_type(right_row['Type 1']), fix_type(right_row['Type 2'])

    # Raw move names from CSV (ALL CAPS with underscores)
    l_fast_raw  = str(left_row['Fast Move'])
    l_c1_raw    = str(left_row['Charged Move 1'])
    l_c2_raw    = str(left_row['Charged Move 2'])
    r_fast_raw  = str(right_row['Fast Move'])
    r_c1_raw    = str(right_row['Charged Move 1'])
    r_c2_raw    = str(right_row['Charged Move 2'])

    # Normalize to Title Case w/ spaces
    l_fast_name = normalize_move_for_lookup(l_fast_raw)
    l_c1_name   = normalize_move_for_lookup(l_c1_raw)
    l_c2_name   = normalize_move_for_lookup(l_c2_raw)
    r_fast_name = normalize_move_for_lookup(r_fast_raw)
    r_c1_name   = normalize_move_for_lookup(r_c1_raw)
    r_c2_name   = normalize_move_for_lookup(r_c2_raw)

    # Map to numeric IDs (same encoding as training)
    l_fast_id = fast_map.get(l_fast_name, fast_map.get('none', 0))
    l_c1_id   = charged_map.get(l_c1_name, charged_map.get('none', 0))
    l_c2_id   = charged_map.get(l_c2_name, charged_map.get('none', 0))
    r_fast_id = fast_map.get(r_fast_name, fast_map.get('none', 0))
    r_c1_id   = charged_map.get(r_c1_name, charged_map.get('none', 0))
    r_c2_id   = charged_map.get(r_c2_name, charged_map.get('none', 0))

    # Move type numbers
    l_fast_type = move2type_num.get(l_fast_name, 0)
    l_c1_type   = move2type_num.get(l_c1_name,   0)
    l_c2_type   = move2type_num.get(l_c2_name,   0)
    r_fast_type = move2type_num.get(r_fast_name, 0)
    r_c1_type   = move2type_num.get(r_c1_name,   0)
    r_c2_type   = move2type_num.get(r_c2_name,   0)

    # Ratios
    attack_ratio  = left_row['Attack']  / (right_row['Attack']  + 1e-5)
    defense_ratio = left_row['Defense'] / (right_row['Defense'] + 1e-5)
    stamina_ratio = left_row['Stamina'] / (right_row['Stamina'] + 1e-5)

    data = {
        'left_pokemon_type_1': l_t1,
        'left_pokemon_type_2': l_t2,
        'right_pokemon_type_1': r_t1,
        'right_pokemon_type_2': r_t2,

        'left_pokemon_fast_move': l_fast_id,
        'left_pokemon_charge_move_1': l_c1_id,
        'left_pokemon_charge_move_2': l_c2_id,
        'left_pokemon_fast_move_type': l_fast_type,
        'left_pokemon_charge_move_1_type': l_c1_type,
        'left_pokemon_charge_move_2_type': l_c2_type,

        'right_pokemon_fast_move': r_fast_id,
        'right_pokemon_charge_move_1': r_c1_id,
        'right_pokemon_charge_move_2': r_c2_id,
        'right_pokemon_fast_move_type': r_fast_type,
        'right_pokemon_charge_move_1_type': r_c1_type,
        'right_pokemon_charge_move_2_type': r_c2_type,

        'attack_ratio': attack_ratio,
        'defense_ratio': defense_ratio,
        'stamina_ratio': stamina_ratio,
    }

    X = pd.DataFrame([data])

    # Cast categoricals
    cat_cols = [
        'left_pokemon_type_1','left_pokemon_type_2','right_pokemon_type_1','right_pokemon_type_2',
        'left_pokemon_fast_move','left_pokemon_charge_move_1','left_pokemon_charge_move_2',
        'left_pokemon_fast_move_type','left_pokemon_charge_move_1_type','left_pokemon_charge_move_2_type',
        'right_pokemon_fast_move','right_pokemon_charge_move_1','right_pokemon_charge_move_2',
        'right_pokemon_fast_move_type','right_pokemon_charge_move_1_type','right_pokemon_charge_move_2_type'
    ]
    for c in cat_cols:
        X[c] = X[c].astype('category')

    return X

def predict_battle(left_name: str, right_name: str):
    # Load model
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    # Data & mappings
    df = pd.read_csv(RANKINGS_CSV)
    fast_map    = load_dict(FAST_MOVE_MAP_CSV,    'Fast_Move',   'Number')   # "Tackle" -> id
    charged_map = load_dict(CHARGED_MOVE_MAP_CSV, 'Charged_Move','Number')
    move2type   = build_move_to_type_num()

    left_row  = get_row(df, left_name)
    right_row = get_row(df, right_name)

    X_pair = build_feature_row(left_row, right_row, fast_map, charged_map, move2type)

    # Reindex to model features (CatBoost stores them)
    feats = model.feature_names_
    X_pair = X_pair.reindex(columns=feats, fill_value=0)

    pred = model.predict(X_pair)[0]
    proba = model.predict_proba(X_pair)[0]

    winner = left_name if int(pred) == 1 else right_name
    conf = max(proba) * 100
    return winner, proba, conf

def main():
    print("Pokemon GO Battle Predictor (CatBoost)")
    print("=" * 35)
    left = input("First Pokémon: ")
    right = input("Second Pokémon: ")
    try:
        winner, proba, conf = predict_battle(left, right)
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"\nPredicted winner: {winner}")
    print(f"Confidence: {conf:.2f}%")
    print(f"Probabilities (left wins, right wins): ({proba[0]:.3f}, {proba[1]:.3f})")

if __name__ == "__main__":
    main()
