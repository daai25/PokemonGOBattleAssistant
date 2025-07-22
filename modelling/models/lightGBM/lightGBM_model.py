
"""
LightGBM model for Pokemon GO Battle predictions with Type Effectiveness features
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

# Set paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BATTLE_DATA_PATH = os.path.join(ROOT_DIR, 'data_acquisition', 'vectorized_data', 'poke_battles.csv')
# We no longer need a separate models directory for the plot

# Explicit mapping from numeric codes to type names
NUM_TO_TYPE = {
    0: 'None', 1: 'Bug', 2: 'Dark', 3: 'Dragon', 4: 'Electric', 5: 'Fairy',
    6: 'Fighting', 7: 'Fire', 8: 'Flying', 9: 'Ghost', 10: 'Grass',
    11: 'Ground', 12: 'Ice', 13: 'Normal', 14: 'Poison', 15: 'Psychic',
    16: 'Rock', 17: 'Steel', 18: 'Water'
}

# Pokemon GO type effectiveness multipliers
TYPE_CHART = {
    'Normal': {'Rock':0.625, 'Ghost':0, 'Steel':0.625},
    'Fire':   {'Grass':1.6,'Ice':1.6,'Bug':1.6,'Steel':1.6,'Fire':0.625,'Water':0.625,'Rock':0.625,'Dragon':0.625},
    'Water':  {'Fire':1.6,'Ground':1.6,'Rock':1.6,'Water':0.625,'Grass':0.625,'Dragon':0.625},
    'Electric':{'Water':1.6,'Flying':1.6,'Electric':0.625,'Grass':0.625,'Dragon':0.625,'Ground':0},
    'Grass':  {'Water':1.6,'Ground':1.6,'Rock':1.6,'Fire':0.625,'Grass':0.625,'Poison':0.625,'Flying':0.625,'Bug':0.625,'Dragon':0.625,'Steel':0.625},
    'Ice':    {'Grass':1.6,'Ground':1.6,'Flying':1.6,'Dragon':1.6,'Fire':0.625,'Water':0.625,'Ice':0.625,'Steel':0.625},
    'Fighting':{'Normal':1.6,'Ice':1.6,'Rock':1.6,'Dark':1.6,'Steel':1.6,'Poison':0.625,'Flying':0.625,'Psychic':0.625,'Bug':0.625,'Fairy':0.625,'Ghost':0},
    'Poison': {'Grass':1.6,'Fairy':1.6,'Poison':0.625,'Ground':0.625,'Rock':0.625,'Ghost':0.625,'Steel':0},
    'Ground': {'Fire':1.6,'Electric':1.6,'Poison':1.6,'Rock':1.6,'Steel':1.6,'Grass':0.625,'Bug':0.625,'Flying':0,'Ice':0.625},
    'Flying': {'Grass':1.6,'Fighting':1.6,'Bug':1.6,'Electric':0.625,'Rock':0.625,'Steel':0.625},
    'Psychic':{'Fighting':1.6,'Poison':1.6,'Psychic':0.625,'Steel':0.625,'Dark':0},
    'Bug':    {'Grass':1.6,'Psychic':1.6,'Dark':1.6,'Fire':0.625,'Fighting':0.625,'Poison':0.625,'Flying':0.625,'Ghost':0.625,'Steel':0.625,'Fairy':0.625},
    'Rock':   {'Fire':1.6,'Ice':1.6,'Flying':1.6,'Bug':1.6,'Fighting':0.625,'Ground':0.625,'Steel':0.625},
    'Ghost':  {'Psychic':1.6,'Ghost':1.6,'Dark':0.625,'Normal':0,'Fairy':0.625},
    'Dragon': {'Dragon':1.6,'Steel':0.625,'Fairy':0},
    'Dark':   {'Psychic':1.6,'Ghost':1.6,'Fighting':0.625,'Dark':0.625,'Fairy':0.625},
    'Steel':  {'Ice':1.6,'Rock':1.6,'Fairy':1.6,'Fire':0.625,'Water':0.625,'Electric':0.625,'Steel':0.625},
    'Fairy':  {'Fighting':1.6,'Dragon':1.6,'Dark':1.6,'Fire':0.625,'Poison':0.625,'Steel':0.625}
}

def load_data():
    """Load battle data CSV"""
    print(f"Loading battle data from: {BATTLE_DATA_PATH}")
    df = pd.read_csv(BATTLE_DATA_PATH)
    print(f"Battle data shape: {df.shape}")
    return df

def get_effectiveness(move_type, def_t1, def_t2):
    """Compute effectiveness multiplier"""
    e1 = TYPE_CHART.get(move_type, {}).get(def_t1, 1)
    e2 = TYPE_CHART.get(move_type, {}).get(def_t2, 1)
    return e1 * e2

def prepare_battle_features(df):
    """Create features including ratios and type-effectiveness"""
    df = df.dropna()
    # Map numeric type codes to names
    for col in ['left_pokemon_type_1','left_pokemon_type_2','right_pokemon_type_1','right_pokemon_type_2']:
        df[col] = df[col].map(NUM_TO_TYPE)
    # Ratio features
    df['attack_ratio'] = df['left_pokemon_attack'] / (df['right_pokemon_attack'] + 1e-5)
    df['defense_ratio'] = df['left_pokemon_defense'] / (df['right_pokemon_defense'] + 1e-5)
    df['stamina_ratio'] = df['left_pokemon_stamina'] / (df['right_pokemon_stamina'] + 1e-5)
    # Effectiveness features
    df['left_fast_eff']    = df.apply(lambda r: get_effectiveness(r['left_pokemon_fast_move_type'],    r['right_pokemon_type_1'], r['right_pokemon_type_2']), axis=1)
    df['left_charge1_eff'] = df.apply(lambda r: get_effectiveness(r['left_pokemon_charge_move_1_type'], r['right_pokemon_type_1'], r['right_pokemon_type_2']), axis=1)
    df['left_charge2_eff'] = df.apply(lambda r: get_effectiveness(r['left_pokemon_charge_move_2_type'], r['right_pokemon_type_1'], r['right_pokemon_type_2']), axis=1)
    df['right_fast_eff']   = df.apply(lambda r: get_effectiveness(r['right_pokemon_fast_move_type'],   r['left_pokemon_type_1'],  r['left_pokemon_type_2']),  axis=1)
    df['right_charge1_eff']= df.apply(lambda r: get_effectiveness(r['right_pokemon_charge_move_1_type'],r['left_pokemon_type_1'],  r['left_pokemon_type_2']),  axis=1)
    df['right_charge2_eff']= df.apply(lambda r: get_effectiveness(r['right_pokemon_charge_move_2_type'],r['left_pokemon_type_1'],  r['left_pokemon_type_2']),  axis=1)
    # Aggregate
    df['left_eff_avg']  = df[['left_fast_eff','left_charge1_eff','left_charge2_eff']].mean(axis=1)
    df['right_eff_avg'] = df[['right_fast_eff','right_charge1_eff','right_charge2_eff']].mean(axis=1)
    df['efficacy_diff']= df['left_eff_avg'] - df['right_eff_avg']
    # One-hot encode categories
    cat_cols = [
        'left_pokemon_type_1','left_pokemon_type_2',
        'left_pokemon_fast_move','left_pokemon_charge_move_1','left_pokemon_charge_move_2',
        'left_pokemon_fast_move_type','left_pokemon_charge_move_1_type','left_pokemon_charge_move_2_type',
        'right_pokemon_type_1','right_pokemon_type_2',
        'right_pokemon_fast_move','right_pokemon_charge_move_1','right_pokemon_charge_move_2',
        'right_pokemon_fast_move_type','right_pokemon_charge_move_1_type','right_pokemon_charge_move_2_type'
    ]
    exclude = [
        'pokemon_winner','pokemon_loser','right_pokemon_dex','left_pokemon_dex',
        'left_pokemon_overall','right_pokemon_overall',
        'left_pokemon_attack','right_pokemon_attack',
        'left_pokemon_defense','right_pokemon_defense',
        'left_pokemon_stamina','right_pokemon_stamina'
    ]
    df = pd.get_dummies(df, columns=cat_cols)
    X = df.drop(columns=['winner']+exclude)
    y = df['winner']
    return X,y

def train_and_evaluate(X,y):
    """Train and evaluate LightGBM"""
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = LGBMClassifier(n_estimators=300,learning_rate=0.1,subsample=0.8,subsample_freq=1,random_state=42)
    print("Training LightGBM model with Type Effectiveness...")
    model.fit(X_tr_s,y_tr)

    y_pred = model.predict(X_te_s)
    acc = accuracy_score(y_te,y_pred)
    print(f"Test accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_te,y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_te,y_pred))

    fi = pd.DataFrame({'feature':X.columns,'importance':model.feature_importances_}).sort_values('importance',ascending=False)
    print("\nTop 10 Feature Importance:")
    print(fi.head(10))

    # Save plot in same folder as this script
    plot_path = os.path.join(os.path.dirname(__file__), 'feature_importance.png')
    plt.figure(figsize=(12,8))
    plt.barh(fi['feature'][:15], fi['importance'][:15])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved feature importance plot to: {plot_path}")

def main():
    print("Starting LightGBM model with Type Effectiveness...")
    df = load_data()
    X,y = prepare_battle_features(df)
    train_and_evaluate(X,y)
    print("Training and evaluation complete.")

if __name__=='__main__':
    main()

