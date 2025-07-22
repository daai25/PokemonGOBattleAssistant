"""
CatBoost model for Pokemon GO Battle predictions
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier

# ---------- Paths ----------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BATTLE_DATA_PATH = os.path.join(ROOT_DIR, 'data_acquisition', 'vectorized_data', 'poke_battles.csv')

# ---------- Constants ----------
NUM_TO_TYPE = {
    0: 'None', 1: 'Bug', 2: 'Dark', 3: 'Dragon', 4: 'Electric', 5: 'Fairy',
    6: 'Fighting', 7: 'Fire', 8: 'Flying', 9: 'Ghost', 10: 'Grass',
    11: 'Ground', 12: 'Ice', 13: 'Normal', 14: 'Poison', 15: 'Psychic',
    16: 'Rock', 17: 'Steel', 18: 'Water'
}

# ---------- Data load ----------
def load_data():
    print(f"Loading battle data from: {BATTLE_DATA_PATH}")
    df = pd.read_csv(BATTLE_DATA_PATH)
    print(f"Battle data shape: {df.shape}")
    return df

# ---------- Feature prep ----------
def prepare_battle_features(df: pd.DataFrame):
    df = df.dropna()

    # Map numeric types to strings (categorical)
    type_cols = ['left_pokemon_type_1', 'left_pokemon_type_2',
                 'right_pokemon_type_1', 'right_pokemon_type_2']
    for c in type_cols:
        df[c] = df[c].map(NUM_TO_TYPE).astype('category')

    # Ratios
    df['attack_ratio']  = df['left_pokemon_attack']   / (df['right_pokemon_attack']   + 1e-5)
    df['defense_ratio'] = df['left_pokemon_defense']  / (df['right_pokemon_defense']  + 1e-5)
    df['stamina_ratio'] = df['left_pokemon_stamina']  / (df['right_pokemon_stamina']  + 1e-5)

    # CatBoost categorical columns
    categorical_columns = type_cols + [
        'left_pokemon_fast_move','left_pokemon_charge_move_1','left_pokemon_charge_move_2',
        'left_pokemon_fast_move_type','left_pokemon_charge_move_1_type','left_pokemon_charge_move_2_type',
        'right_pokemon_fast_move','right_pokemon_charge_move_1','right_pokemon_charge_move_2',
        'right_pokemon_fast_move_type','right_pokemon_charge_move_1_type','right_pokemon_charge_move_2_type'
    ]
    for c in categorical_columns:
        df[c] = df[c].astype('category')

    # Columns to drop from X
    exclude = [
        'pokemon_winner','pokemon_loser','right_pokemon_dex','left_pokemon_dex',
        'left_pokemon_overall','right_pokemon_overall',
        'left_pokemon_attack','right_pokemon_attack',
        'left_pokemon_defense','right_pokemon_defense',
        'left_pokemon_stamina','right_pokemon_stamina'
    ]

    X = df.drop(columns=['winner'] + exclude)
    y = df['winner']

    return X, y, categorical_columns

# ---------- Train & evaluate ----------
def train_and_evaluate(X, y, cat_features):
    # Split (you’re fine with 80/20 for this project)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # CatBoost prefers indices for cat_features
    cat_idx = [X.columns.get_loc(c) for c in cat_features]

    model = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        rsm=0.8,
        bagging_temperature=1.0,
        eval_metric='Accuracy',
        loss_function='Logloss',
        random_seed=42,
        use_best_model=True,
        verbose=200
    )

    model.fit(
        X_train, y_train,
        cat_features=cat_idx,
        eval_set=(X_test, y_test),            # fine for class project
        early_stopping_rounds=200
    )

    print(f"Best iter: {model.get_best_iteration()}  "
          f"Best val acc: {model.get_best_score()['validation']['Accuracy']:.4f}")

    # Final test metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance (normalized)
    imps = model.get_feature_importance()
    fi = pd.DataFrame({'feature': X.columns, 'importance': imps / imps.sum()})
    fi = fi.sort_values('importance', ascending=False)
    print("\nTop 10 Feature Importance (normalized):")
    print(fi.head(10))

    # Plot and save FI
    plt.figure(figsize=(12, 8))
    plt.barh(fi['feature'][:15], fi['importance'][:15])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importance (normalized)')
    plt.tight_layout()
    out_dir = os.path.dirname(__file__)
    plot_path = os.path.join(out_dir, 'feature_importance.png')
    plt.savefig(plot_path)
    print(f"Saved feature importance plot to: {plot_path}")

    # Save model
    model_path = os.path.join(out_dir, 'catboost_model.cbm')
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")

# ---------- Main ----------
if __name__ == '__main__':
    print("Starting CatBoost training...")
    df = load_data()
    X, y, cat_features = prepare_battle_features(df)
    train_and_evaluate(X, y, cat_features)
    print("Training and evaluation complete.")
