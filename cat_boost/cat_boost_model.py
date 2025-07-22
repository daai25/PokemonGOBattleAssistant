"""
CatBoost model for Pokemon GO Battle predictions
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier

# Set paths
BATTLE_DATA_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')),
    'data_acquisition', 'vectorized_data', 'poke_battles.csv'
)

# Explicit mapping from numeric codes to type names
NUM_TO_TYPE = {
    0: 'None', 1: 'Bug', 2: 'Dark', 3: 'Dragon', 4: 'Electric', 5: 'Fairy',
    6: 'Fighting', 7: 'Fire', 8: 'Flying', 9: 'Ghost', 10: 'Grass',
    11: 'Ground', 12: 'Ice', 13: 'Normal', 14: 'Poison', 15: 'Psychic',
    16: 'Rock', 17: 'Steel', 18: 'Water'
}

# Load data
def load_data():
    print(f"Loading battle data from: {BATTLE_DATA_PATH}")
    df = pd.read_csv(BATTLE_DATA_PATH)
    print(f"Battle data shape: {df.shape}")
    return df

# Prepare features
def prepare_battle_features(df):
    df = df.dropna()
    # Map numeric types to category dtype
    type_cols = ['left_pokemon_type_1', 'left_pokemon_type_2',
                 'right_pokemon_type_1', 'right_pokemon_type_2']
    for col in type_cols:
        df[col] = df[col].map(NUM_TO_TYPE).astype('category')
    # Create ratio features
    df['attack_ratio']  = df['left_pokemon_attack']   / (df['right_pokemon_attack']   + 1e-5)
    df['defense_ratio'] = df['left_pokemon_defense']  / (df['right_pokemon_defense']  + 1e-5)
    df['stamina_ratio'] = df['left_pokemon_stamina']  / (df['right_pokemon_stamina']  + 1e-5)
    # Identify categoricals for CatBoost
    categorical_columns = type_cols + [
        'left_pokemon_fast_move','left_pokemon_charge_move_1','left_pokemon_charge_move_2',
        'left_pokemon_fast_move_type','left_pokemon_charge_move_1_type','left_pokemon_charge_move_2_type',
        'right_pokemon_fast_move','right_pokemon_charge_move_1','right_pokemon_charge_move_2',
        'right_pokemon_fast_move_type','right_pokemon_charge_move_1_type','right_pokemon_charge_move_2_type'
    ]
    for col in categorical_columns:
        df[col] = df[col].astype('category')
    # Exclude columns
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

# Train and evaluate with fixed best hyperparameters
def train_and_evaluate(X, y, cat_features):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Initialize CatBoost with best found params
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=50
    )
    print("Training CatBoost model with fixed best parameters...")
    model.fit(X_train, y_train, cat_features=cat_features)
    # Evaluate on test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    # Feature importance normalized
    importances = model.get_feature_importance()
    norm_imp = importances / importances.sum()
    fi = pd.DataFrame({'feature': X.columns, 'importance': norm_imp})
    fi = fi.sort_values('importance', ascending=False)
    print("\nTop 10 Feature Importance (normalized):")
    print(fi.head(10))
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(fi['feature'][:15], fi['importance'][:15])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importance (normalized)')
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'feature_importance.png')
    plt.savefig(plot_path)
    print(f"Saved feature importance plot to: {plot_path}")
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), 'catboost_model.cbm')
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")

# Main
if __name__ == '__main__':
    print("Starting CatBoost training with fixed hyperparameters...")
    df = load_data()
    X, y, cat_features = prepare_battle_features(df)
    train_and_evaluate(X, y, cat_features)
    print("Training and evaluation complete.")
