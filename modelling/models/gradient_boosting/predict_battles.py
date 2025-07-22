"""
Script to predict battle outcomes using the trained Gradient Boosting model
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Set paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
POKEMON_DATA_PATH = os.path.join(ROOT_DIR, 'data_acquisition', 'vectorized_data', 'all_overall_rankings_vectorized.csv')
MODELS_DIR = os.path.join(ROOT_DIR, 'modelling', 'models', 'gradient_boosting')
MODEL_PATH = os.path.join(MODELS_DIR, 'gradient_boosting_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

def load_model_and_data():
    """Load the trained model, scaler, and Pokemon data"""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    pokemon_data = pd.read_csv(POKEMON_DATA_PATH)
    return model, scaler, pokemon_data

def get_pokemon_features(pokemon_name, pokemon_data):
    """Get features for a specific Pokemon"""
    pokemon = pokemon_data[pokemon_data['Pokemon'] == pokemon_name]
    if pokemon.empty:
        print(f"Pokemon '{pokemon_name}' not found in the dataset.")
        return None
    features = {
        'type_1': pokemon['Type 1'].values[0],
        'type_2': pokemon['Type 2'].values[0],
        'attack': pokemon['Attack'].values[0],
        'defense': pokemon['Defense'].values[0],
        'stamina': pokemon['Stamina'].values[0],
        'overall': pokemon['Score'].values[0],
        'fast_move': pokemon['Fast Move'].values[0],
        'charge_move_1': pokemon['Charged Move 1'].values[0],
        'charge_move_2': pokemon['Charged Move 2'].values[0],
        'fast_move_type': None,  # Placeholder, should be mapped from move data
        'charge_move_1_type': None,
        'charge_move_2_type': None,
    }
    return features

def predict_battle_outcome(left_pokemon, right_pokemon, model, scaler):
    """Predict the outcome of a battle between two Pokemon with symmetry handling"""
    battle_features_original = [
        left_pokemon['type_1'], left_pokemon['type_2'],
        left_pokemon['fast_move_type'], left_pokemon['charge_move_1_type'], left_pokemon['charge_move_2_type'],
        left_pokemon['attack'], left_pokemon['defense'], left_pokemon['stamina'], left_pokemon['overall'],
        right_pokemon['type_1'], right_pokemon['type_2'],
        right_pokemon['fast_move_type'], right_pokemon['charge_move_1_type'], right_pokemon['charge_move_2_type'],
        right_pokemon['attack'], right_pokemon['defense'], right_pokemon['stamina'], right_pokemon['overall']
    ]
    battle_features_reversed = [
        right_pokemon['type_1'], right_pokemon['type_2'],
        right_pokemon['fast_move_type'], right_pokemon['charge_move_1_type'], right_pokemon['charge_move_2_type'],
        right_pokemon['attack'], right_pokemon['defense'], right_pokemon['stamina'], right_pokemon['overall'],
        left_pokemon['type_1'], left_pokemon['type_2'],
        left_pokemon['fast_move_type'], left_pokemon['charge_move_1_type'], left_pokemon['charge_move_2_type'],
        left_pokemon['attack'], left_pokemon['defense'], left_pokemon['stamina'], left_pokemon['overall']
    ]
    X_original = np.array(battle_features_original).reshape(1, -1)
    X_reversed = np.array(battle_features_reversed).reshape(1, -1)
    X_original_scaled = scaler.transform(X_original)
    X_reversed_scaled = scaler.transform(X_reversed)
    prediction_original = model.predict(X_original_scaled)
    probability_original = model.predict_proba(X_original_scaled)
    prediction_reversed = model.predict(X_reversed_scaled)
    probability_reversed = model.predict_proba(X_reversed_scaled)
    adjusted_prediction_reversed = 1 - prediction_reversed[0]
    adjusted_probability_reversed = probability_reversed[0][::-1]
    avg_prediction = 1 if (probability_original[0][1] + adjusted_probability_reversed[1])/2 > 0.5 else 0
    avg_probability = [
        (probability_original[0][0] + adjusted_probability_reversed[0])/2,
        (probability_original[0][1] + adjusted_probability_reversed[1])/2
    ]
    return avg_prediction, np.array(avg_probability)

def main():
    """Main function to demonstrate model prediction"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Model or scaler not found. Please train the model first.")
        return
    model, scaler, pokemon_data = load_model_and_data()
    print("Pokemon GO Battle Predictor (Gradient Boosting)")
    print("=" * 30)
    print("Example Pokemon:")
    example_pokemon = pokemon_data['Pokemon'].sample(10).tolist()
    for i, pokemon in enumerate(example_pokemon):
        print(f"{i+1}. {pokemon}")
    print("\nEnter Pokemon names for battle prediction:")
    left_pokemon_name = input("First Pokemon: ")
    right_pokemon_name = input("Second Pokemon: ")
    left_pokemon = get_pokemon_features(left_pokemon_name, pokemon_data)
    right_pokemon = get_pokemon_features(right_pokemon_name, pokemon_data)
    if left_pokemon is None or right_pokemon is None:
        print("One or both Pokemon not found. Please try again.")
        return
    # For demo purposes, use placeholder values for move types
    left_pokemon['fast_move_type'] = 1
    left_pokemon['charge_move_1_type'] = 1
    left_pokemon['charge_move_2_type'] = 1
    right_pokemon['fast_move_type'] = 1
    right_pokemon['charge_move_1_type'] = 1
    right_pokemon['charge_move_2_type'] = 1
    prediction, probability = predict_battle_outcome(
        left_pokemon, right_pokemon, model, scaler
    )
    print("\nBattle Prediction:")
    if prediction == 0:
        print(f"Winner: {left_pokemon_name}")
    else:
        print(f"Winner: {right_pokemon_name}")
    print(f"Confidence: {max(probability) * 100:.2f}%")

if __name__ == "__main__":
    main()
