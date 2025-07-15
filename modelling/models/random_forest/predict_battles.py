"""
Script to predict battle outcomes using the trained Random Forest model
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Set paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
POKEMON_DATA_PATH = os.path.join(ROOT_DIR, 'data_acquisition', 'vectorized_data', 'all_overall_rankings_vectorized.csv')
MODELS_DIR = os.path.join(ROOT_DIR, 'modelling', 'models', 'random_forest')
MODEL_PATH = os.path.join(MODELS_DIR, 'random_forest_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

def load_model_and_data():
    """Load the trained model, scaler, and Pokemon data"""
    # Load the model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Load Pokemon data
    pokemon_data = pd.read_csv(POKEMON_DATA_PATH)
    
    return model, scaler, pokemon_data

def get_pokemon_features(pokemon_name, pokemon_data):
    """Get features for a specific Pokemon"""
    # Find the Pokemon in the dataset
    pokemon = pokemon_data[pokemon_data['Pokemon'] == pokemon_name]
    
    if pokemon.empty:
        print(f"Pokemon '{pokemon_name}' not found in the dataset.")
        return None
    
    # Extract relevant features
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
        'fast_move_type': None,  # We need to map this from another source
        'charge_move_1_type': None,  # We need to map this from another source
        'charge_move_2_type': None,  # We need to map this from another source
    }
    
    return features

def predict_battle_outcome(left_pokemon, right_pokemon, model, scaler):
    """Predict the outcome of a battle between two Pokemon"""
    # Create a feature vector for the battle
    battle_features = [
        # Left Pokemon features
        left_pokemon['type_1'], 
        left_pokemon['type_2'],
        left_pokemon['fast_move_type'],
        left_pokemon['charge_move_1_type'],
        left_pokemon['charge_move_2_type'],
        left_pokemon['attack'],
        left_pokemon['defense'],
        left_pokemon['stamina'],
        left_pokemon['overall'],
        
        # Right Pokemon features
        right_pokemon['type_1'], 
        right_pokemon['type_2'],
        right_pokemon['fast_move_type'],
        right_pokemon['charge_move_1_type'],
        right_pokemon['charge_move_2_type'],
        right_pokemon['attack'],
        right_pokemon['defense'],
        right_pokemon['stamina'],
        right_pokemon['overall']
    ]
    
    # Convert to numpy array and reshape
    X = np.array(battle_features).reshape(1, -1)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)
    probability = model.predict_proba(X_scaled)
    
    return prediction[0], probability[0]

def main():
    """Main function to demonstrate model prediction"""
    # Check if model exists
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Model or scaler not found. Please train the model first.")
        return
    
    # Load model and data
    model, scaler, pokemon_data = load_model_and_data()
    
    # Demo prediction
    print("Pokemon GO Battle Predictor")
    print("=" * 30)
    
    # List some example Pokemon for the user to choose from
    print("Example Pokemon:")
    example_pokemon = pokemon_data['Pokemon'].sample(10).tolist()
    for i, pokemon in enumerate(example_pokemon):
        print(f"{i+1}. {pokemon}")
    
    # Get user input
    print("\nEnter Pokemon names for battle prediction:")
    left_pokemon_name = input("First Pokemon: ")
    right_pokemon_name = input("Second Pokemon: ")
    
    # Get Pokemon features
    left_pokemon = get_pokemon_features(left_pokemon_name, pokemon_data)
    right_pokemon = get_pokemon_features(right_pokemon_name, pokemon_data)
    
    if left_pokemon is None or right_pokemon is None:
        print("One or both Pokemon not found. Please try again.")
        return
    
    # For demo purposes, use placeholder values for move types
    # In a real application, these would be derived from the move dictionaries
    left_pokemon['fast_move_type'] = 1
    left_pokemon['charge_move_1_type'] = 1
    left_pokemon['charge_move_2_type'] = 1
    right_pokemon['fast_move_type'] = 1
    right_pokemon['charge_move_1_type'] = 1
    right_pokemon['charge_move_2_type'] = 1
    
    # Predict outcome
    prediction, probability = predict_battle_outcome(
        left_pokemon, right_pokemon, model, scaler
    )
    
    # Display results
    print("\nBattle Prediction:")
    print(f"Winner: {'First Pokemon' if prediction == 0 else 'Second Pokemon'}")
    print(f"Confidence: {max(probability) * 100:.2f}%")

if __name__ == "__main__":
    main()
