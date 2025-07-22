"""
Gradient Boosting model for Pokemon GO Battle predictions
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# Set paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BATTLE_DATA_PATH = os.path.join(ROOT_DIR, 'data_acquisition', 'vectorized_data', 'poke_battles.csv')
MODELS_DIR = os.path.join(ROOT_DIR, 'modelling', 'models', 'gradient_boosting')

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    """Load and prepare data for modeling"""
    print(f"Loading battle data from: {BATTLE_DATA_PATH}")
    try:
        battle_data = pd.read_csv(BATTLE_DATA_PATH)
        print(f"Battle data shape: {battle_data.shape}")
        return battle_data
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Battle data path exists: {os.path.exists(BATTLE_DATA_PATH)}")
        raise

def calculate_stab(row, poke_type_1, poke_type_2, move_type):
    return int(
        move_type in [row[poke_type_1], row[poke_type_2]]
    )

def prepare_battle_features(battle_data):
    """Prepare features from battle data"""
    # Feature engineering (diffs and squared diffs)
    battle_data = battle_data.dropna()
    # battle_data['attack_diff'] = battle_data['left_pokemon_attack'] - battle_data['right_pokemon_attack']
    # battle_data['defense_diff'] = battle_data['left_pokemon_defense'] - battle_data['right_pokemon_defense']
    # battle_data['stamina_diff'] = battle_data['left_pokemon_stamina'] - battle_data['right_pokemon_stamina']
    
    battle_data['attack_ratio'] = battle_data['left_pokemon_attack'] / battle_data['right_pokemon_attack']
    battle_data['defense_ratio'] = battle_data['left_pokemon_defense'] / battle_data['right_pokemon_defense']
    battle_data['stamina_ratio'] = battle_data['left_pokemon_stamina'] / battle_data['right_pokemon_stamina']

    categorical_columns = [
        'left_pokemon_type_1', 'left_pokemon_type_2',
        'left_pokemon_fast_move', 'left_pokemon_charge_move_1', 'left_pokemon_charge_move_2',
        'left_pokemon_fast_move_type', 'left_pokemon_charge_move_1_type', 'left_pokemon_charge_move_2_type',
        'right_pokemon_type_1', 'right_pokemon_type_2',
        'right_pokemon_fast_move', 'right_pokemon_charge_move_1', 'right_pokemon_charge_move_2',
        'right_pokemon_fast_move_type', 'right_pokemon_charge_move_1_type', 'right_pokemon_charge_move_2_type',
    ]
    exclude_cols = [
        'pokemon_winner', 'pokemon_loser', 'right_pokemon_dex', 'left_pokemon_dex',
        'left_pokemon_overall', 'right_pokemon_overall',
        'left_pokemon_attack', 'right_pokemon_attack',
        'left_pokemon_defense', 'right_pokemon_defense',
        'left_pokemon_stamina', 'right_pokemon_stamina',
        'attack_ratio', 'defense_ratio', 'stamina_ratio',
    ]  # Add any columns you want to exclude from features

    battle_data = pd.get_dummies(battle_data, columns=categorical_columns)
    X = battle_data.drop(columns=['winner'] + exclude_cols)
    y = battle_data['winner']
    # Save feature names for later use in evaluation
    feature_names_path = os.path.join(MODELS_DIR, 'feature_names.pkl')
    joblib.dump(X.columns.tolist(), feature_names_path)
    return X, y

def train_model(X, y):
    """Train a Gradient Boosting model with fixed best hyperparameters"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")

        # Save test indices for evaluation
        test_indices_path = os.path.join(MODELS_DIR, 'test_indices.pkl')
        test_indices = X_test.index.tolist()
        joblib.dump(test_indices, test_indices_path)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Use fixed best parameters
        best_gbc = GradientBoostingClassifier(
            learning_rate=0.1,
            max_depth=7,
            n_estimators=300,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42
        )

        print("Training Gradient Boosting model with best parameters...")
        best_gbc.fit(X_train_scaled, y_train)

        y_pred = best_gbc.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        feature_importance = pd.DataFrame(
            {'feature': X.columns, 'importance': best_gbc.feature_importances_}
        ).sort_values('importance', ascending=False)

        print("\nFeature Importance:")
        print(feature_importance.head(10))

        plt.figure(figsize=(12, 8))
        plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top 15 Feature Importance')
        plt.tight_layout()
        plot_path = os.path.join(MODELS_DIR, 'feature_importance.png')
        plt.savefig(plot_path)
        print(f"Feature importance plot saved to: {plot_path}")

        # Save model and scaler
        model_path = os.path.join(MODELS_DIR, 'gradient_boosting_model.pkl')
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        joblib.dump(best_gbc, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")

        return best_gbc, scaler, X_test_scaled, y_test

    except Exception as e:
        print(f"Error during model training: {e}")
        raise

def main():
    """Main function to execute the modeling process"""
    try:
        print("Starting Gradient Boosting model training process...")
        print(f"Working directory: {os.getcwd()}")
        print(f"Models directory: {MODELS_DIR}")

        print("\nLoading data...")
        battle_data = load_data()
        # Use a subset for faster training (optional)

        print("\nPreparing features...")
        X, y = prepare_battle_features(battle_data)

        print("\nTraining model...")
        model, scaler, X_test, y_test = train_model(X, y)

        print("\nModel training complete!")
        print(f"Model and artifacts saved to: {MODELS_DIR}")

        return True
    except Exception as e:
        print(f"Error in main process: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("Model training failed. Please check the error messages above.")
    else:
        print("Model training completed successfully.")