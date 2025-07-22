"""
Random Forest model for Pokemon GO Battle predictions
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# Set paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
POKEMON_DATA_PATH = os.path.join(ROOT_DIR, 'data_acquisition', 'vectorized_data', 'all_overall_rankings_vectorized.csv')
BATTLE_DATA_PATH = os.path.join(ROOT_DIR, 'data_acquisition', 'vectorized_data', 'battle_data_numeric.csv')
MODELS_DIR = os.path.join(ROOT_DIR, 'modelling', 'models', 'random_forest')

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    """Load and prepare data for modeling"""
    print(f"Loading Pokemon data from: {POKEMON_DATA_PATH}")
    print(f"Loading battle data from: {BATTLE_DATA_PATH}")
    
    try:
        # Load Pokemon data
        pokemon_data = pd.read_csv(POKEMON_DATA_PATH)
        
        # Load battle data
        battle_data = pd.read_csv(BATTLE_DATA_PATH)
        
        print(f"Pokemon data shape: {pokemon_data.shape}")
        print(f"Battle data shape: {battle_data.shape}")
        
        return pokemon_data, battle_data
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Pokemon data path exists: {os.path.exists(POKEMON_DATA_PATH)}")
        print(f"Battle data path exists: {os.path.exists(BATTLE_DATA_PATH)}")
        raise

def prepare_battle_features(battle_data):
    """Prepare features from battle data"""
    # Select relevant features
    feature_cols = [
        # Left Pokemon features
        'left_pokemon_type_1', 'left_pokemon_type_2', 
        'left_pokemon_fast_move_type', 'left_pokemon_charge_move_1_type', 'left_pokemon_charge_move_2_type',
        'left_pokemon_attack', 'left_pokemon_defense', 'left_pokemon_stamina', 'left_pokemon_overall',
        
        # Right Pokemon features
        'right_pokemon_type_1', 'right_pokemon_type_2',
        'right_pokemon_fast_move_type', 'right_pokemon_charge_move_1_type', 'right_pokemon_charge_move_2_type',
        'right_pokemon_attack', 'right_pokemon_defense', 'right_pokemon_stamina', 'right_pokemon_overall',
    ]
    
    # Target variable
    target = 'winner'
    
    X = battle_data[feature_cols]
    y = battle_data[target]
    
    # Handle NaN values (replace with appropriate values)
    X = X.fillna(-1)  # Assuming -1 represents missing values for categorical features
    
    return X, y

def train_model(X, y):
    """Train a Random Forest model with hyperparameter tuning"""
    try:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define model
        rf = RandomForestClassifier(random_state=42)
        
        # Define hyperparameters for grid search
        # Enhanced parameter grid for better performance
        param_grid = {
            'n_estimators': [100, 200],           # Erhöht auf zwei Werte für bessere Ergebnisse
            'max_depth': [15, 25],                # Veränderte Tiefenwerte für bessere Exploration
            'min_samples_split': [2, 5],          # Zwei Werte für bessere Generalisierung
            'min_samples_leaf': [1, 4],           # Breiterer Bereich für Blattgröße
            'max_features': ['sqrt', 'log2', None],     # Beide Optionen für Feature-Auswahl
            'bootstrap': [True]                   # Bootstrapping beibehalten
        }
        
        # Perform grid search to find best parameters
        print("Performing grid search for hyperparameter tuning...")
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        best_rf = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate on test set
        y_pred = best_rf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Feature importance
        feature_importance = pd.DataFrame(
            {'feature': X.columns, 'importance': best_rf.feature_importances_}
        ).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top 15 Feature Importance')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(MODELS_DIR, 'feature_importance.png')
        plt.savefig(plot_path)
        print(f"Feature importance plot saved to: {plot_path}")
        
        # Save model and scaler
        model_path = os.path.join(MODELS_DIR, 'random_forest_model.pkl')
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        
        joblib.dump(best_rf, model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        
        return best_rf, scaler, X_test_scaled, y_test
    
    except Exception as e:
        print(f"Error during model training: {e}")
        raise

def main():
    """Main function to execute the modeling process"""
    try:
        print("Starting Random Forest model training process...")
        print(f"Working directory: {os.getcwd()}")
        print(f"Models directory: {MODELS_DIR}")
        
        print("\nLoading data...")
        pokemon_data, battle_data = load_data()
        
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
