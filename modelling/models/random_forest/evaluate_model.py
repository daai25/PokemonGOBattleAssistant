"""
Utility script to evaluate the Random Forest model
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)

# Set paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BATTLE_DATA_PATH = os.path.join(ROOT_DIR, 'data_acquisition', 'vectorized_data', 'battle_data_numeric.csv')
MODELS_DIR = os.path.join(ROOT_DIR, 'modelling', 'models', 'random_forest')
MODEL_PATH = os.path.join(MODELS_DIR, 'random_forest_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
EVAL_DIR = os.path.join(MODELS_DIR, 'evaluation')

# Create evaluation directory if it doesn't exist
os.makedirs(EVAL_DIR, exist_ok=True)

def load_model_and_data():
    """Load the trained model, scaler, and test data"""
    # Load the model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Load battle data
    battle_data = pd.read_csv(BATTLE_DATA_PATH)
    
    return model, scaler, battle_data

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

def evaluate_model(model, scaler, X, y):
    """Evaluate the model and generate performance metrics and visualizations"""
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(EVAL_DIR, 'confusion_matrix.png'))
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(EVAL_DIR, 'roc_curve.png'))
    
    # Feature importance
    feature_importance = pd.DataFrame(
        {'feature': X.columns, 'importance': model.feature_importances_}
    ).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'feature_importance.png'))
    
    # Save feature importance to CSV
    feature_importance.to_csv(os.path.join(EVAL_DIR, 'feature_importance.csv'), index=False)
    
    # Save all metrics to a summary file
    metrics_summary = {
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1],
        'ROC AUC': [roc_auc]
    }
    
    pd.DataFrame(metrics_summary).to_csv(os.path.join(EVAL_DIR, 'metrics_summary.csv'), index=False)
    
    return accuracy, precision, recall, f1, roc_auc

def main():
    """Main function to execute the evaluation process"""
    # Check if model exists
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Model or scaler not found. Please train the model first.")
        return
    
    print("Loading model and data...")
    model, scaler, battle_data = load_model_and_data()
    
    print("Preparing features...")
    X, y = prepare_battle_features(battle_data)
    
    print("Evaluating model...")
    evaluate_model(model, scaler, X, y)
    
    print(f"Evaluation complete. Results saved to: {EVAL_DIR}")

if __name__ == "__main__":
    main()
