"""
Utility script to evaluate the Gradient Boosting model
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
BATTLE_DATA_PATH = os.path.join(ROOT_DIR, 'data_acquisition', 'vectorized_data', 'poke_battles.csv')
MODELS_DIR = os.path.join(ROOT_DIR, 'modelling', 'models', 'gradient_boosting')
MODEL_PATH = os.path.join(MODELS_DIR, 'gradient_boosting_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
EVAL_DIR = os.path.join(MODELS_DIR, 'evaluation')
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, 'feature_names.pkl')
TEST_INDICES_PATH = os.path.join(MODELS_DIR, 'test_indices.pkl')

# Create evaluation directory if it doesn't exist
os.makedirs(EVAL_DIR, exist_ok=True)

def load_model_and_data():
    """Load the trained model, scaler, test indices, and battle data"""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    battle_data = pd.read_csv(BATTLE_DATA_PATH)
    test_indices = joblib.load(TEST_INDICES_PATH)
    return model, scaler, battle_data, test_indices

def prepare_battle_features(battle_data, test_indices=None):
    """Prepare features from battle data (must match training features)"""
    # Drop rows with NaN (as in training)
    battle_data = battle_data.dropna()
    # Feature engineering (diffs and squared diffs)
    battle_data['dex_diff'] = battle_data['left_pokemon_dex'] - battle_data['right_pokemon_dex']
    battle_data['attack_diff'] = battle_data['left_pokemon_attack'] - battle_data['right_pokemon_attack']
    battle_data['defense_diff'] = battle_data['left_pokemon_defense'] - battle_data['right_pokemon_defense']
    battle_data['stamina_diff'] = battle_data['left_pokemon_stamina'] - battle_data['right_pokemon_stamina']
    battle_data['overall_diff'] = battle_data['left_pokemon_overall'] - battle_data['right_pokemon_overall']
    # battle_data['dex_diff^2'] = (battle_data['left_pokemon_dex'] - battle_data['right_pokemon_dex'])**2
    # battle_data['attack_diff^2'] = (battle_data['left_pokemon_attack'] - battle_data['right_pokemon_attack'])**2
    # battle_data['defense_diff^2'] = (battle_data['left_pokemon_defense'] - battle_data['right_pokemon_defense'])**2
    # battle_data['stamina_diff^2'] = (battle_data['left_pokemon_stamina'] - battle_data['right_pokemon_stamina'])**2
    # battle_data['overall_diff^2'] = (battle_data['left_pokemon_overall'] - battle_data['right_pokemon_overall'])**2

    categorical_columns = [
        'left_pokemon_type_1', 'left_pokemon_type_2',
        'left_pokemon_fast_move', 'left_pokemon_charge_move_1', 'left_pokemon_charge_move_2',
        'left_pokemon_fast_move_type', 'left_pokemon_charge_move_1_type', 'left_pokemon_charge_move_2_type',
        'right_pokemon_type_1', 'right_pokemon_type_2',
        'right_pokemon_fast_move', 'right_pokemon_charge_move_1', 'right_pokemon_charge_move_2',
        'right_pokemon_fast_move_type', 'right_pokemon_charge_move_1_type', 'right_pokemon_charge_move_2_type',
    ]
    exclude_cols = []  # Add any columns you want to exclude from features

    battle_data = pd.get_dummies(battle_data, columns=categorical_columns)
    battle_data.columns = [col.replace('.0', '') for col in battle_data.columns]
    X = battle_data.drop(columns=['winner'] + exclude_cols)
    y = battle_data['winner']
    # Align features to match training
    if os.path.exists(FEATURE_NAMES_PATH):
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        X = X.reindex(columns=feature_names, fill_value=0)
    # Only use test indices if provided
    if test_indices is not None:
        X = X.iloc[test_indices].reset_index(drop=True)
        y = y.iloc[test_indices].reset_index(drop=True)
    return X, y

def evaluate_model(model, scaler, X, y):
    """Evaluate the model and generate performance metrics and visualizations"""
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    else:
        y_pred_proba = np.zeros_like(y_pred, dtype=float)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

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
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(TEST_INDICES_PATH):
        print("Model, scaler, or test indices not found. Please train the model first.")
        return

    print("Loading model and data...")
    model, scaler, battle_data, test_indices = load_model_and_data()

    print("Preparing features (using only test split)...")
    X, y = prepare_battle_features(battle_data, test_indices=test_indices)

    print("Evaluating model...")
    evaluate_model(model, scaler, X, y)

    print(f"Evaluation complete. Results saved to: {EVAL_DIR}")

if __name__ == "__main__":
    main()
