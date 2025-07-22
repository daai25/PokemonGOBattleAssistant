"""
Utility script to evaluate the CatBoost model
"""
import os
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)
from catboost import CatBoostClassifier

# Set paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BATTLE_DATA_PATH = os.path.join(
    ROOT_DIR, 'data_acquisition', 'vectorized_data', 'poke_battles.csv'
)
FEATURE_NAMES_PATH = os.path.join(os.path.dirname(__file__), 'feature_names.pkl')
TEST_INDICES_PATH = os.path.join(os.path.dirname(__file__), 'test_indices.pkl')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'catboost_model.cbm')
EVAL_DIR = os.path.join(os.path.dirname(__file__), 'evaluation')

os.makedirs(EVAL_DIR, exist_ok=True)


def load_model_and_data():
    """Load trained CatBoost model, feature names, test indices, and raw data"""
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    df = pd.read_csv(BATTLE_DATA_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    test_indices = joblib.load(TEST_INDICES_PATH)
    return model, df, feature_names, test_indices


def prepare_battle_features(df, feature_names, test_indices=None):
    """Prepare features identical to training process"""
    df = df.dropna()
    # Map numeric type codes
    NUM_TO_TYPE = {
        0: 'None', 1: 'Bug', 2: 'Dark', 3: 'Dragon', 4: 'Electric', 5: 'Fairy',
        6: 'Fighting', 7: 'Fire', 8: 'Flying', 9: 'Ghost', 10: 'Grass',
        11: 'Ground', 12: 'Ice', 13: 'Normal', 14: 'Poison', 15: 'Psychic',
        16: 'Rock', 17: 'Steel', 18: 'Water'
    }
    type_cols = ['left_pokemon_type_1','left_pokemon_type_2',
                 'right_pokemon_type_1','right_pokemon_type_2']
    for c in type_cols:
        df[c] = df[c].map(NUM_TO_TYPE).astype('category')

    # Ratio features
    df['attack_ratio']  = df['left_pokemon_attack']   / (df['right_pokemon_attack']   + 1e-5)
    df['defense_ratio'] = df['left_pokemon_defense']  / (df['right_pokemon_defense']  + 1e-5)
    df['stamina_ratio'] = df['left_pokemon_stamina']  / (df['right_pokemon_stamina']  + 1e-5)

    # Drop unused columns
    exclude = [
        'pokemon_winner','pokemon_loser','right_pokemon_dex','left_pokemon_dex',
        'left_pokemon_overall','right_pokemon_overall',
        'left_pokemon_attack','right_pokemon_attack',
        'left_pokemon_defense','right_pokemon_defense',
        'left_pokemon_stamina','right_pokemon_stamina'
    ]
    X = df.drop(columns=['winner'] + exclude)
    # Align to training features
    X = X.reindex(columns=feature_names, fill_value=0)
    y = df['winner']
    if test_indices is not None:
        X = X.iloc[test_indices].reset_index(drop=True)
        y = y.iloc[test_indices].reset_index(drop=True)
    return X, y


def evaluate_model(model, X, y):
    """Evaluate model performance and save metrics/plots"""
    # Predictions
    y_pred = model.predict(X)
    try:
        y_proba = model.predict_proba(X)[:,1]
    except:
        y_proba = np.zeros_like(y_pred, dtype=float)

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    # Confusion Matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(EVAL_DIR, 'confusion_matrix.png'))

    # ROC Curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0,1],[0,1], '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(EVAL_DIR, 'roc_curve.png'))

    # Feature Importance
    imps = model.get_feature_importance()
    norm_imps = imps / imps.sum()
    fi = pd.DataFrame({'feature': joblib.load(FEATURE_NAMES_PATH), 'importance': norm_imps})
    fi = fi.sort_values('importance', ascending=False)
    fi.to_csv(os.path.join(EVAL_DIR, 'feature_importance.csv'), index=False)

    plt.figure(figsize=(12,8))
    plt.barh(fi['feature'][:15], fi['importance'][:15])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importance (normalized)')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'feature_importance.png'))

    # Summary CSV
    metrics = pd.DataFrame({
        'Accuracy':[accuracy], 'Precision':[precision],
        'Recall':[recall], 'F1 Score':[f1], 'ROC AUC':[roc_auc]
    })
    metrics.to_csv(os.path.join(EVAL_DIR, 'metrics_summary.csv'), index=False)


def main():
    model, df, feature_names, test_indices = load_model_and_data()
    X, y = prepare_battle_features(df, feature_names, test_indices)
    evaluate_model(model, X, y)
    print(f"Evaluation complete. Results saved to: {EVAL_DIR}")


if __name__ == '__main__':
    main()