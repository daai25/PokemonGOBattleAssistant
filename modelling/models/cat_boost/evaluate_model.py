"""
Evaluate CatBoost model on the SAME deterministic 80/20 split used in training.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, confusion_matrix,
                             classification_report)
from catboost import CatBoostClassifier

# ---------- Paths ----------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BATTLE_DATA_PATH = os.path.join(ROOT_DIR, 'data_acquisition', 'vectorized_data', 'poke_battles.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'catboost_model.cbm')
EVAL_DIR = os.path.join(os.path.dirname(__file__), 'evaluation')
os.makedirs(EVAL_DIR, exist_ok=True)

# ---------- Constants ----------
NUM_TO_TYPE = {
    0: 'None', 1: 'Bug', 2: 'Dark', 3: 'Dragon', 4: 'Electric', 5: 'Fairy',
    6: 'Fighting', 7: 'Fire', 8: 'Flying', 9: 'Ghost', 10: 'Grass',
    11: 'Ground', 12: 'Ice', 13: 'Normal', 14: 'Poison', 15: 'Psychic',
    16: 'Rock', 17: 'Steel', 18: 'Water'
}

def prepare_battle_features(df: pd.DataFrame):
    """Mirror training preprocessing exactly."""
    df = df.dropna()

    type_cols = ['left_pokemon_type_1', 'left_pokemon_type_2',
                 'right_pokemon_type_1', 'right_pokemon_type_2']
    for c in type_cols:
        df[c] = df[c].map(NUM_TO_TYPE).astype('category')

    df['attack_ratio']  = df['left_pokemon_attack']   / (df['right_pokemon_attack']   + 1e-5)
    df['defense_ratio'] = df['left_pokemon_defense']  / (df['right_pokemon_defense']  + 1e-5)
    df['stamina_ratio'] = df['left_pokemon_stamina']  / (df['right_pokemon_stamina']  + 1e-5)

    categorical_columns = type_cols + [
        'left_pokemon_fast_move','left_pokemon_charge_move_1','left_pokemon_charge_move_2',
        'left_pokemon_fast_move_type','left_pokemon_charge_move_1_type','left_pokemon_charge_move_2_type',
        'right_pokemon_fast_move','right_pokemon_charge_move_1','right_pokemon_charge_move_2',
        'right_pokemon_fast_move_type','right_pokemon_charge_move_1_type','right_pokemon_charge_move_2_type'
    ]
    for c in categorical_columns:
        df[c] = df[c].astype('category')

    exclude = [
        'pokemon_winner','pokemon_loser','right_pokemon_dex','left_pokemon_dex',
        'left_pokemon_overall','right_pokemon_overall',
        'left_pokemon_attack','right_pokemon_attack',
        'left_pokemon_defense','right_pokemon_defense',
        'left_pokemon_stamina','right_pokemon_stamina'
    ]

    X = df.drop(columns=['winner'] + exclude)
    y = df['winner']
    return X, y

def main():
    # Load model
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    # Load & prep data
    df = pd.read_csv(BATTLE_DATA_PATH)
    X_all, y_all = prepare_battle_features(df)

    # Align columns to model's expectation
    model_feats = model.feature_names_
    X_all = X_all.reindex(columns=model_feats, fill_value=0)

    # Recreate the SAME split used in training
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # Predict & metrics
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    plt.figure(figsize=(7,5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'confusion_matrix.png'))

    # ROC Curve
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f'AUC={auc_score:.2f}')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'roc_curve.png'))

    # Feature importance
    imps = model.get_feature_importance()
    fi = pd.DataFrame({'feature': model_feats, 'importance': imps / imps.sum()}) \
           .sort_values('importance', ascending=False)
    fi.to_csv(os.path.join(EVAL_DIR, 'feature_importance.csv'), index=False)

    plt.figure(figsize=(12,7))
    plt.barh(fi['feature'][:15], fi['importance'][:15])
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importance (normalized)')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'feature_importance.png'))

    # Summary CSV
    pd.DataFrame({
        'Accuracy':[acc], 'Precision':[prec], 'Recall':[rec],
        'F1 Score':[f1], 'ROC AUC':[auc_score]
    }).to_csv(os.path.join(EVAL_DIR, 'metrics_summary.csv'), index=False)

    print(f"Evaluation complete. Results saved to: {EVAL_DIR}")

if __name__ == '__main__':
    main()
