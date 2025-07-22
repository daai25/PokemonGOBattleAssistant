import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

df = pd.read_csv("battle_data_numeric.csv")

'''# Insert the list of strings for all columns!!!!!
data = ['left_pokemon_pokemon_type_1','left_pokemon_pokemon_type_2','left_pokemon_pokemon_fast_move','left_pokemon_pokemon_charge_move_1','left_pokemon_pokemon_charge_move_2','left_pokemon_pokemon_fast_move_type','left_pokemon_pokemon_charge_move_1_type','left_pokemon_pokemon_charge_move_2_type','left_pokemon_pokemon_dex','left_pokemon_pokemon_attack','left_pokemon_pokemon_defense','left_pokemon_pokemon_stamina','left_pokemon_pokemon_overall',
        'right_pokemon_pokemon_type_1','right_pokemon_pokemon_type_2','right_pokemon_pokemon_fast_move','right_pokemon_pokemon_charge_move_1','right_pokemon_pokemon_charge_move_2','right_pokemon_pokemon_fast_move_type','right_pokemon_pokemon_charge_move_1_type','right_pokemon_pokemon_charge_move_2_type','right_pokemon_pokemon_dex','right_pokemon_pokemon_attack','right_pokemon_pokemon_defense','right_pokemon_pokemon_stamina','right_pokemon_pokemon_overall']

target = "winner"

X = df[data]
y = df[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf = Pipeline(steps=[
    ("logreg", LogisticRegression(max_iter=1000, n_jobs=-1))
])

clf.fit(X_train, y_train)

proba = clf.predict_proba(X_val)[:, 1]
pred  = clf.predict(X_val)

print("Accuracy :", accuracy_score(y_val, pred))
print("ROC AUC  :", roc_auc_score(y_val, proba))
print(classification_report(y_val, pred))'''


cat_cols = [
    'left_pokemon_type_1','left_pokemon_type_2','left_pokemon_fast_move','left_pokemon_charge_move_1',
    'left_pokemon_charge_move_2','left_pokemon_fast_move_type','left_pokemon_charge_move_1_type',
    'left_pokemon_charge_move_2_type',
    'right_pokemon_type_1','right_pokemon_type_2','right_pokemon_fast_move','right_pokemon_charge_move_1',
    'right_pokemon_charge_move_2','right_pokemon_fast_move_type','right_pokemon_charge_move_1_type',
    'right_pokemon_charge_move_2_type'
]

num_cols = [
    'left_pokemon_dex','left_pokemon_attack','left_pokemon_defense','left_pokemon_stamina','left_pokemon_overall',
    'right_pokemon_dex','right_pokemon_attack','right_pokemon_defense','right_pokemon_stamina','right_pokemon_overall'
]

target = "winner"


X = df[cat_cols + num_cols]
y = df[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

preproc = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)

clf = Pipeline(steps=[
    ("pre", preproc),
    ("logreg", LogisticRegression(max_iter=1000, n_jobs=-1))
])

clf.fit(X_train, y_train)

proba = clf.predict_proba(X_val)[:, 1]
pred  = clf.predict(X_val)

print("Accuracy :", accuracy_score(y_val, pred))
print("ROC AUC  :", roc_auc_score(y_val, proba))
print(classification_report(y_val, pred))


#Feature Importance (optional)
# pull one‑hot feature names from the ColumnTransformer
ohe = clf.named_steps["pre"].named_transformers_["cat"]
cat_features = ohe.get_feature_names_out(cat_cols)

all_features = np.concatenate([cat_features, num_cols])
coefs = clf.named_steps["logreg"].coef_.ravel()

fi = (pd.Series(coefs, index=all_features)
        .sort_values(key=abs, ascending=False)
        .head(25))
print(fi)








# Battle Tester

'''altaria = {
    'type_1': 'dragon',
    'type_2': 'flying',
    'fast_move': 'Dragon Breath',
    'charge_move_1': 'Sky Attack',
    'charge_move_2': 'Dazzling Gleam',
    'fast_move_type': 'dragon',
    'charge_move_1_type': 'flying',
    'charge_move_2_type': 'fairy',
    'dex': 334,
    'attack': 141,
    'defense': 201,
    'stamina': 181,
    'overall': 523
}

swampert = {
    'type_1': 'water',
    'type_2': 'ground',
    'fast_move': 'Mud Shot',
    'charge_move_1': 'Hydro Cannon',
    'charge_move_2': 'Earthquake',
    'fast_move_type': 'ground',
    'charge_move_1_type': 'water',
    'charge_move_2_type': 'ground',
    'dex': 260,
    'attack': 171,
    'defense': 150,
    'stamina': 225,
    'overall': 546
}

battle = {
    'left_pokemon_type_1': altaria['type_1'],
    'left_pokemon_type_2': altaria['type_2'],
    'left_pokemon_fast_move': altaria['fast_move'],
    'left_pokemon_charge_move_1': altaria['charge_move_1'],
    'left_pokemon_charge_move_2': altaria['charge_move_2'],
    'left_pokemon_fast_move_type': altaria['fast_move_type'],
    'left_pokemon_charge_move_1_type': altaria['charge_move_1_type'],
    'left_pokemon_charge_move_2_type': altaria['charge_move_2_type'],
    'left_pokemon_dex': altaria['dex'],
    'left_pokemon_attack': altaria['attack'],
    'left_pokemon_defense': altaria['defense'],
    'left_pokemon_stamina': altaria['stamina'],
    'left_pokemon_overall': altaria['overall'],
    
    'right_pokemon_type_1': swampert['type_1'],
    'right_pokemon_type_2': swampert['type_2'],
    'right_pokemon_fast_move': swampert['fast_move'],
    'right_pokemon_charge_move_1': swampert['charge_move_1'],
    'right_pokemon_charge_move_2': swampert['charge_move_2'],
    'right_pokemon_fast_move_type': swampert['fast_move_type'],
    'right_pokemon_charge_move_1_type': swampert['charge_move_1_type'],
    'right_pokemon_charge_move_2_type': swampert['charge_move_2_type'],
    'right_pokemon_dex': swampert['dex'],
    'right_pokemon_attack': swampert['attack'],
    'right_pokemon_defense': swampert['defense'],
    'right_pokemon_stamina': swampert['stamina'],
    'right_pokemon_overall': swampert['overall'],
}

battle_df = pd.DataFrame([battle])

# Predict whether the LEFT-side Pokémon (Altaria) wins
result = clf.predict(battle_df)[0]
proba = clf.predict_proba(battle_df)[0]

if result == 1:
    print(f"Predicted Winner: Altaria (probability = {proba[1]:.2f})")
else:
    print(f"Predicted Winner: Swampert (probability = {proba[0]:.2f})")'''
