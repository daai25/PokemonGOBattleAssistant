import os
import sys
import pandas as pd
import csv
import streamlit as st

# add project root to path and import the CatBoost predictor
sys.path.insert(0, os.getcwd())
from modelling.models.cat_boost.predict_battles import predict_battle

# Page setup
st.set_page_config(page_title="Pokémon Battle Predictor", page_icon="⚔️", layout="wide")

#Pokeball
st.markdown(
    """
    <style>
    .circle-top {
        width: 100px;
        height: 100px;
        background-color: white;
        border-radius: 50%;
        border: 20px solid black;
        justify-self: center;
        z-index: 10;
        margin-top: -50px;
    }
    .top-rectangle {
        width: 100%;
        height: 100px;
        background-color: red;
        border: 5px solid black;
        border-bottom: 20px solid black;
    }
   
    </style>

    <div class="top-rectangle"></div>
    <div class="circle-top"></div>
    """,
    unsafe_allow_html=True
)

# Initialize session state for dynamic inputs on the left side
if "your_pokemon_count" not in st.session_state:
    st.session_state.your_pokemon_count = 1  # start with one input on the left
csv_path = os.path.join("data_acquisition", "processed_data", "all_pokemon.csv")
df = pd.read_csv(csv_path)  # Load your dataset here
# Title / Subtitle
st.title("Pokemon Go Battle Assistant")
st.caption("This is a simple Streamlit application that helps you pick which Pokemon to use in a battle. " \
"It displays the probability that your pokemon will win against the opponent's pokemon. " \
"It is only compatible with the Great League (1500 CP) in Pokemon Go.")

# Layout: Left / VS / Right
left_col, mid_col, right_col = st.columns([5, 2, 5])

with right_col:
    st.subheader("Your Pokémon")

    # Button to add another Pokémon input field
    if st.button("➕ Add another Pokémon", key="add_left_pokemon"):
        st.session_state.your_pokemon_count += 1
    # Button to remove the last Pokémon input field
    if st.session_state.your_pokemon_count > 1 and st.button("➖ Remove last Pokémon", key="remove_left_pokemon"):
        st.session_state.your_pokemon_count -= 1

    # Render all left-side inputs
    your_team = []
    for i in range(st.session_state.your_pokemon_count):
        name = st.text_input(f"Pokémon {i+1}", key=f"your_pokemon_{i}")
        your_team.append(name)

with mid_col:
    # Big VS in the middle
    st.markdown(
        """
        <div style='text-align:center; font-size:48px; font-weight:700; padding-top:32px;'>
            VS
        </div>
        """,
        unsafe_allow_html=True,
    )

with left_col:
    st.subheader("Opponent's Pokémon")
    opponent_pokemon = st.text_input("Pokémon 1", key="opponent_pokemon_0")


battle = st.button('Battle')

def battle_model(opponent, player):
    # Check for empty inputs
    if not opponent or not player:
        st.error("Please enter both opponent's and your Pokémon.")
        return None
    try:
        # call CatBoost prediction: returns (winner_name, [prob_left, prob_right], confidence)
        winner, proba, _ = predict_battle(player, opponent)
    except Exception as e:
        st.error(f"Prediction error for {player}: {e}")
        return None
    # proba[1] is probability that 'player' (left) wins
    return proba[1]

# Battle button click
if battle:
    probability_dict = {}
    for pokemon in your_team:
        if pokemon:
            win_prob = battle_model(opponent_pokemon, pokemon)
            if win_prob is not None:
                probability_dict[pokemon] = win_prob

    if probability_dict:
        # choose the team member with highest win probability
        best = max(probability_dict, key=probability_dict.get)
        best_prob = probability_dict[best]
        if best_prob > 0.5:
            st.success(f"Pick {best} – win chance {best_prob*100:.1f}%")
        else:
            st.warning(f"None of your pokemon is favored to win. Least likely to lose: {best} (win chance {best_prob*100:.1f}%)")
    else:
        st.error("No valid Pokémon for prediction.")

#Pokeball
st.markdown(
    """
    <style>
    .low-rectangle {
        width: 100%;
        height: 100px;
        background-color: white;
        border: 5px solid black;
        border-top: 20px solid black;
        margin-bottom: -150px;
        margin-top: 50px;
    }
    .cover-white {
        width: 100%;
        height: 50px;
        background-color: white;
        margin-top: -150px;
    }
    .circle-bottom {
        z-index: 10;
        margin-top: -50px;
        margin-bottom: 50px;
        width: 100px;
        height: 100px;
        background-color: white;
        border-radius: 50%;
        border: 20px solid black;
        justify-self: center;
    }
    </style>

    
    <div class="low-rectangle"></div>
    <div class="circle-bottom"></div>
    <div class="cover-white"></div>
    """,
    unsafe_allow_html=True
)
