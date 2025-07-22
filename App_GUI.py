import pandas as pd
import csv
import streamlit as st


# Page setup
st.set_page_config(page_title="Pokémon Battle Predictor", page_icon="⚔️", layout="wide")

# Initialize session state for dynamic inputs on the left side
if "your_pokemon_count" not in st.session_state:
    st.session_state.your_pokemon_count = 1  # start with one input on the left
df = pd.read_csv("cp1500_all_overall_rankings.csv")  # Load your dataset here
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
    if not opponent and not player:
        st.error("Please enter both opponent's and your Pokemon.")
        return
    if not opponent:
        st.error("Please enter the opponent's Pokemon.")
        return
    if not player:
        st.error("Please enter your Pokemon.")
        return
    opponent_data = df[df.iloc[:, 0] == opponent.capitalize()]
    player_data = df[df.iloc[:, 0] == player.capitalize()]
    if opponent_data.empty:
        st.error("The opponent's Pokémon is not found in the dataset. Please check the spelling.")
        return
    if player_data.empty:
        st.error("Your Pokémon is not found in the dataset. Please check the spelling.")
        return

    
    # input their data and run them through the model
    # return the probability that player wins
    return 0.75

# Battle button click
if battle:
    probability_dict = {}
    for pokemon in your_team:
        probability = battle_model(opponent_pokemon, pokemon)
        if probability is not None:
            probability_dict[pokemon] = probability
        else:
            st.error(f"An error occurred while calculating the battle probability for {pokemon}.")

    best_pokemon = None
    for pokemon, probability in probability_dict.items():
        if best_pokemon is None or probability > probability_dict[best_pokemon]:
            best_pokemon = pokemon
    st.write(f"Probability that your {best_pokemon} wins: {probability_dict[best_pokemon] * 100}%")
    
