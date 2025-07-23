import pandas as pd
import streamlit as st
from PIL import Image


# Page setup
st.set_page_config(page_title="Pokémon Image Identifier", page_icon="⚔️", layout="wide")

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

# Title / Subtitle
st.title("Pokémon Image Identifier")
st.caption("This is a simple Streamlit application that helps you identify Pokémon. " \
"It displays the name and stats of the given Pokémon.")
st.sidebar.success("Select Any Page from here")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    #st.image(image, caption="Uploaded Image", use)

    # Run the model to identify the Pokémon

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