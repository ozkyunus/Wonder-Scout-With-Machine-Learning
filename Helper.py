import pandas as pd
import streamlit as st

@st.cache_data
def get_data():
    df = pd.read_csv("Fc2425Corr.csv")
    return df