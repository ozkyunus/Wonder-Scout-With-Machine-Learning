import streamlit as st
import plotly.express as px
import pandas as pd
from Helper import get_data

df = get_data()


for col in ["Age", "Value", "Overall_rating"]:
    if col in df.columns:
        df[col] = df[col].fillna(0)
        df[col] = df[col].clip(lower=0)

st.title("Veri Analizi")

# 1 Yaş - Piyasa Değeri
fig_age_value = px.scatter(
    df,
    x="Age",
    y="Value",
    color="Overall_rating",
    size="Overall_rating",
    hover_data=["Best_position", "Club_level", "Player_rank"],
    title="Yaş ve Piyasa Değeri İlişkisi"
)
st.plotly_chart(fig_age_value, use_container_width=True)

# 2 Overall Rating - Piyasa Değeri
fig_rating_value = px.scatter(
    df,
    x="Overall_rating",
    y="Value",
    color="Age",
    size="Value",
    hover_data=["Best_position", "Club_level", "Player_rank"],
    title="Genel Rating ve Piyasa Değeri"
)
st.plotly_chart(fig_rating_value, use_container_width=True)

# 3 Kulüp Seviyesi - Ortalama Piyasa Değeri
club_avg = df.groupby("Club_level", as_index=False)["Value"].mean()
fig_club_value = px.bar(
    club_avg,
    x="Club_level",
    y="Value",
    color="Value",
    title="Kulüp Seviyesi Bazında Ortalama Piyasa Değeri"
)
st.plotly_chart(fig_club_value, use_container_width=True)

# 4 Pozisyonlara Göre Ortalama Piyasa Değeri
pos_avg = df.groupby("Best_position", as_index=False)["Value"].mean()
fig_pos_value = px.bar(
    pos_avg,
    x="Best_position",
    y="Value",
    color="Value",
    title="Pozisyon Bazında Ortalama Piyasa Değeri"
)
st.plotly_chart(fig_pos_value, use_container_width=True)
