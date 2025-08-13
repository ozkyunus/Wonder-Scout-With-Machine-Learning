import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
st.set_page_config(layout="wide")
# Mevcut pipeline import
sys.path.append('/Users/yunusemreozkaya/PycharmProjects/PythonProject11/machine_learning')
from ScoutMLPipeline import players_data_prep


# 1 MODEL & VERİ YÜKLEME

@st.cache_resource
def load_model():
    return joblib.load("best_lgb_model.pkl")

@st.cache_data
def load_fc2425_data():
    df = pd.read_csv("Fc2425Corr.csv")
    X, y = players_data_prep(df)
    return df, X, y

@st.cache_data
def load_filtered_display_data():
    return pd.read_csv("filtered2_df.csv")

model = load_model()
df_fc2425, X_fc2425, y_fc2425 = load_fc2425_data()
df_display = load_filtered_display_data()


# 2 RANDOM FUTBOLCU SEÇİM FONKSİYONU

def select_random_player(data):
    return data.sample(1, random_state=np.random.randint(0, 10000))

def predict_value(player_df):
    y_pred_log = model.predict(player_df)
    return np.exp(y_pred_log)[0]


# 3 İLK ÇALIŞMA - Başlangıçta rastgele oyuncu seç (display data üzerinden)

if "current_display_player" not in st.session_state:
    st.session_state.current_display_player = select_random_player(df_display)

# İlk market value filtered2_df'den direkt alınıyor
if "market_value" not in st.session_state:
    st.session_state.market_value = st.session_state.current_display_player["Value"].values[0]


# 4 GÖSTERİM (filtered2_df den gelen veri + market value)


# Futbolcu bilgilerini güzel bir tasarımla göster
player_info = st.session_state.current_display_player.iloc[0]

# Futbolcu ismini en başta göster
st.markdown(f"""
<div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 20px; border-radius: 10px; margin: 10px 0; text-align: center;">
    <h1 style="color: white; margin: 0; font-size: 2.5em;">👤 {player_info['Name']}</h1>
</div>
""", unsafe_allow_html=True)

# Temel bilgiler - filtered2_df'de mevcut olan sütunları kullan
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("👤 Age", f"{player_info['Age']} years")
    st.metric("📏 Height", f"{player_info['Height']} cm")

with col2:
    st.metric("⚖️ Weight", f"{player_info['Weight']} kg")
    st.metric("⭐ Overall Rating", f"{player_info['Overall_rating']}")

with col3:
    st.metric("🎯 Potential", f"{player_info['Potential']}")
    st.metric("🏃‍♂️ Total Movement", f"{player_info['Total_movement']}")

with col4:
    st.metric("⚡ Total Power", f"{player_info['Total_power']}")
    st.metric("🧠 Total Mentality", f"{player_info['Total_mentality']}")

# İkinci satır
col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric("🎨 Total Skill", f"{player_info['Total_skill']}")
    st.metric("🛡️ Total Defending", f"{player_info['Total_defending']}")

with col6:
    st.metric("🦶 Weak Foot", f"{player_info['Weak_foot']}/5")
    st.metric("🎭 Skill Moves", f"{player_info['Skill_moves']}/5")

with col7:
    st.metric("🌍 International Reputation", f"{player_info['International_reputation']}/5")
    st.metric("🏃 Body Type", player_info['Body_type'])

with col8:
    st.metric("⚽ Best Position", player_info['Best_position'])
    st.metric("🎯 Best Overall", f"{player_info['Best_overall']}")


st.markdown(f"""
<div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); padding: 15px; border-radius: 10px; margin: 20px 0; text-align: center;">
    <h3 style="color: white; margin: 0;">💰 Mevcut Piyasa Değeri</h3>
    <h2 style="color: white; margin: 10px 0; font-size: 2em;">{st.session_state.market_value:,.0f} €</h2>
</div>
""", unsafe_allow_html=True)


# 5 YAŞ ARALIĞI FİLTRE

age_min = int(df_display["Age"].min())
age_max = int(df_display["Age"].max())
selected_age = st.slider("⚽ Futbolcunun yaşını seç", age_min, age_max, (age_min, age_max))

filtered_players = df_display[(df_display["Age"] >= selected_age[0]) & (df_display["Age"] <= selected_age[1])]

if filtered_players.empty:
    st.warning("⚠️ Seçilen yaş aralığında futbolcu bulunamadı.")
else:
    if st.button("🎲 Rastgele Futbolcu Seç", type="primary"):
        st.session_state.current_display_player = select_random_player(filtered_players)
        # Yeni seçilen oyuncunun market value'su filtered2_df den geliyor
        st.session_state.market_value = st.session_state.current_display_player["Value"].values[0]
        st.rerun()  # Bu satırı bozma dedin


# 6 MODEL VERİSİNDEN EŞLEŞEN SATIRI BUL

def find_matching_player_row(display_player):
    # display_player: filtered2_df'den tek satırlık df
    # Daha esnek eşleştirme için sadece temel sütunları kullan
    cols_to_match = ["Age", "Overall_rating", "Height", "Weight"]
    
    mask = pd.Series(True, index=df_fc2425.index)
    for col in cols_to_match:
        if col in display_player.columns and col in df_fc2425.columns:
            mask &= (df_fc2425[col] == display_player.iloc[0][col])
    
    matched = df_fc2425[mask]
    if matched.empty:
        return None
    else:
        return matched.iloc[0]

model_player_row = find_matching_player_row(st.session_state.current_display_player)
if model_player_row is None:
    st.warning("Model verisinde eşleşen futbolcu bulunamadı. Özellikleri düzenleyemezsin.")
    st.stop()


# 7 ÖZELLİKLERİ DÜZENLEME (Fc2425Corr.csv formatında model_player_row üzerinden)

st.markdown("""
<div style="background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
    <h2 style="color: white; text-align: center; margin: 0;">⚽ Futbolcu Özelliklerini Düzenle</h2>
    <p style="color: white; text-align: center; margin: 10px 0; opacity: 0.9;">Futbolcunun özelliklerini değiştirerek yeni piyasa değerini tahmin et!</p>
</div>
""", unsafe_allow_html=True)

# Model verisinden eşleşen satırı al ve X_fc2425'teki karşılığını bul
# Yeni oyuncu seçildiğinde model_player_row güncellenir, bu yüzden her seferinde yeniden al
model_player_data = X_fc2425.iloc[0].copy()

# Sayısal alanlar - model_player_row'dan güncel değerleri al
model_player_data["Age"] = st.slider("Age", 16, 45, int(model_player_row["Age"]))
model_player_data["Overall_rating"] = st.slider("Overall Rating", 40, 99, int(model_player_row["Overall_rating"]))
model_player_data["Height"] = st.slider("Height (cm)", 150, 210, int(model_player_row["Height"]))
model_player_data["Weight"] = st.slider("Weight (kg)", 50, 110, int(model_player_row["Weight"]))

# Tek seçimli kategorikler (one-hot encoding)
def one_hot_select_with_labels(options, label, df_row, user_labels, rare_mapping=None):
    """
    Kullanıcı dostu etiketlerle one-hot encoding yapar
    options: one-hot encoding sütunları
    user_labels: kullanıcıya gösterilecek etiketler
    rare_mapping: rare değerlerin mapping'i
    """
    # Mevcut seçimi bul - model_player_row'dan orijinal değeri al
    # model_player_row orijinal format (Best_position, Body_type, Player_rank)
    # options one-hot encoding format (Best_position_ST, Body_type_Normal, etc.)
    
    # Orijinal sütun adını bul
    original_column = label.replace(" ", "_").lower()
    if "best" in original_column:
        original_column = "Best_position"
    elif "body" in original_column:
        original_column = "Body_type"
    elif "player" in original_column:
        original_column = "Player_rank"
    
    # Orijinal değeri al
    original_value = model_player_row[original_column]
    
    # One-hot encoding sütun adından kullanıcı etiketini çıkar
    # Örnek: "Best_position_ST" -> "ST"
    current_label = original_value
    
    # Eğer current_label user_labels'da yoksa, ilk etiketi kullan
    try:
        default_index = user_labels.index(current_label)
    except ValueError:
        default_index = 0
    
    # Kullanıcıya etiketleri göster
    choice = st.selectbox(label, user_labels, index=default_index)
    
    # Seçilen etiketi one-hot encoding'e çevir
    for opt in options:
        if rare_mapping and choice in rare_mapping:
            # Rare değer seçildiyse, rare sütununu 1 yap
            df_row[opt] = 1 if "Rare" in opt else 0
        else:
            # Normal değer seçildiyse, ilgili sütunu 1 yap
            if choice in opt:
                df_row[opt] = 1
            elif "Rare" in opt and choice not in user_labels:
                # Eğer seçilen değer user_labels'da yoksa, rare sütununu 1 yap
                df_row[opt] = 1
            else:
                df_row[opt] = 0

# Best Position için kullanıcı dostu etiketler
best_position_labels = ['CAM', 'ST', 'RW', 'CM', 'CDM', 'RM', 'LM', 'CB', 'RB', 'GK', 'LB', 'LW', 'CF', 'LWB', 'RWB']
best_positions = [col for col in X_fc2425.columns if col.startswith("Best_position_")]
# Best_position için rare mapping - kullanıcı etiketlerinde olmayan değerler rare olarak gidecek
one_hot_select_with_labels(best_positions, "Best Position", model_player_data, best_position_labels, best_position_labels)

# Body Type için kullanıcı dostu etiketler
body_type_labels = ['Normal', 'Lean', 'Stocky', 'Unique']
body_types = [col for col in X_fc2425.columns if col.startswith("Body_type_")]
# Unique -> Rare mapping
body_type_rare_mapping = ['Unique']
one_hot_select_with_labels(body_types, "Body Type", model_player_data, body_type_labels, body_type_rare_mapping)

# Player Rank için kullanıcı dostu etiketler
player_rank_labels = ['Young', 'Old', 'WonderKid', 'Expert']
player_ranks = [col for col in X_fc2425.columns if col.startswith("Player_rank_")]
one_hot_select_with_labels(player_ranks, "Player Rank", model_player_data, player_rank_labels)

# Diğer numeric alanlar - model_player_row'dan güncel değerleri al
model_player_data["Club_level"] = st.slider("Club Level", 1, 10, int(model_player_row["Club_level"]))
model_player_data["Skill_moves"] = st.slider("Skill Moves", 1, 5, int(model_player_row["Skill_moves"]))
model_player_data["Weak_foot"] = st.slider("Weak Foot", 1, 5, int(model_player_row["Weak_foot"]))
model_player_data["International_reputation"] = st.slider("International Reputation", 1, 5, int(model_player_row["International_reputation"]))

model_player_data["Total_skill"] = st.slider("Total Skill", 0, 500, int(model_player_row["Total_skill"]))
model_player_data["Total_movement"] = st.slider("Total Movement", 0, 500, int(model_player_row["Total_movement"]))
model_player_data["Total_power"] = st.slider("Total Power", 0, 500, int(model_player_row["Total_power"]))
model_player_data["Total_mentality"] = st.slider("Total Mentality", 0, 500, int(model_player_row["Total_mentality"]))
model_player_data["Total_defending"] = st.slider("Total Defending", 0, 500, int(model_player_row["Total_defending"]))


# 8 TAHMİN BUTONU

st.markdown("""
<div style="background: linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%); padding: 15px; border-radius: 10px; margin: 20px 0; text-align: center;">
    <h3 style="color: white; margin: 0;">🎯 Yeni Piyasa Değeri Tahmini</h3>
</div>
""", unsafe_allow_html=True)

if st.button("🚀 Tahmin Et!", type="primary", use_container_width=True):
    # DataFrame formatına al
    df_for_pred = pd.DataFrame([model_player_data])
    
    # Model tahmini yap
    new_value = predict_value(df_for_pred)
    
    # Yeni değeri session state'e kaydet (otomatik analiz için)
    st.session_state.last_prediction_value = new_value
    
    # Değer değişimini hesapla
    value_change = new_value - st.session_state.market_value
    value_change_percentage = (value_change / st.session_state.market_value) * 100
    
    # Sonucu özel bir kutu içinde göster
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); padding: 20px; border-radius: 10px; margin: 20px 0; text-align: center;">
        <h3 style="color: white; margin: 0;">🎉 Tahmin Sonucu</h3>
        <h2 style="color: white; margin: 10px 0; font-size: 2.5em;">{new_value:,.0f} €</h2>
        <p style="color: white; margin: 10px 0; opacity: 0.9;">Yeni Piyasa Değeri</p>
        <p style="color: white; margin: 5px 0; font-size: 1.2em;">
            {'📈' if value_change >= 0 else '📉'} Değişim: {value_change:+,.0f} € ({value_change_percentage:+.1f}%)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
