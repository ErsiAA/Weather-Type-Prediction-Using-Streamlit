import pandas as pd
import pickle
import streamlit as st
import base64
from sklearn.preprocessing import LabelEncoder


model_filename = 'weather_prediction_model.pkl'
features_filename = 'training_features.pkl'
label_encoder_filename = 'label_encoder.pkl'

with open(model_filename, 'rb') as file:
    model = pickle.load(file)
with open(features_filename, 'rb') as file:
    training_features = pickle.load(file)
with open(label_encoder_filename, 'rb') as file:
    weather_type_encoder = pickle.load(file)

def load_image(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    return f"data:image/png;base64,{encoded_string}"

image_path = "download.jpg" 
background_image = load_image(image_path)
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("{background_image}");
        background-size: cover;
        background-position: center;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("Prediksi Jenis Cuaca")
st.markdown("Gunakan aplikasi ini untuk memprediksi jenis cuaca berdasarkan input dari pengguna.")


st.sidebar.header("Masukkan Data Cuaca")
def user_input_features():
    Temperature = st.sidebar.slider('Temperature (°C)', -30, 50, 20)
    Humidity = st.sidebar.slider('Humidity (%)', 0, 100, 50)
    Wind_Speed = st.sidebar.slider('Wind Speed (km/h)', 0, 50, 10)
    Precipitation = st.sidebar.slider('Precipitation (%)', 0, 100, 20)
    Atmospheric_Pressure = st.sidebar.slider('Atmospheric Pressure (hPa)', 800, 1200, 1005)
    UV_Index = st.sidebar.slider('UV Index', 0, 14, 5)
    Visibility = st.sidebar.slider('Visibility (km)', 0, 20, 10)
    Cloud_Cover = st.sidebar.selectbox('Cloud Cover', ['clear', 'partly cloudy', 'overcast'])
    Season = st.sidebar.selectbox('Season', ['Winter', 'Spring', 'Summer', 'Autumn'])
    Location = st.sidebar.selectbox('Location', ['inland', 'coastal', 'mountain'])

    input_data = {
        'Temperature': Temperature,
        'Humidity': Humidity,
        'Wind Speed': Wind_Speed,
        'Precipitation (%)': Precipitation,
        'Atmospheric Pressure': Atmospheric_Pressure,
        'UV Index': UV_Index,
        'Visibility (km)': Visibility,
        'Cloud Cover': Cloud_Cover,
        'Season': Season,
        'Location': Location
    }
    return input_data

input_data = user_input_features()


input_df = pd.DataFrame([input_data])


label_encoders = {
    'Cloud Cover': LabelEncoder(),
    'Season': LabelEncoder(),
    'Location': LabelEncoder()
}


for col, le in label_encoders.items():
    le.fit(['clear', 'partly cloudy', 'overcast'] if col == 'Cloud Cover' else 
           ['Winter', 'Spring', 'Summer', 'Autumn'] if col == 'Season' else 
           ['inland', 'coastal', 'mountain'])
    input_df[col] = le.transform(input_df[col])


input_df = input_df.reindex(columns=training_features, fill_value=0)


prediction = model.predict(input_df)[0]
predicted_label = weather_type_encoder.inverse_transform([prediction])[0]


st.subheader("Hasil Prediksi")
st.write(f"Jenis cuaca yang diprediksi adalah: **{predicted_label}**")







#Kelompok 2 (Ersi Aditya Al'Amin)