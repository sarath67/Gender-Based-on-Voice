import streamlit as st
import numpy as np
import librosa
import joblib
import os
from io import BytesIO

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

model = joblib.load('gender_classifier.pkl')
scaler = joblib.load('scaler.pkl')

def predict_gender(file_path):
    audio_features = extract_features(file_path)
    audio_features_scaled = scaler.transform([audio_features])
    prediction = model.predict(audio_features_scaled)
    print(prediction)
    return "Male" if prediction[0] == 0 else "Female"

st.title("Gender Prediction from Audio")
st.write("Upload an audio file (.wav or .mp3) to predict the gender of the speaker.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    with open("temp_audio_file", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    file_path = "temp_audio_file"
    predicted_gender = predict_gender(file_path)
    st.write(f"Predicted Gender: {predicted_gender}")
    os.remove(file_path)
