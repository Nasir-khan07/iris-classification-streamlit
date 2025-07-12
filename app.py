import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load('iris_model.pkl')
scaler = joblib.load('scaler.pkl')

species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

st.title("🌸 Iris Flower Species Predictor")

st.write("Use the sliders to input measurements and click Predict!")

sepal_length = st.slider('Sepal length (cm)', 4.0, 8.0, 5.1)
sepal_width = st.slider('Sepal width (cm)', 2.0, 4.5, 3.5)
petal_length = st.slider('Petal length (cm)', 1.0, 7.0, 1.4)
petal_width = st.slider('Petal width (cm)', 0.1, 2.5, 0.2)

if st.button('Predict'):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]
    species = species_map[prediction]
    st.success(f"✅ Predicted species: **{species}**")
