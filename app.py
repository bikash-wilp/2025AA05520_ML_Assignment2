import streamlit as st
import joblib
import numpy as np

model = joblib.load("model/random_forest.pkl")

st.title("Classification Prediction App")

features = st.text_input("Enter features comma separated")

if st.button("Predict"):
    data = np.array([float(i) for i in features.split(",")]).reshape(1,-1)
    prediction = model.predict(data)
    st.write("Prediction:", prediction[0])