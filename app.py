import streamlit as st
import joblib

bundle = joblib.load("model/model.pkl")

model = bundle["model"]
FEATURES = bundle["features"]
TARGET = bundle["target"]
mae = bundle["mae"]

st.title("Home Price Estimator")
st.write("Enter property details to estimate a home price.")

sqft_living = st.number_input("Living Area (sqft)", min_value=0, step=10)
bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
bathrooms = st.number_input("Bathrooms", min_value=0, step=1)
