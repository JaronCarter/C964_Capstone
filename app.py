import streamlit as st
import joblib
import datetime
import pandas as pd

bundle = joblib.load("model/model.pkl")

model = bundle["model"]
FEATURES = bundle["features"]
TARGET = bundle["target"]
mae = bundle["mae"]
current_year = datetime.date.today().year

st.title("Home Price Estimator")
st.write("Enter property details to estimate a home price.")

col1, col2, col3 = st.columns(3)

with col1:
    sqft_living = st.number_input("Living Area (sqft)", min_value=0, step=10)
    sqft_lot = st.number_input("Lot Area (sqft)", min_value=0, step=100)
with col2:
    bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=0, step=1)
with col3:
    yr_built = st.number_input(
        "Enter the year built",
        min_value=1900,
        max_value=current_year,
        value=current_year,
        step=1
    )

input_data = [
    sqft_living,
    bedrooms,
    bathrooms,
    sqft_lot,
    yr_built
]

input_df = pd.DataFrame([input_data], columns=FEATURES)

prediction = model.predict(input_df)[0]

if prediction > 0:
    st.html(f"<pre>Your home is estimated to be worth: <strong>${prediction:,.2f}  +-</strong>${mae:,.2f}</pre>")
else:
    st.html("<pre>Please enter in a few attributes that will help with prediction.</pre>")