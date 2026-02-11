# Streamlit application for estimating home prices using a trained ML model and providing exploratory data visualizations.
# Created by Jaron Carter

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import datetime
import pandas as pd

# Define cached loaders for the model and dataset to improve performance during reactive rerenders.
@st.cache_resource
def load_model():
    return joblib.load("model/model.pkl")
@st.cache_data
def load_dataset():
    return pd.read_csv("data/kc_house_data.csv")

# Set page layout to wide view by default. Add markdown to reduce default Streamlit vertical padding.
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load pkl bundle and instantiate variables with the subsequent passed objects plus a datetime for UI year selection.
bundle = load_model()
model = bundle["model"]
FEATURES = bundle["features"]
mae = bundle["mae"]
current_year = datetime.date.today().year

# Read CSV data for comprehensive chart visuals.
df = load_dataset()

st.title("Home Price Estimator")
st.caption(
    "This application uses a custom-trained machine learning model to generate predictions "
    "based on patterns learned from the King County dataset."
)

# Split UI into two main views.
left_col, right_col = st.columns([8, 7])

# Isolate the input and prediction UI into a Streamlit fragment function to prevent unnecessary rerenders of components outside of the fragment scope.
@st.fragment
def input_section():
    st.subheader("Property Details")
    st.write("Enter property details to estimate a home price.")

    # Create 3 columns for better UI layout. Add Number input boxes for each corresponding expected model field.
    col1, col2 = st.columns(2)

    with col1:
        sqft_living = st.number_input("Living Area (sqft)", min_value=0, step=10)
        sqft_lot = st.number_input("Lot Area (sqft)", min_value=sqft_living, step=100)
        yr_built = st.number_input(
            "Year Built",
            min_value=1900,
            max_value=current_year,
            value=current_year,
            step=1
        )

    with col2:
        bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
        bathrooms = st.number_input(
            "Bathrooms", min_value=0.0, step=0.5, format="%.1f"
        )

    st.divider()

    # Initialize an array, matching model feature list order, that contains each input state.
    input_data = [
        sqft_living,
        bedrooms,
        bathrooms,
        sqft_lot,
        yr_built
    ]

    # Create a dataframe using the user's input data and the model's expected feature columns.
    input_df = pd.DataFrame([input_data], columns=FEATURES)

    # Generate the prediction using the newly created user input dataframe. The predict method returns an array, so grab the first index for a more human readable output.
    prediction = model.predict(input_df)[0]

    # Conditional check to make sure a valid prediction is available before showing the prediction and MAE estimates on screen to the user.
    if prediction > 0:
        st.markdown(
            f"""
        **Estimated Home Value:**  
        ${prediction:,.2f}

        **Estimated Error (MAE):**  
        ± ${mae:,.2f}
        """
        )

    else:
        st.markdown("_Please enter property attributes to generate an estimate._")

# Left view for Machine Learning Predictor Input and Output layer.
with left_col:
    input_section()

# Right side view for containing charts and visuals of main data.
with right_col:
    st.subheader("Data Representation Views")

    # Create a price by living sqft scatter plot view using a subplot, to keep from having shared axis', and set labels plus a formatter which takes a lambda expression for the yaxis indicators to be more legible.
    fig1, ax1 = plt.subplots(figsize=(7,2))
    ax1.scatter(df["sqft_living"], df["price"],alpha=0.3)
    ax1.set_xlabel("Living Area (sqft)")
    ax1.set_ylabel("Price")
    ax1.set_title("Living Area vs Sale Price")
    ax1.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"${y/1000000:.0f}M")
    )
    st.pyplot(fig1)
    plt.close(fig1)

    # Create two columns for the remaining two charts to display neatly.
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        
        # Create a sales price histogram, set labels, and add a formatter lambda for the xaxis indicators to improve human readability.
        fig2, ax2 = plt.subplots()
        ax2.hist(df["price"], bins=500)
        ax2.set_xlabel("Sale Price (USD)")
        ax2.set_ylabel("Number of Homes Sold")
        ax2.set_title("Home Price Distribution")
        ax2.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"${x/1_000_000:.0f}M")
        )
        st.pyplot(fig2)
        plt.close(fig2)

    with viz_col2:

        # Create a price per bedrooms boxplot, set labels, and add a formatter lambda for the yaxis indicators to improve human readability.
        fig3, ax3 = plt.subplots()
        df.boxplot(column="price", by="bedrooms", ax=ax3)
        plt.suptitle("")
        ax3.set_xlabel("Number of Bedrooms")
        ax3.set_ylabel("Sale Price (USD)")
        ax3.set_title("Price by Total \nNumber of Bedrooms")
        ax3.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"${y/1_000_000:.0f}M")
        )
        st.pyplot(fig3)
        plt.close(fig3)
        
st.caption("— Created by Jaron Carter")
