# Residential Price Estimation Application  
C964 – Computer Science Capstone  
Author: Jaron Carter  

---

## Overview

This project is a machine learning–based residential real estate price estimation application developed as part of the WGU C964 Computer Science Capstone.

The application uses a trained Linear Regression model to generate residential price predictions based on property characteristics. It integrates predictive modeling with descriptive data visualizations to support valuation analysis and decision-making.

The system consists of:

- A data preprocessing and model training script  
- A serialized machine learning model  
- A Streamlit-based interactive web application  
- A publicly deployed version for browser-based access  

---

## Features

- Supervised Linear Regression model for price prediction  
- Data preprocessing including missing value removal and Winsorization  
- Train-test split for reproducible model evaluation  
- Model evaluation using MAE and RMSE  
- Interactive Streamlit dashboard  
- Three descriptive visualizations:
  - Scatter plot (Living Area vs Price)
  - Histogram (Sale Price Distribution)
  - Box plot (Price by Bedrooms)
- Real-time valuation scenario simulation  
- Modular architecture separating training and deployment  

---

## Dataset

The model was trained using the publicly available King County housing dataset.  
The dataset includes residential transaction records with structural property features and historical sale prices.

Preprocessing steps included:

- Removal of missing values using Panda's `dropna()` method  
- Winsorization of selected numerical features  
- Feature selection for regression modeling  
- Train-test split with fixed random state for reproducibility  

---

## Deployment

### Hosted Version (Recommended)

The application is deployed via Streamlit and can be accessed through a web browser:

https://c964capstone-jaroncarter.streamlit.app/

No installation is required for hosted use.

---

## Local Installation

To run the application locally:

1. Clone the repository:

    ```
   git clone https://github.com/JaronCarter/C964_Capstone.git
    ```

3. Navigate into the project directory.

4. Install dependencies:

    ```
   pip install -r requirements.txt
    ```

5. Launch the application:

    ```
   streamlit run streamlit_app.py
    ```

6. Open the provided local URL in a web browser (typically http://localhost:8501).

---

## Model Training (Optional)

To retrain the model:

    python train_model.py

This regenerates the serialized model file used by the Streamlit application.

---

## Project Structure

```
C964_Capstone/
│
├── data/
├── model/
├── streamlit_app.py
├── train_model.py
├── requirements.txt
└── README.md
```

- `train_model.py` – Handles preprocessing, training, and evaluation  
- `streamlit_app.py` – User interface and prediction logic  
- `requirements.txt` – Python dependencies  
- `model/` – Serialized trained model file directory
- `data/` – Directory housing static data for model training

---

## Model Evaluation

The final model achieved a Mean Absolute Error (MAE) of approximately ± $152,920.60.

MAE is displayed within the application to provide transparency regarding prediction uncertainty.

---

## Security and Data Considerations

- No personally identifiable information (PII) is used.  
- The dataset contains only structural property attributes and historical sale prices.  
- The deployed application uses HTTPS via Streamlit hosting.  
- No user input data is stored or persisted.  

---

## Author

Jaron Carter  
Computer Science – WGU  
Software Engineer and Project Lead  

