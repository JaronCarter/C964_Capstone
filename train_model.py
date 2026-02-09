import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import joblib

# Read CSV data into a data fram object.
df = pd.read_csv("data/kc_house_data.csv")

# Create a FEATURES list detailing each column we'd like to use for training data by header.
FEATURES = [
    "sqft_living",
    "bedrooms",
    "bathrooms",
    "sqft_lot",
    "yr_built",
    "grade",
    "waterfront",
    "view",
    "yr_renovated"
]

# Set the TARGET we'd like the model to evaluate by and predict for.
TARGET = "price"

# Reselect the datafram data with only the features and target columns selected.
df = df[FEATURES + [TARGET]]
# Drop potential missing data rows from dataframe and reassign to dataframe variable.
df = df.dropna()

# Winsorize the target data so as to not have sharp outliers affect prediction of model.
upper_cap = df[TARGET].quantile(0.99)
df[TARGET] = df[TARGET].clip(upper=upper_cap)

# Instantiate X and Y variables to use for split training utilizing the respective dataframe data.
X, y = df[FEATURES], df[TARGET]

# Split 80% of data for training and the remaining 20% of data for testing.
x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=73)

# Train the linear regression model using the split data.
model = LinearRegression().fit(x_train, y_train)

# Get the R^2 score for how good the data is on this regression model.
score = model.score(x_test, y_test)

# Create a prediction variable based on the test data. This can be verified against the expected y_test data and calculate the differences for MAE and RMSE accordingly.
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print(f"CSV data read and used to train the model successfully.\nR^2: {score}\nMAE: {mae}\nRMSE: {rmse}\nExporting serialized model as a pkl file now.")

# Create a python dictionary object to store for pkl serialization.
data = {
    "model": model,
    "features": FEATURES,
    "target": TARGET,
    "mae": mae
}

# Serialize data to pkl.
joblib.dump(data, "model/model.pkl")

print("Model successfully dumped to model/model.pkl")