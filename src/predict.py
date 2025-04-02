import joblib
import numpy as np
import pandas as pd

# Load the trained model
model_name = "xgboost"  # Adjust according to your model
model = joblib.load(f"models/{model_name}.pkl")

# Load the scaler and feature names
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Example input data (user-provided)
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)

# Convert input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape to be 2D (for a single prediction)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Create a DataFrame with feature names
input_df = pd.DataFrame(input_data_reshaped, columns=feature_names)

# Scale the numerical features (important for models like Logistic Regression)
numerical_features = ["age", "resting_blood_pressure", "cholesterol", "max_heart_rate_achieved", "st_depression"]
input_df[numerical_features] = scaler.transform(input_df[numerical_features])

# Make prediction
prediction = model.predict(input_df)

# Output the result
if prediction[0] == 0:
    print("The Person does not have a Heart Disease")
else:
    print("The Person has Heart Disease")
