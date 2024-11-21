import streamlit as st
import pandas as pd
from data_preprocessing import load_and_preprocess_data
from feature_selection import select_features
from ann_model import train_ann
from model_tuning import tune_model  # Import the tune_model function
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess data (without scaling)
X, y = load_and_preprocess_data()  # Now we just load the raw data without scaling

# Feature selection
X_selected, selector = select_features(X, y)  # Get both the selected data and the selector

# Get the selected feature names from the selector
selected_columns = X.columns[selector.get_support()]

# Create a DataFrame with the selected features
X_selected = pd.DataFrame(X_selected, columns=selected_columns)

# Model tuning using the tune_model function
model = tune_model(X_selected, y)  # Get the best model from grid search

# Model training
train_ann(X_selected, y, model)

# Streamlit UI for interaction
st.title('Breast Cancer Prediction using ANN')
st.write("Welcome to the Breast Cancer Prediction app!")

st.write("""
This app uses an Artificial Neural Network (ANN) to predict breast cancer classification as **Malignant** or **Benign**. 
The prediction is based on the following 10 features, which represent specific properties of cell nuclei present in breast cancer cell images:

- **Mean Radius (mm)**: The average distance from the center to the outermost point of the cell nucleus.
- **Mean Perimeter (mm)**: The average length of the boundary of the nucleus.
- **Mean Area (mm²)**: The average area occupied by the nucleus.
- **Mean Concavity**: Measures the degree to which the boundary of the nucleus is concave.
- **Mean Concave Points**: A measure of how many concave points are found along the boundary of the nucleus.
- **Worst Radius (mm)**: The maximum distance from the center to the outermost point of the nucleus in the worst-case scenario.
- **Worst Perimeter (mm)**: The maximum length of the boundary of the nucleus in the worst-case scenario.
- **Worst Area (mm²)**: The maximum area occupied by the nucleus in the worst-case scenario.
- **Worst Concavity**: Measures the degree of concavity in the boundary of the nucleus in the worst-case scenario.
- **Worst Concave Points**: A measure of how many concave points are found along the boundary of the nucleus in the worst-case scenario.

Each of these features is derived from a set of images of cell nuclei, and by adjusting the sliders below, you can input values for these features to predict whether the condition is benign or malignant.
""")

# Display feature names in bold and capitalized
st.subheader("Please select the values for the following features:")

user_input = {}

# Loop through selected features and display sliders
for feature in selected_columns:  # Loop through the selected columns
    feature_name = feature.upper()  # Capitalize feature name
    user_input[feature] = st.slider(f"**{feature_name}**", 
                                    min_value=float(X[feature].min()),  # Access columns using the feature name
                                    max_value=float(X[feature].max()))

# Create a DataFrame for the user's input
user_data = pd.DataFrame(user_input, index=[0])

# Make a prediction automatically as the sliders change
prediction = model.predict(user_data)
result = "Malignant" if prediction[0] == 1 else "Benign"

# Display the prediction result
st.write(f"Prediction: **{result}**")
