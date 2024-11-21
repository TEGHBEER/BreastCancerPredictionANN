import streamlit as st
import pandas as pd
from data_preprocessing import load_and_preprocess_data
from feature_selection import select_features
from ann_model import train_ann
from model_tuning import tune_model  # Import the tune_model function
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess data
X_scaled, y, scaler = load_and_preprocess_data()

# Feature selection
X_selected = select_features(X_scaled, y)

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
- **Mean Texture (Standard Error)**: A measure of the variation in pixel intensities within the cell's nucleus.
- **Mean Perimeter (mm)**: The average length of the boundary of the nucleus.
- **Mean Area (mm²)**: The average area occupied by the nucleus.
- **Mean Smoothness (scaled)**: A measure of the smoothness of the nucleus boundary, where lower values represent smoother contours.
- **Mean Compactness (scaled)**: A measure of the shape of the nucleus, specifically the compactness or how tightly the nucleus is packed.
- **Mean Concavity (scaled)**: Measures the degree to which the boundary of the nucleus is concave.
- **Mean Concave Points (scaled)**: A measure of how many concave points are found along the boundary of the nucleus.
- **Mean Symmetry (scaled)**: A measure of how symmetrical the shape of the nucleus is.
- **Mean Fractal Dimension (scaled)**: A measure of the complexity of the boundary of the nucleus.

Each of these features is derived from a set of images of cell nuclei, and by adjusting the sliders below, you can input values for these features to predict whether the condition is benign or malignant.

Please adjust the values of each feature using the sliders below and click **'Predict'** to see the result.
""")

# Display feature names in bold and capitalized
st.subheader("Please select the values for the following features:")

user_input = {}

# Loop through features and display sliders
for i, feature in enumerate(['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
                              'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 
                              'mean fractal dimension']):
    feature_name = feature.upper()  # Capitalize feature name
    user_input[feature] = st.slider(f"**{feature_name}**", 
                                    min_value=float(X_scaled[:, i].min()), 
                                    max_value=float(X_scaled[:, i].max()))

# Button for prediction
if st.button("Predict"):
    # Create a DataFrame for the user's input
    user_data = pd.DataFrame(user_input, index=[0])

    # Make a prediction
    prediction = model.predict(user_data)
    result = "Malignant" if prediction[0] == 1 else "Benign"
    st.write(f"Prediction: **{result}**")
