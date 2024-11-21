import streamlit as st
import pandas as pd
from data_preprocessing import load_and_preprocess_data
from feature_selection import select_features
from ann_model import train_ann
from model_tuning import tune_model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess data
X_scaled, y, scaler = load_and_preprocess_data()

# Feature selection
X_selected = select_features(X_scaled, y)

# Model training
model = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=1000)
train_ann(X_selected, y, model)

# Streamlit UI for interaction
st.title('Breast Cancer Prediction using ANN')
st.write("Welcome to the Breast Cancer Prediction app!")
st.write("""
This app uses an Artificial Neural Network (ANN) to predict breast cancer classification as **Malignant** or **Benign**. 
The prediction is based on the following 10 features:
- **Mean Radius**
- **Mean Texture**
- **Mean Perimeter**
- **Mean Area**
- **Mean Smoothness**
- **Mean Compactness**
- **Mean Concavity**
- **Mean Concave Points**
- **Mean Symmetry**
- **Mean Fractal Dimension**

Each feature represents a specific property of cell nuclei present in breast cancer cell images. By adjusting the sliders, you can input values for these features, and the app will predict whether the condition is benign or malignant.

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

