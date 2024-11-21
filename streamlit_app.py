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

# You can also integrate model tuning later if needed
# tuned_model = tune_model(X_selected, y, model)

# Streamlit UI for interaction
st.title('Breast Cancer Prediction using ANN')
st.write("This app uses an artificial neural network to predict breast cancer classification based on features.")

user_input = {feature: st.slider(feature, min_value=float(X_scaled[:, i].min()), max_value=float(X_scaled[:, i].max())) for i, feature in enumerate(['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension'])}

# Create a DataFrame for the user's input
user_data = pd.DataFrame(user_input, index=[0])

# Make a prediction
prediction = model.predict(user_data)
st.write("Prediction: " + ("Malignant" if prediction[0] == 1 else "Benign"))
