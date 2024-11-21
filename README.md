# Breast Cancer Prediction Using Artificial Neural Network (ANN)

## Overview

This project aims to predict breast cancer classifications using an Artificial Neural Network (ANN). The dataset used for this project is the well-known Breast Cancer dataset from the `sklearn.datasets` library. The goal is to classify tumors as **Malignant** (cancerous) or **Benign** (non-cancerous) based on a set of features extracted from breast cancer cell images. These features represent various properties of cell nuclei and can help in determining the nature of the tumor.

The project follows a typical machine learning pipeline, including data preprocessing, feature selection, model training, and tuning, followed by making predictions using a trained model. The final application uses Streamlit to create an interactive web interface, allowing users to input values and predict the classification of a tumor.

---

## Steps Involved

### 1. **Data Preprocessing**
In the first step, the raw Breast Cancer dataset is loaded using the `load_breast_cancer()` function from `sklearn.datasets`. The dataset contains 30 features related to cell nuclei properties and the target variable, which indicates whether the tumor is malignant or benign.

- The data is split into features (X) and target labels (y).
- Features represent different properties of the cell nucleus, while the target variable represents the tumor classification (0 for benign, 1 for malignant).

### 2. **Feature Selection**
The next step involves selecting the most important features that contribute significantly to the classification task. This is done using the **ANOVA F-test** (`f_classif`), which measures the relationship between the features and the target variable.

- **Top 10 Features:** The top 10 features that have the highest correlation with the target are selected for model training. This reduces the dimensionality and focuses the model on the most important information.

### 3. **Model Training**
After selecting the relevant features, the next step is to train an Artificial Neural Network (ANN) using the selected features. The model is trained to recognize patterns in the data and learn the relationships between the features and the target variable.

- **Model Evaluation:** The performance of the model is evaluated using accuracy and other metrics like precision, recall, and F1-score. These metrics provide insights into the model's ability to correctly classify benign and malignant tumors.

### 4. **Model Tuning**
To improve the performance of the ANN model, hyperparameter tuning is performed using **GridSearchCV**. GridSearchCV evaluates different combinations of hyperparameters (such as the number of hidden layers, activation functions, and solvers) to find the best-performing model.

- **Hyperparameters:** The ANN model is tuned for parameters such as:
  - **Hidden Layer Sizes:** Defines the number of neurons in each layer.
  - **Activation Function:** The function that determines the output of each neuron.
  - **Solver:** The optimization algorithm used to train the model (e.g., Adam or SGD).
  - **Learning Rate:** Controls how quickly the model learns.

### 5. **Web Interface Using Streamlit**
Once the model is trained and tuned, the final step is to create a user-friendly interface where users can input values for the selected features and get predictions. This is done using **Streamlit**, a framework for building interactive web applications.

- **User Input:** The app allows users to select values for the 10 selected features via sliders.
- **Prediction:** Based on the input values, the trained model predicts whether the tumor is **Malignant** or **Benign**.

### 6. **Model Evaluation Metrics**
After training and tuning, the model's performance is evaluated using the following metrics:
- **Accuracy:** The overall accuracy of the model is approximately 92.6%, meaning that the model correctly classifies around 92.6% of the samples.
- **Precision, Recall, F1-Score:** These metrics help evaluate the model's ability to classify tumors correctly, considering both benign and malignant cases.

The results for the model's classification performance are:
- **Accuracy:** 92.6%
- **Precision (Malignant):** 0.91
- **Recall (Malignant):** 0.97
- **F1-Score (Malignant):** 0.94
- **Precision (Benign):** 0.95
- **Recall (Benign):** 0.84
- **F1-Score (Benign):** 0.90

---

## Project Workflow

1. **Load the Breast Cancer Dataset.**
2. **Preprocess the data (split into features and target).**
3. **Select the top 10 most significant features.**
4. **Train an Artificial Neural Network (ANN) model.**
5. **Tune the model using GridSearchCV to optimize hyperparameters.**
6. **Evaluate the model's performance (accuracy, precision, recall, F1-score).**
7. **Build a Streamlit app for interactive user input and predictions.**

---

## Technologies Used

- **Python**: The programming language used for the entire project.
- **Streamlit**: A framework for creating the interactive web interface.
- **Scikit-learn**: For machine learning algorithms, data preprocessing, feature selection, and model tuning.
- **Pandas & NumPy**: For data manipulation and processing.
- **Matplotlib & Seaborn**: For visualizing the data and model performance.
