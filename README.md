# Breast Cancer Prediction Using Artificial Neural Network (ANN)

## Overview

This project aims to predict breast cancer classifications using an Artificial Neural Network (ANN). The dataset used for this project is the well-known Breast Cancer dataset from the `sklearn.datasets` library. The goal is to classify tumors as **Malignant** (cancerous) or **Benign** (non-cancerous) based on a set of features extracted from breast cancer cell images. These features represent various properties of cell nuclei and can help in determining the nature of the tumor.

The project follows a typical machine learning pipeline, including data preprocessing, feature selection, model training, and tuning, followed by making predictions using a trained model. The final application uses Streamlit to create an interactive web interface, allowing users to input values and predict the classification of a tumor.

---
### Dataset Details

The dataset contains **569 samples** and **30 features** (columns). Each feature represents a different property of cell nuclei, which can be used to predict whether the tumor is malignant or benign. The target variable indicates whether the tumor is malignant (1) or benign (0).

#### Dataset Shape:
- **Number of Samples (Rows):** 569
- **Number of Features (Columns):** 30
- **Target Variable:** 1 (Malignant), 0 (Benign)

#### Feature Names:

1. **mean radius**
2. **mean texture**
3. **mean perimeter**
4. **mean area**
5. **mean smoothness**
6. **mean compactness**
7. **mean concavity**
8. **mean concave points**
9. **mean symmetry**
10. **mean fractal dimension**
11. **radius error**
12. **texture error**
13. **perimeter error**
14. **area error**
15. **smoothness error**
16. **compactness error**
17. **concavity error**
18. **concave points error**
19. **symmetry error**
20. **fractal dimension error**
21. **worst radius**
22. **worst texture**
23. **worst perimeter**
24. **worst area**
25. **worst smoothness**
26. **worst compactness**
27. **worst concavity**
28. **worst concave points**
29. **worst symmetry**
30. **worst fractal dimension**

The features are divided into three categories:
- **Mean values**: Represents average measurements of the tumor cells.
- **Error values**: Represents the standard error of the measurements.
- **Worst values**: Represents the most extreme measurements (i.e., the largest value of the feature) among the different samples.

---

## Steps Involved

### 1. **Data Preprocessing**
In the first step, the raw Breast Cancer dataset is loaded using the `load_breast_cancer()` function from `sklearn.datasets`. The dataset contains 30 features related to cell nuclei properties and the target variable, which indicates whether the tumor is malignant or benign.

- The data is split into features (X) and target labels (y).
- Features represent different properties of the cell nucleus, while the target variable represents the tumor classification (0 for benign, 1 for malignant).

### 2. **Feature Selection**
The next step involves selecting the most important features that contribute significantly to the classification task. This is done using the **ANOVA F-test** (`f_classif`), which measures the relationship between the features and the target variable.

- **Top 10 Features:** The top 10 features that have the highest correlation with the target are selected for model training. This reduces the dimensionality and focuses the model on the most important information.
- After performing feature selection using the **ANOVA F-test** (`f_classif`), the following 10 features were selected as the most significant predictors for classifying breast cancer as malignant or benign:

1. **mean radius**
2. **mean perimeter**
3. **mean area**
4. **mean concavity**
5. **mean concave points**
6. **worst radius**
7. **worst perimeter**
8. **worst area**
9. **worst concavity**
10. **worst concave points**

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
