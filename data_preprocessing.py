from sklearn.datasets import load_breast_cancer
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    scaler = MinMaxScaler()  # MinMaxScaler scales data to a range of 0 to 1
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler
