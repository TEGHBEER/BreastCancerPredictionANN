from sklearn.datasets import load_breast_cancer
import pandas as pd

def load_and_preprocess_data():
    # Load the dataset
    data = load_breast_cancer()
    
    # Convert to DataFrame for features and Series for target
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    
    return X, y
