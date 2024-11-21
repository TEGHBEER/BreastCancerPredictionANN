from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

def tune_model(X, y):
    mlp = MLPClassifier(max_iter=1000)
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)], 
        'activation': ['relu', 'tanh'], 
        'solver': ['adam', 'sgd'], 
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }
    grid_search = GridSearchCV(mlp, param_grid, cv=5)
    grid_search.fit(X, y)
    return grid_search.best_estimator_
