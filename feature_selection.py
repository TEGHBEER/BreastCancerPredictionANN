from sklearn.feature_selection import SelectKBest, f_classif

def select_features(X, y):
    selector = SelectKBest(f_classif, k=10)
    X_selected = selector.fit_transform(X, y)
    return X_selected