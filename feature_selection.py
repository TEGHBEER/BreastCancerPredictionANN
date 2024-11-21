from sklearn.feature_selection import SelectKBest, f_classif

def select_features(X, y):
    # Select the top 10 features based on ANOVA F-test
    selector = SelectKBest(f_classif, k=10)
    X_selected = selector.fit_transform(X, y)
    
    # Return both the selected features and the selector object
    return X_selected, selector
