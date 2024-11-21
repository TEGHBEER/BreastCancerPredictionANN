from sklearn.metrics import accuracy_score, classification_report

def train_ann(X, y, model):
    model.fit(X, y)
    y_pred = model.predict(X)
    print("Accuracy:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))
