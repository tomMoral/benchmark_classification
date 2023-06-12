from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import stats
    from sklearn.metrics import accuracy_score

    
class AverageClassifier:
    """
    Return the average predictions

    Parameters
    ----------
    estimators: list of estimators already fitted
    """
    def __init__(self, estimators):
        self.estimators = estimators

    def predict(self, X_test):
        y_pred = []
        for e in self.estimators:
            y_pred.append(e.predict(X_test))
        y_pred = np.array(y_pred)
        y_pred = stats.mode(y_pred, keepdims=True)[0][0]

        return y_pred

    def predict_proba(self, X_test):
        y_pred = []
        for e in self.estimators:
            y_pred.append(e.predict_proba(X_test))
        y_pred = np.array(y_pred)
        y_pred = np.mean(y_pred, axis=0)

        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)