from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris


class Dataset(BaseDataset):

    name = "iris"

    parameters = {
        'test_size': [0.25],
        'seed': [27]
    }

    def get_data(self):
        X, y = load_iris(
            return_X_y=True
        )
        cat_indicator = [False]*X.shape[1]

        return dict(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            categorical_indicator=cat_indicator
        )
