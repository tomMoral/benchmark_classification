from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
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
            X=X,
            y=y,
            categorical_indicator=cat_indicator
        )
