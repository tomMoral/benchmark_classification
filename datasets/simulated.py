from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.model_selection import train_test_split

    from benchopt.datasets import make_correlated_data


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'n_samples, n_features': [
            (1000, 500),
            (5000, 200),
        ],
        'test_size': [0.25],
        'seed': [27],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Generate pseudorandom data using `numpy`.
        rng = np.random.RandomState(self.seed)
        X, y, _ = make_correlated_data(self.n_samples, self.n_features)
        y = y > 0

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=rng
        )
        cat_indicator = [False]*X.shape[1]

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            categorical_ind=cat_indicator
        )
