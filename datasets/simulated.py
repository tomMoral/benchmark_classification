from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchopt.datasets import make_correlated_data
    import numpy as np


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
        'seed': [42]
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Generate pseudorandom data using `numpy`.
        rng = np.random.RandomState(self.seed)
        X, y, _ = make_correlated_data(
            self.n_samples, self.n_features, random_state=rng
        )
        y = y > 0

        cat_indicator = [False]*X.shape[1]

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            X=X,
            y=y,
            categorical_indicator=cat_indicator
        )
