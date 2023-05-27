from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.model_selection import train_test_split
    import openml


class Dataset(BaseDataset):

    name = "phoneme"

    parameters = {
        'test_size': [0.25],
        'seed': [42]
    }

    def get_data(self):
        rng = np.random.RandomState(self.seed)
        dataset = openml.datasets.get_dataset(44127)
        X, y, cat_indicator, attribute_names = dataset.get_data(
            dataset_format="array", target=dataset.default_target_attribute
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=rng
        )

        return dict(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            categorical_indicator=cat_indicator
        )
