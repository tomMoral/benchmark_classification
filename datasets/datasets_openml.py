from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import openml
    import pandas as pd


DATASETS = {
    'bank-marketing': 44126,
    'california': 44090,
    'compass': 44162,
    'covertype': 44121,
    'covertype_categorical': 44159,
    'credit': 44089,
    'electricity': 44120,
    'electricity_categorical': 44156,
    'eyemovements': 44130,
    'eyemovements_categorical': 44156,
    'higgs': 44129,
    'house_16H': 44123,
    'jannis': 44131,
    'KDDCup06_upselling': 44186,
    'kdd_ipums_la_97-small': 44124,
    'magictelescope': 44125,
    'MiniBooNE': 44128,
    'phoneme': 44127,
    'pol': 44122,
    'rl': 44160,
    'road_safety': 44161,
    'wine': 44091,
}


class Dataset(BaseDataset):

    name = 'openml'

    install_cmd = 'conda'
    requirements = ["pip:chardet", "pip:openml", "pip:scikit-learn"]

    parameters = {
        "dataset": list(DATASETS),
    }

    def get_data(self):
        dataset = openml.datasets.get_dataset(
            DATASETS[self.dataset], download_data=True,
            download_qualities=True, download_features_meta_data=True
        )
        X, y, cat_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        # convert int and object to categories
        # TODO: handle cases where categorical cardinality is > 255
        # (this is not supported using HGBT native categorical handling)
        for col in X.columns:
            is_integer = pd.api.types.is_integer_dtype(X[col])
            is_object = pd.api.types.is_object_dtype(X[col])
            if is_integer or is_object:
                X[col] = X[col].astype('category')

        return dict(
            X=X,
            y=y,
            categorical_indicator=cat_indicator
        )
