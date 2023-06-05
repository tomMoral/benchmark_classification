from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import openml


DATASETS = {
    'bank_marketing': 44126,
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
    requirements = ["pip:openml", "pip:chardet"]

    parameters = {
        "test_size": [0.25],
        "seed": [42],
        "dataset": list(DATASETS),
    }

    def get_data(self):
        dataset = openml.datasets.get_dataset(
            self.dataset
        )
        X, y, cat_indicator, attribute_names = dataset.get_data(
            dataset_format="array", target = dataset.default_target_attribute
        )

        return dict(
            X=X,
            y=y,
            categorical_indicator=cat_indicator
        )