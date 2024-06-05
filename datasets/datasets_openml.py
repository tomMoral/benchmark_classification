from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import openml
    import pandas as pd


DATASETS = {
        # 44126,  # bank-marketing
        # 44090,  # california
        # 44162,  # compass
        # 44121,  # covertype
        # 44159,  # covertype_categorical
        # 44089,  # credit
        # 44120,  # electricity
        # 44156,  # electricity_categorical
        # 44130,  # eyemovements
        # 44156,  # eyemovements_categorical
        # 44129,  # higgs
        # 44123,  # house_16H
        # 44131,  # jannis
        # 44186,  # KDDCup06_upselling
        # 44124,  # kdd_ipums_la_97-small
        # 44125,  # magictelescope
        # 44128,  # MiniBooNE
        # 44127,  # phoneme
        # 44122,  # pol
        # 44160,  # rl
        # 44161,  # road_safety
        # 44091,  # wine
        # 42803,
        151,      # electricity
        44121,    # covertype
        44122,    # pol
        44123,    # house_16H
        44124,    # kdd_ipums_la_97-small
        44125,    # magictelescope
        44126,    # bank-marketing
        44127,    # phoneme
        44128,    # MiniBooNE
        44129,    # higgs
        44130,    # eyemovements
        44131,    # jannis
        44089,    # credit
        44090,    # california
        44091,    # wine   
}


class Dataset(BaseDataset):

    name = 'openml'

    install_cmd = 'conda'
    requirements = ["pip:openml", "pip:chardet"]

    parameters = {
        "dataset": list(DATASETS),
    }

    def get_data(self):
        dataset = openml.datasets.get_dataset(
            self.dataset
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
