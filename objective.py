from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score as BAS
    from sklearn.metrics import roc_auc_score as RAS


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "Classification"
    url = "https://github.com/tommoral/benchmark_classification"

    is_convex = False

    requirements = ["pip:scikit-learn"]

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.
    parameters = {
        'seed': list(range(100)),
        'test_size': [0.25],
    }

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"

    def set_data(
            self, X, y,
            categorical_indicator
    ):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        rng = np.random.RandomState(self.seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=rng,
            stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25,
            random_state=rng, stratify=y_train
        )

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        self.categorical_indicator = categorical_indicator

    def evaluate_result(self, model):
        # The arguments of this function are the outputs of the
        # `Solver.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.
        score_train = model.score(self.X_train, self.y_train)
        score_test = model.score(self.X_test, self.y_test)
        score_val = model.score(self.X_val, self.y_val)
        bl_acc = BAS(self.y_test, model.predict(self.X_test))
        pred = model.predict_proba(self.X_test)
        if len(np.unique(self.y_test)) > 2:
            roc_score = RAS(self.y_test, pred, multi_class='ovr')
        else:
            roc_score = RAS(self.y_test, pred[:, 1])

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            score_test=score_test,
            score_train=score_train,
            score_val=score_val,
            balanced_accuracy=bl_acc,
            roc_auc_score=roc_score,
            value=1-score_test
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.compute`. This is mainly for testing purposes.
        return dict(model=DummyClassifier().fit(self.X_train, self.y_train))

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        return dict(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            categorical_indicator=self.categorical_indicator
        )
