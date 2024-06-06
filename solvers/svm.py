from benchopt import safe_import_context
from benchmark_utils.optuna_solver import OSolver

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import optuna  # noqa: F401
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder as OHE
    from sklearn.svm import SVC


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(OSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'SVM'

    requirements = ["pip:optuna", "pip:scikit-learn"]

    def get_model(self):
        size = self.X_train.shape[1]
        preprocessor = ColumnTransformer(
            [
                ("one_hot", OHE(
                        categories="auto", handle_unknown="ignore",
                    ), [i for i in range(size) if self.cat_ind[i]]),
                ("numerical", "passthrough",
                 [i for i in range(size) if not self.cat_ind[i]],)
            ]
        )
        return Pipeline(steps=[("preprocessor", preprocessor),
                               ("model", SVC(probability=True))])

    def skip(self, X_train, **kwargs):
        if X_train.shape[0] > 5000:
            return True, "Too large for SVC"
        return False, None

    def sample_parameters(self, trial):
        params = {}
        params['C'] = trial.suggest_float("C", 1e-1, 1e1, log=True)
        params['kernel'] = trial.suggest_categorical(
            "kernel", ["linear", "rbf", "poly"]
        )
        if params['kernel'] == 'rbf' or params['kernel'] == 'poly':
            params['gamma'] = trial.suggest_float("gamma", 1e-4, 1, log=True)
        return params
