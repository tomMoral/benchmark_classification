from benchopt import safe_import_context
from benchmark_utils.optuna_solver import OSolver

with safe_import_context() as import_ctx:
    import optuna  # noqa: F401
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import HistGradientBoostingClassifier


class Solver(OSolver):

    name = 'HistGradientBoostingClassifier'
    requirements = ["pip:optuna"]
    extra_model_params = {
    }

    def get_model(self):
        preprocessor = "passthrough"
        return Pipeline(steps= [("preprocessor", preprocessor),
                                  ("model", HistGradientBoostingClassifier(categorical_features='from_dtype'))])

    def sample_parameters(self, trial):
        max_iter = trial.suggest_int("max_iter", 10, 2000, step=10)
        l_rate = trial.suggest_float("learning_rate", 1e-2, 1, log=True)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 3, 300, log=True)
        min_samples_leaf = trial.suggest_int(
            "min_samples_leaf", 1, 300, log=True
        )
        return dict(
            max_iter=max_iter,
            learning_rate=l_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf
        )
