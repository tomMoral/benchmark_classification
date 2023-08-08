from benchopt import safe_import_context
from benchmark_utils.optuna_solver import OSolver

with safe_import_context() as import_ctx:
    import optuna  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingClassifier


class Solver(OSolver):

    name = 'HistGradientBoostingClassifier'
    requirements = ["pip:optuna"]

    extra_model_params = {
        "preprocessor__one_hot__sparse_output": False
    }

    def get_model(self):
        return HistGradientBoostingClassifier()

    def sample_parameters(self, trial):
        max_iter = trial.suggest_int("max_iter", 100, 2000, step=10)
        l_rate = trial.suggest_float("learning_rate", 1e-4, 1, log=True)
        l2 = trial.suggest_float("l2_regularization", 1e-8, 1e-1, log=True)
        return dict(
            max_iter=max_iter,
            learning_rate=l_rate,
            l2_regularization=l2
        )
