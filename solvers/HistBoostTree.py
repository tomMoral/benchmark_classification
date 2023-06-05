from benchopt import safe_import_context
from benchmark_utils.optuna_solver import OSolver

with safe_import_context() as import_ctx:
    from sklearn.ensemble import HistGradientBoostingClassifier
    import optuna


class Solver(OSolver):

    name = 'HistGradientBoostingClassifier'
    requirements = ["pip:optuna"]

    def get_model(self):
        return HistGradientBoostingClassifier()

    def sample_parameters(self, trial):
        max_iter = trial.suggest_int("max_iter", 100, 2000, step=10)
        l_rate = trial.suggest_int("learning_rate", 1e-1, 1, log=True)
        return dict(
            model__max_iter=max_iter,
            model__learning_rate=l_rate
        )
