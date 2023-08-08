from benchopt import safe_import_context
from benchmark_utils.optuna_solver import OSolver

with safe_import_context() as import_ctx:
    import optuna  # noqa: F401
    from xgboost import XGBClassifier


class Solver(OSolver):

    name = 'XGBoost'

    requirements = ['py-xgboost', 'pip:optuna']

    def get_model(self):
        return XGBClassifier()

    def sample_parameters(self, trial):
        n_estimators = trial.suggest_int("n_estimators", 10, 2000, step=10)
        max_depth = trial.suggest_int("max_depth", 1, 11, step=1)
        l_rate = trial.suggest_float("learning_rate", 1e-4, 1, log=True)
        l2 = trial.suggest_float("reg_lambda", 1e-8, 1e-1, log=True)
        return dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=l_rate,
            reg_lambda=l2
        )
