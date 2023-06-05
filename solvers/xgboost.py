from benchopt import safe_import_context
from benchmark_utils.optuna_solver import OSolver

with safe_import_context() as import_ctx:
    from xgboost import XGBClassifier
    import optuna


class Solver(OSolver):

    name = 'XGBoost'

    requirements = ['py-xgboost', 'pip:optuna']

    def get_model(self):
        return XGBClassifier()

    def sample_parameters(self, trial):
        n_estimators = trial.suggest_int("n_estimators", 10, 2000, step=10)
        max_depth = trial.suggest_int("max_depth", 1, 11, step=1)
        return dict(
            model__n_estimators=n_estimators,
            model__max_depth=max_depth,
        )
