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
        n_estimators = trial.suggest_int("max_iter", 10, 2000, step=10)
        l_rate = trial.suggest_float("learning_rate", 1e-2, 1, log=True)
        max_leaves = trial.suggest_int("max_leaf_nodes", 3, 300, log=True)
        min_child_weight = trial.suggest_int(
            "min_samples_leaf", 1, 300, log=True
        )
        return dict(
            n_estimators=n_estimators,
            learning_rate=l_rate,
            max_leaves=max_leaves,
            min_child_weight=min_child_weight
        )
