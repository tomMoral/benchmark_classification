from benchopt import safe_import_context
from benchmark_utils.gridsearch_solver import GSSolver

with safe_import_context() as import_ctx:
    from xgboost import XGBClassifier
    from optuna.distributions import IntDistribution


class Solver(GSSolver):

    name = 'XGBoost'

    requirements = ['py-xgboost', 'pip:optuna']

    parameter_grid = {
        'model__n_estimators': IntDistribution(100, 2000, step=10),
        'model__max_depth': IntDistribution(1, 11, step=1)
    }

    def get_model(self):
        return XGBClassifier()
