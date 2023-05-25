from benchopt import safe_import_context
from benchmark_utils.gridsearch_solver import GSSolver

with safe_import_context() as import_ctx:
    from xgboost import XGBClassifier


class Solver(GSSolver):

    name = 'XGBoost'

    requirements = ['py-xgboost']

    parameter_grid = {
        'n_estimators': [100, 1000, 2000],
        'max_depth': range(1, 12)
    }

    def get_model(self):
        return XGBClassifier()
