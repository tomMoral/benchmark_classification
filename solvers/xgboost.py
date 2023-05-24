from benchopt import safe_import_context
from benchmark_utils.gridsearch_solver import GSSolver

with safe_import_context() as import_ctx:
    from xgboost import XGBClassifier

class Solver(GSSolver):

    name='XGBoost'

    parameter_grid = {
        'n_estimators' : [1000, 2000, 5000],
        'max_depth' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    }

    def get_model(self):
        return XGBClassifier()