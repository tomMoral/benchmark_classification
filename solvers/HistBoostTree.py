from benchopt import safe_import_context
from benchmark_utils.gridsearch_solver import GSSolver

with safe_import_context() as import_ctx:
    from sklearn.ensemble import HistGradientBoostingClassifier

class Solver(GSSolver):
    name = 'HistGradientBoostingClassifier'

    parameter_grid = {
        'max_iter' : [100, 1000, 2000],
        'validation_fraction' : [0.1, 0.2, 0.3, 0.5],
        'n_iter_no_change' : [10, 20, 40, 50]
    }

    def get_model(self):
        return HistGradientBoostingClassifier()