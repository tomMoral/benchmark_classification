from benchopt import safe_import_context
from benchmark_utils.gridsearch_solver import GSSolver

with safe_import_context() as import_ctx:
    from sklearn.ensemble import HistGradientBoostingClassifier


class Solver(GSSolver):

    name = 'HistGradientBoostingClassifier'

    parameter_grid = {
        'model__max_iter': [100, 1000, 2000],
        'model__learning_rate': [0.1, 0.2, 0.5, 1]
    }

    def get_model(self):
        return HistGradientBoostingClassifier()
