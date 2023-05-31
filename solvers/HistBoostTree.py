from benchopt import safe_import_context
from benchmark_utils.gridsearch_solver import GSSolver

with safe_import_context() as import_ctx:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from optuna.distributions import IntDistribution, FloatDistribution


class Solver(GSSolver):

    name = 'HistGradientBoostingClassifier'

    requirements = ['pip:optuna']

    parameter_grid = {
        'model__max_iter': IntDistribution(100, 2000, step=10),
        'model__learning_rate': FloatDistribution(1e-1, 1, log=True)
    }

    def get_model(self):
        return HistGradientBoostingClassifier()
