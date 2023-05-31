from benchopt import safe_import_context
from benchmark_utils.gridsearch_solver import GSSolver

with safe_import_context() as import_ctx:
    from sklearn.ensemble import RandomForestClassifier
    from optuna.distributions import IntDistribution


class Solver(GSSolver):

    name = 'RandomForest'

    parameter_grid = {'model__n_estimators': IntDistribution(10, 200, step=10)}

    def get_model(self):
        return RandomForestClassifier()
