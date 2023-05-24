from benchopt import safe_import_context
from benchmark_utils.gridsearch_solver import GSSolver

with safe_import_context() as import_ctx:
    from sklearn.ensemble import RandomForestClassifier

class Solver(GSSolver):
    name = 'RandomForest'

    parameter_grid = {'n_estimators' : [10,20,50,100,200]}
    
    def get_model(self):
        return RandomForestClassifier()