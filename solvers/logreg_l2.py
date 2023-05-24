from benchopt import safe_import_context
from benchmark_utils.gridsearch_solver import GSSolver

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.linear_model import LogisticRegression


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(GSSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'logreg_l2'

    parameters = {
        'penalty': ['l1', 'l2', 'elasticnet'],
    }

    parameter_grid = {
        'C': [10, 1, .1]
    }

    def get_model(self):
        solver = 'lbfgs'
        l1_ratio = None
        if self.penalty == 'l1':
            solver = 'liblinear'
        elif self.penalty == 'elasticnet':
            solver = 'saga'
            l1_ratio = 0.5
        return LogisticRegression(
            penalty=self.penalty, solver=solver, l1_ratio=l1_ratio
        )
