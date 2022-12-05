from benchopt import safe_import_context
from benchmark_utils.gridsearch_solver import GSSolver

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.svm import SVC


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(GSSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'SVM'

    parameter_grid = {
        'C': [10, 1, .1]
    }

    def get_model(self):
        return SVC()
