from benchopt import safe_import_context
from benchmark_utils.gridsearch_solver import GSSolver

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.svm import SVC
    from optuna.distributions import FloatDistribution, CategoricalDistribution


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(GSSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'SVM'

    parameter_grid = {
        'model__C':FloatDistribution(1e-10, 1, log=True),
        'model__kernel' : CategoricalDistribution(choices=("linear", "poly", "rbf"))
    }

    def get_model(self):
        return SVC(probability=True)
