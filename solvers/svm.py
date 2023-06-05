from benchopt import safe_import_context
from benchmark_utils.optuna_solver import OSolver

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.svm import SVC
    import optuna


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(OSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'SVM'

    requirements = ["pip:optuna"]

    parameter_grid = {
        'model__C': [10, 1, .1]
    }

    def get_model(self):
        return SVC(probability=True)

    def sample_parameters(self, trial):
        c = trial.suggest_float("C", 1e-1, 1e1, log=True)
        return dict(
            model__C=c
        )
