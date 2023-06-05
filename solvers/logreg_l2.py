from benchopt import safe_import_context
from benchmark_utils.optuna_solver import OSolver

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.linear_model import LogisticRegression
    import optuna


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(OSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'logreg_l2'

    parameters = {
        'penalty': ['l1', 'l2', 'elasticnet'],
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

    def sample_parameters(self, trial):
        c = trial.suggest_float("C", 1e-1, 1e1, log=True)
        return dict(
            model__C=c
        )
