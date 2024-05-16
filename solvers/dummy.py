from benchopt import safe_import_context
from benchmark_utils.optuna_solver import OSolver

with safe_import_context() as import_ctx:
    import optuna  # noqa: F401
    from sklearn.dummy import DummyClassifier


class Solver(OSolver):

    name = 'dummy'
    requirements = ["pip:optuna"]

    def get_model(self):
        return DummyClassifier()

    def sample_parameters(self, trial):
        strategy = trial.suggest_categorical("strategy", ["most_frequent", "prior", "stratified", "uniform"])
        return dict(
            strategy=strategy
        )
