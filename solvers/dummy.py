from benchopt import safe_import_context
from benchmark_utils.optuna_solver import OSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import optuna  # noqa: F401
    from sklearn.dummy import DummyClassifier


class Solver(OSolver):

    name = 'dummy'
    requirements = ["pip:optuna"]

    stopping_criterion = SufficientProgressCriterion(
        strategy='callback', patience=200
    )

    def get_model(self):
        return DummyClassifier(strategy='uniform')

    def sample_parameters(self, trial): 
        seed = trial.suggest_int("seed", 0, 2**31)
        return dict(
            random_state=seed
        )
