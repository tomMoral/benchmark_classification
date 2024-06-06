from benchopt import safe_import_context
from benchmark_utils.optuna_solver import OSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import optuna  # noqa: F401
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder as OHE
    from sklearn.dummy import DummyClassifier


class Solver(OSolver):

    name = 'dummy'
    requirements = ["pip:optuna", "pip:scikit-learn"]

    stopping_criterion = SufficientProgressCriterion(
        strategy='callback', patience=200
    )

    def get_model(self):
        size = self.X_train.shape[1]
        preprocessor = ColumnTransformer(
            [
                ("one_hot", OHE(
                        categories="auto", handle_unknown="ignore",
                    ), [i for i in range(size) if self.cat_ind[i]]),
                ("numerical", "passthrough",
                 [i for i in range(size) if not self.cat_ind[i]],)
            ]
        )
        return Pipeline(steps=[("preprocessor", preprocessor),
                               ("model", DummyClassifier())])

    def sample_parameters(self, trial):
        seed = trial.suggest_int("seed", 0, 2**31)
        return dict(
            random_state=seed
        )
