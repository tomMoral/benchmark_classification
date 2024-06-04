from benchopt import safe_import_context
from benchmark_utils.optuna_solver import OSolver

with safe_import_context() as import_ctx:
    import optuna  # noqa: F401
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder as OHE
    from sklearn.ensemble import ExtraTreesClassifier


class Solver(OSolver):

    name = 'ExtraTrees'
    requirements = ["pip:optuna"]

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
        return Pipeline(steps= [("preprocessor", preprocessor),
                                  ("model", ExtraTreesClassifier())])

    def sample_parameters(self, trial):
        n_estimators = trial.suggest_int("n_estimators", 10, 200, step=10)
        return dict(
            n_estimators=n_estimators
        )
