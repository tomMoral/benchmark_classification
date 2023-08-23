from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import optuna
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder as OHE
    from sklearn.model_selection import cross_validate
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import train_test_split
    from .average_clf import AverageClassifier


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class OSolver(BaseSolver):

    stopping_criterion = SufficientProgressCriterion(
        strategy='callback', patience=5
    )

    params = {
        'test_size': 0.25,
        'seed': 42,
    }

    extra_model_params = {}

    def set_objective(
            self, X_train, y_train,
            categorical_indicator
    ):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        X, X_val, y, y_val = train_test_split(
            X_train, y_train, test_size=self.params['test_size'],
            random_state=self.params['seed'], stratify=y_train
        )

        self.X_train, self.y_train = X, y
        self.X_val, self.y_val = X_val, y_val
        self.cat_ind = categorical_indicator
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
        gm = self.get_model()
        self.model = Pipeline(
            steps=[("preprocessor", preprocessor),
                   ("model", gm)]
        )

    def objective(self, trial):
        param = self.sample_parameters(trial)
        params = self.extra_model_params.copy()
        params.update({
            f"model__{p}": v for p, v in param.items()
        })
        model = self.model.set_params(**params)
        cross_score = cross_validate(
            model, self.X_train, self.y_train, return_estimator=True
        )
        trial.set_user_attr('model', cross_score['estimator'])

        return cross_score['test_score'].mean()

    def run(self, callback):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        sampler = optuna.samplers.RandomSampler()
        best_model = DummyClassifier().fit(self.X_train, self.y_train)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        while callback(best_model):
            study.optimize(self.objective, n_trials=10)
            best_model = AverageClassifier(
                study.best_trial.user_attrs['model']
            )
        self.best = study.best_params
        self.clf = self.get_model().set_params(**self.best).fit(self.X_train, self.y_train)

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return self.clf

    def get_next(self, n_iter):

        return n_iter + 1

    def warmup_solver(self):
        pass
