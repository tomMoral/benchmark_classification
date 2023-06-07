from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder as OHE


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class GSSolver(BaseSolver):

    stopping_criterion = SufficientProgressCriterion(strategy='callback')

    def set_objective(
            self, X_train, y_train,
            categorical_indicator
    ):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.X, self.y = X_train, y_train
        self.cat_ind = categorical_indicator
        size = self.X.shape[1]
        preprocessor = ColumnTransformer(
            [("one_hot", OHE(categories="auto",
                             handle_unknown="ignore"),
             [i for i in range(size) if self.cat_ind[i]]),
             (
                "numerical",
                "passthrough",
                [i for i in range(size) if not self.cat_ind[i]],
            )]
        )

        gm = self.get_model()
        model = Pipeline(steps=[("preprocessor", preprocessor), ("model", gm)])
        self.clf = GridSearchCV(
            model, self.parameter_grid
        )

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.

        self.clf.fit(self.X, self.y)

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return self.clf.best_estimator_

    def warmup_solver(self):
        pass
