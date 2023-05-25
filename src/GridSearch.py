from itertools import product
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate

# def grid_search(X, y, param_grid, model_fn):
#     best_params = None
#     best_score = 0.0

#     # Generate all possible combinations of hyperparameters
#     hyperparameter_combinations = list(product(*param_grid.values()))

#     # Perform grid search
#     for hyperparameters in hyperparameter_combinations:
#         model = model_fn(*hyperparameters)
#         model.fit(X, y)

#         # Calculate the score on validation set
#         score = model.score(X_val, y_val)

#         # Check if this configuration is the best so far
#         if score > best_score:
#             best_score = score
#             best_params = hyperparameters

#     return best_params, best_score


param_grid = {
    "hidden_layer_sizes": [(10,), (50,), (100,)],
    "activation": ["relu", "tanh"],
    "learning_rate": ["constant", "adaptive"],
}


def score(y_true, y_pred):
    correct_predictions = 0
    total_predictions = len(y_true)

    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label == predicted_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy


class GridSearcher:
    def __init__(self, model, params: dict, x, y):
        self.params = params
        self.model = model
        self.x = x
        self.y = y
        self.results = None

    def search(
        self, folds: int = 5, n_jobs: int = -1, verbose: int = 0, scoring="f1_macro"
    ):
        self.results = pd.DataFrame(
            columns= ["params", "test_score", "fit_time", "score_time", "score_mean"]
        )
        hyperparameter_combinations = list(product(*self.params.values()))
        total_combinations: int = len(hyperparameter_combinations)
        for index, combination in enumerate(hyperparameter_combinations):
            params = dict(zip(self.params.keys(), combination))
            self.model.set_params(**params)
            param_score = cross_validate(
                self.model,
                self.x,
                self.y,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                cv=folds,
            )

            self.results.loc[len(self.results.index)] = [
                params,
                param_score["test_score"],
                param_score["fit_time"],
                param_score["score_time"],
                np.mean(param_score["test_score"]),
            ]
            if verbose >= 0:
                print(f"{index + 1} / {total_combinations}")
        return self.results


if __name__ == "__main__":
    gs = GridSearcher(None, param_grid, [0], [1])
    print(gs.search())
