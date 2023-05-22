from itertools import product
import pandas as pd

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

    def search(self):
        self.results = pd.DataFrame(columns=list(self.params.keys()) + ["score"])
        hyperparameter_combinations = list(product(*param_grid.values()))
        for combination in hyperparameter_combinations:
            param_score = score(self.x, self.y)  # Why does this return an object

            # Splitting
            # (Preprocessing)
            # Training
            # Testing
            # -> Score

            self.results.loc[len(self.results.index)] = [*combination, score]
        return self.results


if __name__ == "__main__":
    gs = GridSearcher(None, param_grid, [0], [1])
    print(gs.search())
