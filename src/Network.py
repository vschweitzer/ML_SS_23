import numpy as np
import pandas as pd
from FCLayer import FcLayer
from ACLayer import AcLayer
from activation_functions import sigmoid, relu, sigmoid_prime


class Network:
    def __init__(self, learning_rate, epochs: int = 1000, expand_dims: bool = True):
        self.layers = []
        self.learning_rate = learning_rate
        self.output = []
        self.epochs = epochs
        self.expand_dims = expand_dims
        self.min_category: int
        self.max_category: int
        self.last_result = None

    # rework when working with the actual datasets
    def preprocess(self, data):
        data[np.isnan(data)] = np.nanmean(data)

    def add(self, layer):
        self.layers.append(layer)

    def predict_score(self, test_data):
        # sample dimension first
        samples = len(test_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = test_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        # print("Predict function works")
        self.last_result = result
        # print(result)
        return result

    def predict(self, data):
        data = self._clean_data(data)
        scores = self.predict_score(data)
        return [
            round(
                (elem[0][0] - self.min_category)
                * (self.max_category - self.min_category)
            )
            for elem in scores
        ]

    def train(self, x_train, y_train):
        self.min_category = min(y_train, key=lambda x: x[0])[0]
        self.max_category = max(y_train, key=lambda x: x[0])[0]

        # Epoch times needed to achieve accurate NN
        for i in range(self.epochs):
            error = 0
            # result = []
            for j in range(len(x_train)):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                # result.append(output)
                # err = np.mean(np.power(y_train[j] - output, 2))
                error = (2 / y_train[j].size) * (output - y_train[j])

                # print("Train function works")
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, self.learning_rate)
                    # print(f"error:{error}")

    ##########################################
    # Methods for compatibility with sklearn #
    ##########################################

    def fit(self, data, targets):
        data = self._clean_data(data)
        targets = self._clean_data(targets)
        self.train(data, targets)

    # Utility functions

    def _clean_data(self, data):
        if type(data) == pd.DataFrame or type(data) == pd.Series:
            data = data.to_numpy()

        if self.expand_dims:
            data = np.expand_dims(data, axis=1)

        return data
