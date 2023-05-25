import numpy as np
import pandas as pd
from FCLayer import FcLayer
from ACLayer import AcLayer
from activation_functions import sigmoid, relu, sigmoid_derivative, tanh, tanh_derivative


class Network:
    def __init__(
        self,
        learning_rate,
        epochs: int = 1000,
        expand_dims: bool = True,
        node_counts: list = [3],
        activation_function = (tanh, tanh_derivative)
    ):
        self.layers = []
        self.learning_rate = learning_rate
        self.output = []
        self.epochs = epochs
        self.expand_dims = expand_dims
        self.last_result = None
        self.node_counts = node_counts
        self.activation_function = activation_function


    def get_params(self, deep: bool = False):
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "expand_dims": self.expand_dims,
            "node_counts": self.node_counts,
            "activation_function": self.activation_function,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

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
        return self._softmax(result)

    def predict(self, data):
        data = self._clean_data(data)
        scores = self.predict_score(data)
        return [
            [1 if value == np.max(elem[0]) else 0 for value in elem[0]] if len(elem[0]) > 1 else round(elem[0])
            for elem in scores
        ]

    def train(self, x_train, y_train):

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
        n_features = len(data[0][0])
        n_classes = targets.shape[-1]
        self._build_network(n_features, n_classes, self.node_counts)
        self.train(data, targets)

    # Utility functions

    def _clean_data(self, data):
        if type(data) == pd.DataFrame or type(data) == pd.Series:
            data = data.to_numpy()

        if self.expand_dims:
            data = np.expand_dims(data, axis=1)

        return data

    def _build_network(
        self,
        n_features: int,
        n_classes: int,
        node_counts: list = [3],

    ):
        input_shape = n_features
        output_shape = n_classes if n_classes > 2 else 1
        prior_output = input_shape
        for n_nodes in node_counts:
            i = prior_output
            o = n_nodes
            prior_output = o
            fc = FcLayer(i, o)
            ac = AcLayer(self.activation_function[0], self.activation_function[1])
            self.add(fc)
            self.add(ac)

        # Add output layer
        self.add(FcLayer(prior_output, output_shape))
        self.add(AcLayer(self.activation_function[0], self.activation_function[1]))

    def _softmax(self, values: np.array):
        return values / np.sum(values)