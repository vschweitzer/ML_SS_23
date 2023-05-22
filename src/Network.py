import numpy as np
from FCLayer import FcLayer
from ACLayer import AcLayer
from activation_functions import sigmoid, relu, sigmoid_prime


class Network:
    def __init__(self, learning_rate):
        self.layers = []
        self.learning_rate = learning_rate
        self.output = []

    # rework when working with the actual datasets
    def preprocess(self, data):
        data[np.isnan(data)] = np.nanmean(data)

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, test_data):
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
        print("Predict function works")
        print(result)
        return result

    def train(self, x_train, y_train, epochs):
        # Epoch times needed to achieve accurate NN
        for i in range(epochs):
            error = 0
            # result = []
            for j in range(len(x_train)):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                # result.append(output)
                # err = np.mean(np.power(y_train[j] - output, 2))
                error = 2 * (output - y_train[j]) / y_train[j].size

                # print("Train function works")
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, self.learning_rate)
                    # print(f"error:{error}")
