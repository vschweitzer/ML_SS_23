import numpy as np


class FcLayer:
    np.random.seed(1)

    # Initialize weights necessary
    def __init__(self, input_shape, output_shape):
        self.weights = np.random.rand(input_shape, output_shape)
        self.bias = np.random.rand(1, output_shape)
        self.learning_rate = 0.1

    def forward_propagation(self, input):
        self.input = input
        self.output = self.bias + np.dot(self.input, self.weights)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        # weights_error = np.dot(np.transpose(self.input), output_error)
        # input_error = np.dot(output_error, np.transpose(self.weights))

        # self.weights -= weights_error * learning_rate
        # self.bias -= input_error * learning_rate
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

    def print_data(self):
        return print(
            f"The weights are  {self.weights} and the biases are {self.biases}"
        )
