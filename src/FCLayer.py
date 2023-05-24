import numpy as np


class FcLayer:
    
    
    np.random.seed(1)

    # Initialize weights necessary
    def __init__(self, input_shape, output_shape):
        self.weights = np.random.rand(input_shape, output_shape)
        self.bias = np.random.rand(1, output_shape)
        self.learning_rate = 0.1

    def forward_propagation(self, input):
        self.output = self.bias  + np.dot(input, self.weights)
        self.input = input
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        in_error = np.dot(output_error, self.weights.T)
        w_error = np.dot(self.input.T, output_error)
    
        #Adjust bias and weights
        self.weights = self.weights - learning_rate * w_error
        self.bias = self.bias - learning_rate * output_error
        return in_error

    def print_data(self):
        return print(
            f"The weights are  {self.weights} and the biases are {self.biases}"
        )
