import numpy as np


# class AcLayer:
#     def __init__(self, activation):
#         self.activation = activation
#         # self.activation_prime = activation_prime

#     def forward_propagation(self, input):
#         self.input = input
#         self.output = self.activation(input)
#         return self.output

#     def backward_propagation(self, output_error):
#         return self.activation_prime(self.input) * output_error


class AcLayer:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
