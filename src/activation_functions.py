import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def relu_derivative(x):
    return np.array(x >= 0).astype("int")

def relu(x):
    return np.maximum(x, 0)