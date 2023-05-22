import sys
import os
import numpy as np

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))

from FCLayer import FcLayer
from ACLayer import AcLayer
from Network import Network
from activation_functions import sigmoid, relu, sigmoid_prime, tanh, tanh_prime

learning_rate = 0.1
# x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
# # x_train = np.array([[[10000, 1111]], [[111110, 5000]], [[100, 220]]])
# # y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
# y_train = np.array([[[0]], [[1]], [[1]], [[0]]])


x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])


network1 = Network(learning_rate)

# fc1 = FcLayer(2, 3)
# ac1 = AcLayer(tanh, tanh_prime)
# fc2 = FcLayer(3, 1)
# ac2 = AcLayer(tanh, tanh_prime)
# network1.add(fc1)
# network1.add(ac1)
# network1.add(fc2)
# network1.add(ac2)

input_values = 2
output_values = 1
intermediate_values = 3
layers = 3
epochs = 1000

for j in range(layers):
    i = intermediate_values
    o = intermediate_values

    if not j:
        i = input_values
    elif j + 1 == layers:
        o = output_values

    fc = FcLayer(i, o)
    ac = AcLayer(tanh, tanh_prime)
    network1.add(fc)
    network1.add(ac)

network1.train(x_train, y_train, epochs=epochs)

out = network1.predict(x_train)
# print(out)


# fc1.forward_propagation(input)
# fc2.forward_propagation(fc1.output)
# print(fc2.output)

# x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
# y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# # network
# net = Network()
# net.add(FCLayer(2, 3))
# net.add(ActivationLayer(tanh, tanh_prime))
# net.add(FCLayer(3, 1))
# net.add(ActivationLayer(tanh, tanh_prime))


# train
# net.use(mse, mse_prime)
# net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# # test
# out = net.predict(x_train)
# print(out)
