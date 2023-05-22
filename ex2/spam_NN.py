import sys
import os
import numpy as np

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

from FCLayer import FcLayer
from ACLayer import AcLayer
from Network import Network
from activation_functions import sigmoid, relu, sigmoid_prime, tanh, tanh_prime,relu_prime

learning_rate = 0.1
data=pd.read_csv('spam_final_df.data')
print(data.head(10))
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=26)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_test=np.expand_dims(X_test,axis=1)
X_train=np.expand_dims(X_train,axis=1)
y_train=np.expand_dims(y_train,axis=1)
y_test=np.expand_dims(y_test,axis=1)


network1 = Network(learning_rate)
# print(X_train.shape)
print(y_train.shape)

fc1 = FcLayer(57,50)
ac1 = AcLayer(relu, relu_prime)
fc2 = FcLayer(50, 40)
ac2 = AcLayer(relu, relu_prime)
fc3 = FcLayer(40, 10)
ac3 = AcLayer(relu, relu_prime)
fc4 = FcLayer(30, 20)
ac4 = AcLayer(relu, relu_prime)
fc5 = FcLayer(20, 10)
ac5 = AcLayer(sigmoid, sigmoid_prime)
# fc6 = FcLayer(10, 5)
# ac6 = AcLayer(sigmoid, sigmoid_prime)
# fc7 = FcLayer(5, 4)
# ac7 = AcLayer(sigmoid, sigmoid_prime)

# fc8 = FcLayer(4, 3)
# ac8 = AcLayer(sigmoid, sigmoid_prime)
# fc9 = FcLayer(3, 2)
# ac9 = AcLayer(sigmoid, sigmoid_prime)
fc10 = FcLayer(10, 1)
ac10= AcLayer(relu, relu_prime)

network1.add(fc1)
network1.add(ac1)
network1.add(fc2)
network1.add(ac2)

network1.add(fc3)
network1.add(ac3)
# network1.add(fc4)
# network1.add(ac4)
# network1.add(fc5)
# network1.add(ac5)

# network1.add(fc6)
# network1.add(ac6)
# network1.add(fc7)
# network1.add(ac7)
# network1.add(fc8)
# network1.add(ac8)

# network1.add(fc9)
# network1.add(ac9)
network1.add(fc10)
network1.add(ac10)

def score(y_true, y_pred):
    correct_predictions = 0
    total_predictions = len(y_true)

    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label == predicted_label:
            correct_predictions += 1



    accuracy = correct_predictions / total_predictions
    return accuracy


epochs=1000
network1.train(X_train, y_train, epochs=epochs)

y_pred = network1.predict(X_test)
print("y_pred unique")
print(np.unique(y_pred))
accuracy=score(y_test,y_pred)
print("accuracy")
print(accuracy)