import numpy as np

import matplotlib.pyplot as plt

X = list()
y = list()

with open("../datasets/bezdekIris.data") as data:
    for line in data.readlines():
        fields = line.split(',')
        x_row = [float(i) for i in fields[:4]]
        flower = fields[4].replace('\n', '')
        if 'setosa' in flower:
            y_row = 1
        elif 'versicolor' in flower:
            y_row = 2
        elif 'virginica' in flower:
            y_row = 3

        X.append(x_row)
        y.append([y_row])

X = np.array(X)
y = np.array(y)

# Normalizing data
X = X/np.amax(X)
y = y/np.amax(y)


class Neural_Network(object):
    def __init__(self):
        #parameters
        self.inputSize = 4
        self.outputSize = 1
        self.hiddenSize = 6

        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) #  weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) #  weight matrix from hidden to output layer

    def forward(self, X):
        #forward propagation through our network
        self.z = np.dot(X, self.W1) # dot product of X (input) and first set of weights
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of weights
        o = self.sigmoid(self.z3) # final activation function
        return o

    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        # backward propgate through the network
        self.o_error = y - o  # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o)  # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T)  # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)  # applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta)  # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta)  # adjusting second set (hidden --> output) weights

    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)


NN = Neural_Network()
mse = list()

# Train Neural Network 325 times
for i in range(325):
    print("Input: \n" + str(X))
    print("Actual Output: \n" + str(y))
    print("Predicted Output: \n" + str(NN.forward(X)))
    loss = np.mean(np.square(y - NN.forward(X)))
    print("Loss: \n" + str(loss))
    print("\n")
    NN.train(X, y)
    mse.append(loss)

mse = np.array(mse)
iterations = np.array(list(range(1, 326)))
plt.title('4 Input Neurons, 6 Hidden Neurons, 1 Output Neurons')
plt.xlabel('epochs')
plt.ylabel('mean squared error')
plt.plot(iterations, mse)
plt.show()
