import matplotlib.pyplot as plt
import numpy as np
import math


def find_mse(err):
    total = 0
    for e in err:
        total += math.pow(e, 2)
    return total/len(err)


x = list()
d = list()

with open("../datasets/Two_Class_FourDGaussians.dat") as data:

        for line in data.readlines():
            fields = line.split(",")
            x_data, d_data = list(map(float, fields[:4])), float(fields[4])
            x.append(x_data)
            d.append(d_data)

d = np.array(d)  # Desired responses
x = np.array(x)  # Sample features

w = np.random.random_sample(x.shape)  # Weight vector
eita = 0.0001  # Learning Rate
mse = 100
max_iter = 100

y = np.zeros((len(d),))  # Predicted responses
e = np.zeros((len(d),))

error = 1
iteration = 0

x_coor = list()
y_coor = list()
while error != 0 and iteration < max_iter:
    for i in range(len(d)):
        y[i] = np.sign(np.dot(x[i], w[i]))
        e[i] = y[i] - d[i]
        if y[i] != d[i]:
            w[i] = w[i]+np.dot(eita*(d[i]-y[i]), x[i])

    iteration += 1
    eita = 0.0001/iteration
    error = find_mse(e)
    x_coor.append(error)
    y_coor.append(iteration)

plt.title('Learning Rate of 0.0001 with 100 iterations')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.plot(y_coor, x_coor)
plt.show()
