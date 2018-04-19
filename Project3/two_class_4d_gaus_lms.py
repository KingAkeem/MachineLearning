import itertools
import math
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
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
eita = 0.1  # Learning Rate
max_iter = 100

y = np.zeros((len(d),))  # Predicted responses
e = np.zeros((len(d),))  # Errors

error = 1
iteration = 0

x_coor = list()
y_coor = list()
while error != 0 and iteration < max_iter:

    # Adjusting weights based on error and learning rate
    for i in range(len(d)):
        y[i] = np.sign(np.dot(x[i], w[i]))
        e[i] = y[i] - d[i]
        if y[i] != d[i]:
            w[i] = w[i]+np.dot(eita*(d[i]-y[i]), x[i])

    iteration += 1
    eita = 0.1/iteration
    error = find_mse(e)

    x_coor.append(error)
    y_coor.append(iteration)

# Plotting Learning Curve
plt.title('Learning Rate of 0.1 with 100 iterations')
plt.xlabel('epochs')
plt.ylabel('mean squared error')
plt.plot(y_coor, x_coor)
plt.show()

"""
cnf_matrix = confusion_matrix(y, d)
np.set_printoptions(precision=2)
plt.figure()

plot_confusion_matrix(cnf_matrix, classes=np.array(['Class 1', 'Class 2']),
                      normalize=True, title='Learning Rate of .0001')
"""
