import matplotlib.pyplot as plt
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pprint import pprint

labels = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2,
    0: 'Iris-setosa',
    1: 'Iris-versicolor',
    2: 'Iris-virginica'
}

x = list()
y = list()
with open('../datasets/bezdekIris.data') as dataset:
    for line in dataset.readlines():
        data = line.replace('\n', '').split(',')
        x_data, y_data = data[:len(data)-1], labels[data[len(data)-1]]
        x_data = np.array([float(d) for d in x_data])
        x.append(x_data)
        y.append(y_data)

x = np.array(x)
y = np.array(y)
lda = LDA(n_components=2)
reduced_X = lda.fit(x, y).transform(x)
print(reduced_X)
plt.figure()
colors = ['r', 'b', 'g']
for color, label_num, label_name in zip(colors,
                                        list(labels.keys())[3:],
                                        list(labels.keys())[:3]):
    plt.scatter(reduced_X[y == label_num, 0], reduced_X[y == label_num, 1],
                color=color, label=label_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of Iris dataset')
plt.show()
