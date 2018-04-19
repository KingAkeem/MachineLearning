import numpy as np


x = list()
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

        x.append(x_row)
        y.append([y_row])

x = np.array(x)
y = np.array(y)
print(x, y)
