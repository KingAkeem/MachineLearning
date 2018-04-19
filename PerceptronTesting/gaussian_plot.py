import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pandas.tools.plotting import scatter_matrix

x = list()
y = list()
with open('../datasets/Two_Class_FourDGaussians.dat') as dataset:
    for line in dataset.readlines():
        data = line.split()
        x_data, y_data = np.array([float(d) for d in data[:len(data)-1]]), int(data[len(data)-1])
        x.append(x_data)
        y.append(y_data)
x = np.array(x)
y = np.array(y)

features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
gau_data = pd.DataFrame(data=x, columns=features)
gau_data["target"] = y

color_wheel = {1: "#0392cf",
               2: "#7bc043"
               }
colors = gau_data["target"].map(lambda x: color_wheel.get(x))
ax = scatter_matrix(gau_data, color=colors, alpha=0.8, figsize=(25, 25), diagonal='kde')
plt.show()
